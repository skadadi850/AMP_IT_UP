# CMD-AMP

Conditioned Masked Discrete Diffusion for Antimicrobial Peptide Design.
Submission to the [AMP Challenge](https://szczurek-lab.github.io/amp-challenge-website/)
(NeurIPS 2026 Competition Track). The implementation is the `ampx` package.

## Abstract

Antimicrobial peptides (AMPs) have emerged as promising therapeutic candidates, with advances in computational approaches and high-throughput screening accelerating their discovery and optimization. However, designing effective de novo AMPs requires addressing potency, selectivity, and toxicity. Existing generative approaches often rely on post-hoc filtering, which selects candidates only after generation, or indirect conditioning through latent representations from protein language models (PLMs), which may limit simultaneous control over these design objectives. Here, we introduce CMD-AMP, a masked discrete diffusion model conditioned on structural and functional features to directly generate AMP sequences. To explicitly learn amphipathic structural organization, CMD-AMP masks residues on the same face of an α-helix, directing the model to reconstruct these membrane-interacting surfaces. CMD-AMP further incorporates sequence-derived features as well as experimental data, including MIC and hemolytic activity, using observation masking. Across 50,000 generated peptides, no sequence exceeds 80% identity to any of 39,448 antibacterial peptides used for training, and a held-out MIC regressor predicts potency comparable to that of known AMPs. These results indicate that CMD-AMP can be used for de novo AMP generation for concurrent design constraints.

This work is submitted for wetlab validation. 

## Quick start

```bash
uv sync
uv run generate
```

This writes:

```
generate/library.fasta         50,000 designed peptides
generate/top.fasta             100 ranked candidates
docs/top500_overflow.fasta     ranked overflow list (500), used to replace invalid entries
```

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.10 or newer, network access on the
first run (see [Weights and downloads](#weights-and-downloads)). A GPU is used when available.

### PyTorch builds

A bare `uv sync` installs the CUDA 12.8 build of torch 2.11.0, which runs on any NVIDIA
driver that supports CUDA 12.8 or newer. Other builds are selected by dependency group:

| Group | Command | Use for |
|---|---|---|
| `cu128` (default) | `uv sync` | Drivers with CUDA 12.8 or newer, Turing through Blackwell |
| `cu126` | `uv sync --no-default-groups --group cu126` | Drivers with CUDA 12.6, and Volta GPUs (V100) |
| `cu130` | `uv sync --no-default-groups --group cu130` | Drivers with CUDA 13.0 or newer |
| `cpu` | `uv sync --no-default-groups --group cpu` | No GPU. Much slower |

The generator's transformer blocks run under bfloat16 autocast on CUDA. The conditioning
path, output head, guidance combination and sampling softmax stay in float32. On CPU everything
runs in float32.

## Options

| Flag | Default | Description |
|---|---|---|
| `--n-sequences` | `50000` | Library size |
| `--top-k` | `100` | Ranked candidates |
| `--seed` | `42` | Random seed |
| `--length` | `50` | Maximum peptide length in residues. Template flag; the competition ceiling and the default are both 50 |
| `--device` | auto | `cuda`, `cpu`, or omitted to resolve automatically |
| `--identity-ceiling` | `0.75` | Maximum Levenshtein ratio to any reference sequence. The organizers' validator fails above `0.80`; the default leaves margin |
| `--overflow-k` | `500` | Length of the ranked overflow list written to `docs/`. The organizers replace an invalid top-100 entry with the next valid candidate, so the ordering past 100 is used |
| `--out` | entry-point name | Output directory |
| `--kappa` | `1.5` | Inert. Risk aversion in the LCB ranking. The oracle is a single model, so `score_std` is 0 and `lcb` collapses to the mean; changing this flag does not change the output |

## Method

Data provenance and known biases are in [Data disclosure](#data-disclosure). The rebuild path for the harmonized activity table is in [data/processed/README.md](data/processed/README.md).

### Generator

A 58.9M-parameter bidirectional transformer with adaLN-Zero modulation (12 blocks, width 512,
8 heads, feed-forward width 2048) over a fixed 50-position canvas. The vocabulary is the 20
standard residues plus PAD, which is a learned terminus, and MASK, the absorbing state. Training
minimizes the continuous-time masked-diffusion objective, a cross entropy over masked positions
weighted by 1/t, in two stages:

1. Pretraining on 309,272 sequences: 39,448 AMPs from MLAMP as positives and 269,824 general small
   proteins from sORFdb as negatives, length-filtered to 8 to 50 residues and deduplicated.
2. Conditioned fine-tuning on 60,405 rows joining the MLAMP peptides to GRAMPA MIC values and to
   HC50 values predicted with HemoPI-2.

### Helical-face corruption

The forward process mixes three corruption structures per training example (default weights 0.5,
0.25, 0.25):

| Mode | Masked set |
|---|---|
| independent | each position independently with probability t |
| span | one contiguous block of width t times the canvas |
| helical face | positions whose angle on a 100-degree-per-residue helical wheel falls inside a window of width t times 360 degrees, at a random phase |

An amphipathic helix places hydrophobic residues on one face and cationic residues on the other.
Masking one face forces the network to reconstruct that face from the complementary one. Only the
independent mode gives the exact continuous-time bound; span and face are auxiliary corruptions.
The face mode has a per-position masking probability of exactly t. The span mode matches t only on
average across positions, and edge positions are masked less often than central ones.

The reverse process unmasks in an order matched to the corruption structure of the sample, so the
still-masked set keeps the joint shape the network was trained on. Committed positions are never
remasked. Forward passes at steps that commit no position are skipped.

### Assay-aware conditioning

| Axis | Encoding |
|---|---|
| species | categorical, 9 panel species |
| Gram class | categorical, derived from species |
| is_amp | categorical |
| MIC | log-scaled value plus a censoring flag (exact or right-censored) |
| HC50 | log-scaled value plus a provenance flag (measured or predicted) |
| Gram selectivity | log10 ratio of Gram-positive to Gram-negative MIC |
| length | normalized, 8 to 50 |
| charge per residue, hydrophobic moment, GRAVY | normalized values |
| cysteine class | categorical |

Continuous axes carry an observation mask, so missing assay values are unobserved, not zero. Each
axis is dropped independently during training, and sampling uses classifier-free guidance in logit
space (weight 2.0, 128 steps, temperature 1.0). Axes exempt from dropout are carried into the
unconditional branch, because their null embedding is never trained.

At generation the conditioning grid is the cross product of the nine panel species (weighted 15:5
Gram-negative to Gram-positive), target MIC of 2, 4 and 8 uM, target HC50 of 128 and 256 uM, three
selectivity targets, and three levels each of charge per residue, hydrophobic moment and GRAVY.
Cysteine class is fixed to zero cysteines.

### Constraints

- Length is drawn to match the reference length distribution and pinned exactly.
- Cysteine count is capped in logit space, so no sample can exceed two cysteines.
- Every library sequence uses the 20 standard residues, has length 8 to 50, is unique, has no exact
  match in the reference set, and stays under the Levenshtein ratio ceiling against all 39,448
  reference sequences.

### Ranking oracle

An XGBoost regressor predicts log10 MIC from 9 physicochemical descriptors, species and Gram
indicators, and ESM-2 embeddings (`esm2_t12_35M_UR50D`, 480 dimensions). It is fit on 39,117
GRAMPA MIC rows across 5,602 peptides with a cluster-disjoint MMseqs2 split at 40% identity
(2,824 clusters). The checkpoint holds 1,165 trees, early-stopped at iteration 1,064, and inference
uses the first 1,065 explicitly.

Candidates are ranked by the strain-weighted mean predicted log10 MIC across the challenge panel,
lowest first:

| Species | E. coli | P. aeruginosa | K. pneumoniae | A. baumannii | S. enterica | E. cloacae | S. aureus | E. faecalis | B. subtilis |
|---|---|---|---|---|---|---|---|---|---|
| Strains | 5 | 3 | 2 | 2 | 2 | 1 | 2 | 2 | 1 |

E. faecium is folded into E. faecalis. A candidate must be predicted at or below 16 uM against at
least one Gram-negative and one Gram-positive species, and ineligible candidates rank below all
eligible ones. The fraction of the panel predicted at or below 16 uM is reported as a diagnostic
only.

### Selection

Greedy selection from the ranked library under four hard constraints:

1. Levenshtein ratio of at most the identity ceiling to every reference sequence, checked
   exhaustively.
2. Pairwise Levenshtein ratio of at most 0.60 between selected candidates, relaxed in steps of
   0.05 only if the list cannot be completed. The ladder in `ranking.py` stops at 0.75.
3. At most 20% of the list may have length 38 to 50, enforced as a running share at every
   prefix, so the bound holds for the first 50 as well as the first 100.
4. Eligibility floor from the oracle section above.

No sequence is hand-selected, edited or removed by inspection. Selection runs once at length 500,
and the submitted 100 are the first 100 of that ranking.

## Weights and downloads

| Artifact | Location | Handling |
|---|---|---|
| `masked_diffusion_best.pt`, `mic_regressor.json` | GitHub Release `weights-v1` | Downloaded on first use into `checkpoint/`, verified against `checkpoint/SHA256SUMS` |
| Small checkpoint files (metadata, species vocabulary, length calibration) | `checkpoint/` | Committed, verified in place |
| ESM-2 `esm2_t12_35M_UR50D` | Hugging Face Hub, pinned to a fixed commit | Downloaded on first use, then read from the local cache with no network calls |

The Release assets are not tracked in git, because an LFS pointer clones as a small text file on
machines without git-lfs. A truncated or altered download fails the checksum and is removed.
`uv run generate` fetches whatever is missing. To fetch ahead of time or from a mirror:

```bash
uv run python scripts/fetch_weights.py
AMPX_WEIGHTS_BASE_URL=<mirror url> uv run generate
```

## Reproducibility

- The seed is 42. Every model is loaded before any RNG is seeded, sampling uses a dedicated
  generator with a derived seed per round, and the library is written in sorted order.
- Two runs on the same machine, driver and locked environment produce byte-identical files.
- Outputs are not guaranteed to match across different GPU models, drivers or CUDA builds.
  Kernel selection changes low-order bits of the logits, which can change a sampled residue and
  every draw after it in that batch. CPU and GPU runs differ for the same reason.
- `uv.lock` pins every dependency, including the torch build.

## Validating before you submit

```bash
uv run generate
rm -rf submission
uv run python scripts/verify_submission.py https://github.com/skadadi850/AMP_IT_UP
```

The verifier clones the pushed repository, so push first. It refuses to run while `submission/`
exists, so remove that directory before each check.

## Layout

```
src/ampx/
  generate.py        entry point: conditioned sampling, ranking, selection
  compliance.py      hard-constraint gate (alphabet, length, novelty ceiling)
  ranking.py         top-100 selection with identity and length constraints
  predictor.py       MIC regressor and its feature construction
  weights.py         checksummed weight download
  models/            generator, conditioning encoder, sampler, ESM-2 encoder,
                     feature computation, exhaustive novelty index
scripts/             dataset build, training, calibration, evaluation
checkpoint/          weights and calibration files (large files fetched from the Release)
data/                organizers' exclusion set (antibacterial.fasta)
docs/                ranked overflow list, written by uv run generate
results/             calibration and selection outputs
generate/            library.fasta and top.fasta
```


## Data disclosure

**Generator pretraining (309,272 sequences).** Positives are 39,448 antibacterial peptides from MLAMP_db. The generator was trained on the sequences it must stay novel against, so novelty is enforced at sampling and selection, not by holding that set out. Negatives are 269,824 real translated small ORFs from sORFdb (Zenodo record 10688271, redistribution terms not confirmed). Of 34,007,166 protein records, 318,268 fall in the 8 to 50 residue range, and an AMP-like filter removed 48,435 of them, leaving 269,833 in the negatives file, of which 269,824 entered the pretraining corpus. The filter script is not included in this repository. SmProt v2 was downloaded but never used, and no SmProt sequence entered any corpus.

**Fine-tuning (60,405 rows).** The MLAMP peptides joined to GRAMPA MIC values and to HC50 values, measured where HemoPI-2 has a measurement and predicted otherwise.

**MIC labels.** GRAMPA aggregates four databases. Of 41,500 rows surviving the ingest filter, DBAASP contributes 32,880 (CC BY 4.0), DRAMP 4,459, APD 3,554 and DADP 607 (terms not confirmed for the last three). The harmonized table holds 39,117 MIC rows over 5,602 peptides and 659 species. The filter keeps peptides with modifications and excludes unusual modifications, so 37.4% of the rows are C-terminally amidated. The MIC regressor was retrained under a cluster-disjoint split after an earlier split leaked between train and test; the superseded model is not shipped.

**HC50 labels.** 1,908 measured values from the HemoPI-2 repository (GPL-3.0, not redistributed here) and 39,448 predicted values from running HemoPI-2's regression script over the positives. The predicted values are model output, so the HC50 axis steers generation toward HemoPI-2's predictions, not toward measured hemolysis, and the provenance flag lets the model tell the two apart. That script truncates sequences longer than 40 residues before featurizing, so 1,367 predictions (3.47%) describe a truncated peptide.

**Not redistributed.** The harmonized activity table, because its sources have differing or unconfirmed terms. Every source is public. [data/processed/README.md](data/processed/README.md) gives the rebuild command, retrieval dates, expected row counts and the checksum of the table used for training.

**Manual intervention.** None. No sequence was hand-selected, edited or removed by inspection. Human decisions were the conditioning targets, the ranking key, the length cap and the filter thresholds.

## Data sources

1. Marczak, B., Bocian, A., and Lyskowski, A. (2026). MarLys AMP database, MLAMP_db, version 3.
   Mendeley Data. https://doi.org/10.17632/w4hb5grjwb.3
2. Hahnfeld, J. M., Schwengers, O., Jelonek, L., et al. (2025). sORFdb: a database for sORFs,
   small proteins, and small protein families in bacteria. BMC Genomics, 26, 110.
   https://doi.org/10.1186/s12864-025-11301-w
3. Witten, J., and Witten, Z. (2019). Deep learning regression model for antimicrobial peptide
   design. bioRxiv, 692681. https://doi.org/10.1101/692681
4. Rathore, A. S., Kumar, N., Choudhury, S., et al. (2025). Prediction of hemolytic peptides and
   their hemolytic concentration. Communications Biology, 8, 176.
   https://doi.org/10.1038/s42003-025-07615-w

## Use of AI assistants

Large language models were used to help write code and documentation for this repository.

## License

MIT. See [LICENSE](LICENSE).
