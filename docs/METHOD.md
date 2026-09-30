# Method

> Fill this in as you build. It is a **submission requirement**: the abstract,
> training-data disclosure, filters applied, and selection procedure are all
> required for benchmark participation, and the LLM-assistance disclosure is
> required by the NeurIPS Main Track Handbook.
##FInal 


## Abstract

We design antimicrobial peptides with a masked discrete diffusion model
operating directly on amino-acid tokens over a fixed 50-position canvas, with
no latent bottleneck and no autoregressive decoder, so there is no
reconstruction error between what the model scores and what it emits.
Generation is steered by continuous physicochemical conditioning — length,
charge per residue, hydrophobic moment, hydrophobicity, cysteine content —
alongside assay-derived MIC, HC50 and Gram-selectivity axes and a categorical
species axis, injected through adaLN-Zero modulation. Axes are dropped
independently per sample during training and each carries an explicit
observation mask, so a partially specified request is a state the model was
trained on rather than a point off the manifold. This matters because potency
labels are sparse while sequence-derived properties are dense. Sampling
sweeps a length grid matched to the reference length distribution, with the
PAD-logit bias calibrated against the shipped checkpoint.

The 50,000-member library is filtered for the competition's structural
constraints, deduplicated, and screened exhaustively so no member exceeds 80%
Levenshtein identity to any of the 39,448 reference antibacterials.

The top 100 are ranked by predicted breadth across the competition's
20-strain panel rather than potency against one organism. An XGBoost MIC
regressor over ESM-2 embeddings and physicochemical descriptors scores each
candidate against every panel species, weighted by strain count; candidates
are ordered by the **strain-weighted mean predicted log₁₀ MIC** across the
panel, under a hard floor requiring activity against at least one
Gram-negative and one Gram-positive strain. The fraction of the panel
predicted at or below 16 µM — the Overall Success Rate the competition
scores — is computed and reported as a diagnostic rather than used as the
key, because it saturates at 1.00 among the candidates actually competing
for the hundred places and so cannot order them; see
[The oracle is not extrapolating](#the-oracle-is-not-extrapolating-and-the-ranking-key-reflects-that).
The key is a breadth key either way: it is the strain-count-weighted mean
over nine species, not potency against one. The
regressor is validated on a cluster-disjoint 40%-identity split (Spearman
0.533) and used as a breadth estimator, not a fine-grained ranking.

## Training data

See [../data/README.md](../data/README.md) for the full provenance table.

- **Generator corpus:** trained in two stages over the sources tabulated in
  [Generator corpus, by stage](#generator-corpus-by-stage) below. Stage 5a
  pretrains on 309,272 sequences; stage 5b fine-tunes on 60,405 conditioned
  rows.
- **Predictor corpus:** the MIC arm of `activity_harmonized.csv` — 39,117 rows
  over **5,602 unique peptides** and 659 species, from GRAMPA. That table is
  not redistributed here; see
  [../data/processed/README.md](../data/processed/README.md) for the rebuild
  command, the upstream sources with retrieval dates, the expected row counts
  and a checksum of the table the shipped models were trained on. The
  row-to-peptide ratio is the
  reason the split is clustered on sequence: one peptide assayed against many
  species must not straddle the boundary. HC50 rows (41,356: 39,448
  `hemopi2_predicted`, 1,908 `hemopi2_measured`) train the generator's
  conditioning, not the MIC regressor.
- **Splits:** MMseqs2 clustering at 40% identity, cluster-disjoint. This holds
  for the shipped regressor, but was not true of the first attempt: an earlier
  run's split silently degenerated and the model was retrained. See
  [A leaking split, caught and corrected](#a-leaking-split-caught-and-corrected)
  for what happened and both sets of numbers.

### Generator corpus, by stage

The generator is not trained once. Stage 5a pretrains a masked diffusion model
to tell antimicrobial peptides from general small proteins; stage 5b takes
those weights and fine-tunes them on the subset that carries measured
activity, conditioned on potency, haemolysis, species and selectivity.

**The MLAMP peptide set is the positive class in both stages.** What changes
between them is not the peptides but the labels attached to them: stage 5a
adds a contrasting negative class, and stage 5b adds activity measurements
that turn the same 39,448 peptides into 60,405 conditioned rows.

| | Stage 5a — pretrain | Stage 5b — selectivity fine-tune |
|---|---|---|
| positives | MLAMP, 39,448 peptides (`is_amp` index 2) | the same 39,448 peptides |
| negatives | 269,824 general small proteins (`is_amp` index 1) | none |
| rows | 309,272 | 60,405 |
| activity labels | none | GRAMPA (MIC), HemoPI2 (HC50) |
| initialised from | scratch | stage 5a's best checkpoint |
| learning rate | 6e-4 | 3e-5 |
| epochs | 200 | 150 |
| class balancing | `--balance-is-amp`, ~3:1 negative:positive | `--balance-species` |

Sources, named in full:

- **MLAMP** (`raw/mlamp_positives.csv`) — 40,171 rows, of which 39,448 are
  antimicrobial (`label == 1`) and 723 are annotated non-antimicrobial
  function and excluded from the positive class. The 39,448 are exactly the
  reference exclusion set in `data/antibacterial.fasta`, which aggregates
  DBAASP, dbAMP and APD.
- **General small proteins** (`processed/negatives.fasta`) — 269,833 real
  sequences with GenBank accessions, drawn from **sORFdb alone**,
  length-filtered to 8–50 residues; 269,824 survive deduplication into the
  pretrain table. These are real translated open reading frames, **not**
  shuffled, random or mutated decoys. Earlier drafts of this document credited
  this class to "sORFdb/SmProt"; see [SmProt v2 was obtained but never
  used](#smprot-v2-was-obtained-but-never-used).
- **GRAMPA** (`raw/grampa.csv`) — MIC measurements, filtered to entries
  without unusual chemical modifications, since the generator emits unmodified
  linear peptides.
- **HemoPI2** (`raw/hemopi2_measured.csv`, `raw/hemopi2_predicted.csv`) —
  HC50. The predicted table dominates; see
  [HC50 conditioning steers toward predicted labels, not measurements](#hc50-conditioning-steers-toward-predicted-labels-not-measurements).

#### The pretrain validation set contained no negatives

Both stages hold out fold 0 of the MMseqs2 40% clustering. The clustering
covers the MLAMP peptides but not the sORFdb small proteins, and
`select_train_val` keeps a sequence with no assigned fold in training rather
than dropping it. In stage 5a this means all 269,824 general proteins went
into training and validation was ~7,787 held-out AMP clusters only.

Early stopping in the pretrain was therefore driven by reconstruction of
held-out AMPs, with no held-out negatives to detect a model that had stopped
distinguishing the two classes. This does not affect the shipped library's
validity — stage 5b conditions on `is_amp` index 2 throughout and the
compliance gate is independent of it — but the pretrain's validation curve
should not be read as evidence about the general-peptide class.

### SmProt v2 was obtained but never used

Earlier revisions of this document, and the machine-readable source registry
in `src/ampx/data/sources.py`, described the general small-protein class as
drawn from "sORFdb/SmProt" and declared SmProt v2 as a pooled second source.
**That was wrong. The class is sORFdb alone.** SmProt v2 was downloaded and
staged, the source entry was written in anticipation of pooling, and the
pooling step was never implemented. No SmProt sequence entered any corpus.

The record is corrected here rather than quietly amended, because a training
data disclosure that names a database the model never saw is a defect of the
same kind as omitting one it did see.

How it is known, from the build rather than from recollection:

- `scripts/hpc/02_negatives.sbatch` in the research repository reads exactly
  one raw input, `raw/sorfdb_proteins.fasta.gz`, and contains no pooling,
  concatenation or second-input step.
- The job log closes arithmetically with no room for a second database:
  318,268 sORFdb records survive the 8–50 residue filter and deduplication,
  48,435 are removed by the MMseqs2 screen against the reference
  antibacterials, and 318,268 − 48,435 = **269,833**, which is exactly the
  size of `processed/negatives.fasta`.
- A search for "smprot" across the entire research repository returns nothing.
  The identifier appears in one place in either repository: its own
  declaration in `sources.py`.
- The input file is the sORFdb Zenodo protein release (`sorfdb.faa.gz`), whose
  records carry sORFdb's `>GenBank|<accession>|<protein_id>` headers. No
  SmProt-style identifier occurs in it.

The staged SmProt v2 download remains on the cluster and is untouched by any
script in the pipeline. Its licence terms were never confirmed, so removing it
from the disclosure also removes an unconfirmed-terms source from the
submission rather than introducing one.

This correction does not change the corpus, the checkpoints or any reported
number. The 269,833 negatives and the 269,824 that survive deduplication into
the pretrain table are the same sequences they always were; only their stated
provenance changes.

### The generator was trained on the set it must stay novel against

This is the single largest tension in the method and it is declared here
rather than left for a reader to infer.

`data/antibacterial.fasta` serves two conflicting purposes. It is a curated,
high-quality AMP set — the best available training signal — and it is also the
novelty constraint: no submitted top-100 sequence may exceed 80% identity to
any of its members. Training on it pushes generations toward exactly the
sequences the submission must then stay away from.

**The decision was to train on it and apply novelty pressure at sampling
time**, rather than hold it out. Its 39,448 sequences are the positive class
in both stage 5a and stage 5b; they are byte-for-byte the `label == 1` rows of
MLAMP. The alternative — holding it out and training on the other positive
sources — was rejected because it would have discarded the majority of the
curated positives and left the conditioning axes with far less measured
activity to bind to.

Three things make that defensible, and all three are structural rather than
incidental:

1. **The screen is exhaustive, not a shortlist.** Every top-100 candidate is
   compared against all 39,448 reference sequences. An approximate
   nearest-neighbour shortlist can miss a true high-identity match, and one
   sequence over the ceiling invalidates the entry.
2. **The screen is a hard gate, not a ranking penalty.** A candidate over 0.80
   is removed, not down-weighted.
3. **The ceiling is applied to the whole 50,000-member library**, not only to
   the top 100, because near-exact matches to the reference databases are
   scored against the library as a population.

**This must be re-verified for whichever run ships.** The maximum identity of
the selected 100 against the reference set is a measured property of a
specific library, not a guarantee of the method, and it is reported with the
run that produced the submitted files. For the 2026-09-28 generation run the
selected 100 sat at maximum 0.745, median 0.550, minimum 0.427 — clear of the
ceiling, but with no margin to spare at the top of that range. If the library
is regenerated, this number is regenerated with it.

### Harmonized activity table

`activity_harmonized.csv` is the table `scripts/build_dataset.py` writes and
the MIC regressor trains on, with columns `sequence, species, strain, endpoint,
value_um, censored, source`.

**It is deliberately not redistributed in this repository.** It aggregates
sources whose redistribution terms differ and, in three cases, could not be
confirmed:

- 1,908 measured HC50 rows are HemoPI-2's own dataset, released under
  **GPL-3.0**, which cannot ship under this repository's MIT licence.
- 39,117 MIC rows come from GRAMPA, itself an aggregation: DBAASP (CC BY 4.0)
  but also DRAMP, APD and DADP, whose redistribution terms are **not
  confirmed**.

An earlier revision committed the table whole, on the reasoning that a
disclosure a reader cannot check is not much of a disclosure. That reasoning
still holds; the conclusion was wrong. Shipping the file under MIT would have
asserted a redistribution right over third-party data that we had not verified
and, for HemoPI-2, do not have. Removing only the GPL-3.0 rows would have
traded a known problem for an unverified one.

**A rebuild path is provided instead, and it gives a reviewer more than the
file would.** `data/processed/README.md` carries the exact rebuild command,
every upstream source with DOI or URL and retrieval date, the expected row
counts, and a SHA-256 checksum of the table the shipped models were actually
trained on, so a rebuild can be compared byte-for-byte rather than merely
looking plausible. The source specification
`data/activity_sources.json` — our own file, carrying the per-source column
mappings and row filters and no third-party data — **is** committed, which is
what makes the rebuild reproducible rather than approximate.

Every upstream source is public. Nothing here is withheld; what is withheld is
a redistribution we are not entitled to make.

Expected row counts, recorded so a rebuild is checkable:

| Partition | Rows |
|---|---:|
| Total | 80,473 |
| `endpoint == "mic"` | 39,117 |
| `endpoint == "hc50"` | 41,356 |
| `source == "grampa"` | 39,117 |
| `source == "hemopi2_measured"` | 1,908 |
| `source == "hemopi2_predicted"` | 39,448 |

Replicate measurements are collapsed upstream by geometric mean in log10
space, so a peptide assayed repeatedly against one species contributes one
row and cannot dominate the fit.

### Version pinning of shipped checkpoints

The inference dependencies in `pyproject.toml` pin `torch`, `transformers` and
`xgboost` to exact versions, because each determines how a shipped checkpoint
is read rather than merely how fast it runs:

- **`xgboost==3.2.0`** — the version recorded in `checkpoint/mic_regressor.json`'s
  own `version` field. That model holds 1,165 trees with `best_iteration=1064`,
  so 100 trees sit past the early-stopping point. The raw `Booster.predict`
  that inference uses defaults to predicting with **every** tree, including
  those 100, which gives different MIC estimates and therefore a different top
  100 from the truncated predictions the model was selected under. We pin the
  version and name the tree count explicitly rather than relying on either
  default. See [Tree count at prediction
  time](#tree-count-at-prediction-time).
- **`transformers==5.17.0`** — produced the 480-dimensional ESM-2 embedding
  block. The regressor's 510 input features are 9 descriptors, 16 species
  indicators, 5 Gram indicators and those 480 embedding dimensions, so a
  different encoder yields a feature matrix of the wrong width or the wrong
  meaning.
- **`torch==2.11.0`** — trained `checkpoint/masked_diffusion_best.pt`.

### Encoder identity, recovered after the fact

`checkpoint/mic_regressor_meta.json` now carries
`"esm_model": "facebook/esm2_t12_35M_UR50D"`. This key was **not written at
training time** — the training script recorded only `uses_embeddings` and
`embedding_dim`. It was recovered from the training job's own log line
(`scripts/hpc/32_mic_regressor.sbatch`, which echoes
`ESM=facebook/esm2_t12_35M_UR50D`) and added to the metadata by hand rather
than by retraining. Two independent facts corroborate it: that model's hidden
size is 480, and the regressor's 510 input features decompose exactly as
9 descriptors + 16 species indicators + 5 Gram indicators + 480 embedding
dimensions.

Training and inference reach those embeddings by different routes but the same
model and the same pooling path. The training run consumed a precomputed array
(`mic_esm.npy`, written by `scripts/embed_sequences.py`), whereas `Oracle`
encodes sequences on the fly through `ampx.models.encoder.ESM2Encoder`. Both
take the mask-weighted mean over `last_hidden_state` of
`esm2_t12_35M_UR50D`, over real tokens only. The encoder never uses the
`pooler.dense` layer, which Hugging Face reports as newly initialized when
loading these weights; it is constructed but does not contribute to any value
we use.

### A leaking split, caught and corrected

The regressor shipped here was retrained on 2026-09-27. The history matters
and is recorded rather than smoothed over.

The original run of 2026-09-13 intended a cluster-disjoint split: hold out
whole MMseqs2 clusters at 40% identity, so no held-out peptide has a close
relative in training. It did not get one. The sequence-to-cluster mapping
matched nothing, every sequence fell through to the singleton fallback, and
all 5,602 sequences became their own cluster. Holding out 20% of "clusters"
was then identical to holding out 20% of sequences, and because near-identical
peptides recur across AMP databases, relatives straddled the split freely.

The defect is visible in that run's own output: every one of the 5,691
held-out rows in its `mic_holdout_predictions.csv` carries a `cluster` value
equal to its own sequence. It was diagnosed and guarded on 2026-09-21, by a
check that refuses to run when more than a small fraction of rows are missing
from the cluster table — the singleton fallback is a safety net for
stragglers, not a mode of operation. That guard would now stop the Sep 13
configuration outright.

The two runs differ only in the split. Same cached ESM-2 embeddings, same
hyperparameters, same seed, same holdout fraction:

| Run | Split actually used | Clusters | Held-out Spearman | MAE |
|---|---|---|---|---|
| 2026-09-13 | degenerate: 5,602 singletons, effectively random by sequence | 5,602 | 0.6183 | 0.4427 |
| 2026-09-27 (**shipped**) | cluster-disjoint, MMseqs2 40% identity | 2,824 | **0.5331** | 0.5086 |

The drop from 0.618 to 0.533 is the correction, not a regression: roughly
0.085 of Spearman in the original figure was identity leakage rather than
predictive skill. The corrected model clears the preregistered gate of 0.4,
and its label-shuffled leakage control reads 0.0314, far below the 0.15 ceiling.

The superseded model is retained locally, with its own metadata, so the
comparison above can be re-run rather than taken on trust. It is deliberately
not committed — it is evidence for a claim made here in prose, not something
needed to reproduce the submission, and nothing loads it. Available on request.

**On the ceiling this is measured against.** We attempted to bound how much
headroom 0.5331 leaves by measuring how well independent sources agree with
each other on the same peptide, and could not: every MIC record in
`activity_harmonized.csv` comes from a single upstream source (GRAMPA), so
there are no cross-source pairs to correlate. The within-source repeats that
do exist are not a substitute — their median spread is 0.0013 log₁₀ units,
which means they are overwhelmingly the same measurement recorded against
different strain labels rather than independent assays. A ceiling derived from
them would be biased towards perfect agreement and would overstate the model's
remaining headroom, so no ceiling is reported here. The held-out Spearman
should be read as an uncalibrated number: it is not known how much of the
residual is irreducible assay noise.

### Tree count at prediction time

`Oracle` predicts with an explicit `iteration_range=(0, 1065)`, read from
`predict_n_trees` in the checkpoint metadata, instead of relying on any
library default. The shipped model holds 1,165 trees and was early-stopped at
`best_iteration=1064`, so 100 trees sit past the point it was selected at.

This is load-bearing because the two XGBoost prediction APIs disagree, and we
changed which one we use. Verified on both checkpoints under xgboost 3.2.0:

| API | Default behaviour |
|---|---|
| `XGBRegressor.predict` (used during training) | truncates at `best_iteration + 1` |
| `Booster.predict` (used by `Oracle`) | uses **every** tree |

`Oracle` was switched from `XGBRegressor` to a raw `Booster` so that inference
need not ship scikit-learn. Without an explicit `iteration_range` that switch
would silently have started scoring with all 1,165 trees — predictions the
model was never selected under. Scoring the shipped model's own 5,751 held-out
rows:

| Variant | Held-out Spearman | Max abs. difference from recorded predictions |
|---|---|---|
| `Booster.predict()` default (all 1,165) | 0.5337120793 | 7.2e-02 |
| `iteration_range=(0, 1065)` | 0.5330712034 | 1.1e-07 |

Only `(0, 1065)` reproduces the recorded predictions, to float32 rounding,
against the recorded `holdout_spearman` of 0.5330633633. The number is stored
with the checkpoint, so it travels with the artifact it describes rather than
being a constant in the code.

## Generative model

- **Architecture:** conditioned masked discrete diffusion. A 12-layer
  bidirectional transformer, `d_model` 512, 8 heads, feed-forward 2048,
  dropout 0.1 — 58,876,438 parameters. Trained to denoise a partially masked
  peptide rather than to generate left to right, so every position is
  predicted in the context of every other.
- **Representation:** a fixed 50-position canvas over the 20 canonical
  residues plus `MASK` and `PAD`, which is the competition's length ceiling.
  Length is a property of where `PAD` begins. Corruption during training mixes
  three masking modes — i.i.d., span and **helical face** — weighted
  0.5 / 0.25 / 0.25; see
  [Helical-face masking](#helical-face-masking) below.
- **Conditioning:** a 128-dimensional embedding summed from seven continuous
  axes (`log_mic`, `log_hc50`, `length_norm`, `charge_per_res`, `mu_h`,
  `gravy`, `gram_selectivity`) and the categorical axes, of which `species`
  carries a 125-entry vocabulary. Axes absent from a training row are
  dropped out so the model learns to sample without them; `is_amp` is exempt
  from that dropout, which has a consequence documented in
  [A caught-and-fixed defect](#a-caught-and-fixed-defect-the-first-library-was-sampled-unconditionally).
- **Sampler:** iterative unmasking over 128 steps from a fully masked canvas,
  revealing positions in an order matching the training corruption mixture
  (`reveal="structured"`). Length is pinned by the condition
  (`free_length=False`) rather than left to the model to place `PAD`, which
  is what restored the upper tail of the length distribution; see
  [Filters applied to the library](#filters-applied-to-the-library).
  Classifier-free guidance is applied at sampling time.
- **Checkpoint:** `checkpoint/masked_diffusion_best.pt`, the stage-5b
  selectivity fine-tune. Not the stage-5a pretrain and not the earlier
  `conditioned/` run.

### Helical-face masking

Standard masked diffusion corrupts positions independently. That is the right
process for a generic sequence, but it ignores the one structural regularity
that makes an antimicrobial peptide work: amphipathic AMPs fold into an
α-helix and segregate hydrophobic residues onto a single membrane-facing
surface. An α-helix advances about **100° per residue**, so that surface is a
set of positions periodic in `i` — never a contiguous block. Independent or
span corruption therefore almost never asks the model to reconstruct one face
from the other.

The forward process mixes three corruption structures per sample:

| mode | weight | structure |
|---|---:|---|
| `iid` | 0.50 | each position masked independently |
| `span` | 0.25 | one contiguous block, SpanBERT-style |
| `face` | 0.25 | one helical face: positions whose angle `(i × 100°) mod 360` falls inside a window of width `t × 360°` |

**Every mode holds the per-position marginal masking rate at `t`.** Only the
joint distribution changes, so the diffusion schedule and its loss weighting
are untouched. Of the three, only `iid` makes the 1/t-weighted loss the exact
NELBO; `span` and `face` trade that exactness for an inductive bias toward the
structure the peptides actually have, and this is stated rather than glossed.

**Sampling mirrors the corruption.** `reveal="structured"` assigns a
per-position priority governing the order the reverse process unmasks, so the
still-masked set keeps the shape of the corruption the model trained on: a
contiguous remainder for `span`, and for `face` a remainder whose priority
falls with angular distance from a random phase at 100° per residue. The
default uniform reveal does not: for the 50% of training samples corrupted as
span or face, a uniformly-unmasking reverse process never encounters the joint
the forward process produced. That forward/reverse mismatch was a recorded
defect and `reveal="structured"` is the fix; the shipped run uses it.

A `face_only` ablation — the same architecture trained with face masking
alone — exists in the research repository and is **not** shipped. It is named
here because a length calibration fitted against it leaked forward once:
`checkpoint/length_calibration.json` was fitted on `face_only` weights, which
are short-biased, and was briefly carried over to the shipped stage-5b
checkpoint, which is long-biased. Refitting against the shipped weights gave
`pad_bias = +0.50` (`checkpoint/length_calibration_stage5b.json`). The shipped
path then stopped depending on the value at all: generation pins length from
the condition (`free_length=False`) and passes `pad_bias = 0.0`, so the
calibration is recorded for provenance and never applied.

## Surrogate predictors

**There is no ensemble.** A single XGBoost regressor predicts log10 MIC for a
(peptide, species) pair, and the submission says so rather than implying
breadth it does not have. One consequence is recorded under
[Top-100 selection procedure](#top-100-selection-procedure): the `kappa`
uncertainty term in the ranking code is inert, because the standard deviation
across a one-member ensemble is zero. It is left in place and documented
rather than replaced with a fabricated uncertainty.

- **Member:** `checkpoint/mic_regressor.json`, XGBoost, `reg:squarederror`,
  scored with an explicit `iteration_range` of 1,065 trees. See
  [Tree count at prediction time](#tree-count-at-prediction-time) for why that
  number is pinned in the metadata rather than inferred.
- **Features:** 9 sequence descriptors, one-hot species (with an `other`
  catch-all), one-hot Gram class, and a 480-dimensional mean-pooled ESM-2
  embedding (`facebook/esm2_t12_35M_UR50D`, revision pinned). The encoder
  identity is stored in the checkpoint metadata; see
  [Encoder identity, recovered after the fact](#encoder-identity-recovered-after-the-fact).
- **Calibration:** none. The model is used as a ranking key over predicted
  log10 MIC, not as a calibrated probability, so no post-hoc calibration is
  applied and none is claimed.
- **Held-out performance** (cluster-disjoint, MMseqs2 40%, 2,824 clusters):
  Spearman **0.533**, MAE **0.509** log10 units. FPR, Prec@100 and AUPRC are
  not reported because they are classification metrics and this is a
  regressor; there is no threshold in the pipeline to compute them at.

An MAE of 0.509 log10 units is a factor of about 3.2 in concentration. This is
a coarse enrichment filter, not a fine ranking, and the top-100 ordering should
be read with that resolution in mind.

## Conditioning, and what it does and does not demonstrate

Generation is conditioned on a 4,374-cell grid crossing the nine panel
species (weighted by strain count, reproducing the panel's 15:5
Gram-negative:Gram-positive split), target MIC (2, 4, 8 µM), target HC50
(128, 256 µM), Gram selectivity (−0.5, 0.0, +0.5 log₁₀ ratio), and
normalized charge per residue, hydrophobic moment and GRAVY swept across the
ranges the training corpus covers. Length is drawn separately to match the
reference length distribution and enforced as a hard constraint.

### A caught-and-fixed defect: the first library was sampled unconditionally

An earlier full generation run passed an empty conditioning request. Every
axis fell through to its unknown index or a zero observation mask, leaving
length as the only axis carrying signal, so a selectivity-conditioned model
was sampled as though it were unconditional.

On the `is_amp` axis this was worse than neutral. That axis is exempt from
conditional dropout, which means its index-0 "unknown" embedding row never
receives a gradient anywhere in training — the run was feeding the model an
untrained, randomly initialised vector on the one axis it always saw. Index 1
is genuinely trained: the pretraining corpus holds 269,824 general peptides
against 39,448 AMPs, so a real general-versus-AMP contrast exists, and the
fine-tuning corpus is entirely index 2. The shipped run sets `is_amp` to the
validated-AMP value.

The same run also lost the whole upper length band. It sampled with a free
length and a positive PAD-logit bias, which lets the model end sequences
early and overrides the requested length: 12.0% of requests were for peptides
longer than 29 residues and none were produced, against a reference set that
runs to 50. Length is now a hard constraint, verified to reproduce requested
lengths of 10, 25 and 45 exactly.

### HC50 conditioning steers toward predicted labels, not measurements

The `log_hc50` axis is observed on every training row, but 53,806 of those
60,405 values (89%) are **predicted** rather than measured; only 6,599 carry
an assay measurement. Conditioning toward non-hemolytic therefore steers
toward a hemolysis predictor's output. We use it because the Optimal
Selectivity category is scored on HC50 and no other axis reaches it, but the
resulting library should be understood as enriched for peptides a predictor
considers non-hemolytic, which is a weaker claim than low measured
hemolysis.

**The predictor is HemoPI-2** (Rathore et al., *Commun Biol* 8:176, 2025). The
53,806 predicted values were produced by running its own
`hemopi2_regression.py` over all 39,448 MLAMP positives, so the `log_hc50`
axis inherits that model's assumptions wholesale. Two consequences follow, and
both matter more for the Optimal Selectivity category than anywhere else:

- **HemoPI-2 cannot be used to evaluate this axis.** Scoring our generated
  library with HemoPI-2 would test only whether the generator reproduces the
  predictor it was conditioned on. Any evaluation of `log_hc50` must use the
  measured subset, or an independently computed correlate such as `gravy` or
  `mu_h`.
- **Predictions for our longest peptides are computed on truncated
  sequences.** HemoPI-2's `lenchk()` truncates inputs over 40 residues before
  featurization. **1,367 of the 39,448 peptides (3.47%)** exceed 40 residues,
  so their HC50 label describes a prefix, not the peptide. Our library reaches
  the full 50-residue ceiling, so this affects exactly the long tail the length
  conditioning was fixed to restore.

The predicted values themselves are model output, generated by us, and are not
covered by HemoPI-2's GPL-3.0 licence; HemoPI-2's own measured dataset is, and
is not redistributed here. See
[Harmonized activity table](#harmonized-activity-table).

### The MIC labels include modified peptides; our designs are unmodified

A larger bias than the HC50 one above, in the opposite endpoint. The ingest
filter for GRAMPA keeps rows whose source database records chemical
modifications (`datasource_has_modifications: true`) while excluding unusual
ones (`has_unusual_modification: false`). The effect is that **14,626 of the
39,117 MIC rows (37.4%), covering 2,277 of the 5,602 unique peptides, are
C-terminally amidated.**

C-terminal amidation removes the terminal negative charge, raising net charge
by roughly +1, and is generally associated with increased antimicrobial
potency. The competition requires linear peptides with free termini, and our
generator emits exactly those. The oracle was therefore fitted against labels
systematically more potent than an unmodified analogue of the same sequence
would achieve, and its predictions for our designs should be read as
optimistic by an amount this corpus cannot quantify.

This was not corrected by dropping the amidated rows. Doing so would discard
37% of an already small corpus — 5,602 unique peptides — and the resulting
model would be worse in a way that is easy to measure while the bias it
removes is not. The choice is disclosed rather than hidden, which is the only
honest option available at this corpus size.

### Predicted potency after conditioning is not independent evidence

The MIC oracle used for ranking was trained on the same harmonised activity
data that supplies the `log_mic`, `species` and `gram_selectivity`
conditioning axes. Conditioning the generator toward potent MIC values and
then observing that the oracle predicts potency is close to circular: we
asked for it, and the two models share a corpus.

We therefore report the panel success rate before and after conditioning as a
diagnostic of whether the conditioning had any effect, not as evidence of
potency. The unconditioned run gave 27,753 of 50,000 candidates (55.5%)
clearing the Gram-class eligibility floor, with mean success rates of 0.455
overall, 0.442 Gram-negative and 0.494 Gram-positive. Any improvement over
those numbers shows the conditioning is doing something; it does not show the
peptides are more active. Only the Phase 2 assays can show that.

### The oracle is not extrapolating, and the ranking key reflects that

Conditioning the generator toward potent MIC values raises an obvious worry:
that the library has been pushed somewhere the MIC regressor has no training
support, making its predictions confident and meaningless. We checked before
committing to the library, scoring the generated peptides and a 2,000-sequence
sample of the organizers' reference AMPs through the same oracle, and
comparing both against measured MICs for the panel species.

| distribution | p5 | median | p95 | below 16 µM |
|---|---:|---:|---:|---:|
| generated library (predicted) | 0.569 | 1.334 | 1.955 | 40.3% |
| reference AMPs (predicted) | 0.615 | 1.305 | 1.927 | 39.8% |
| panel species (**measured**) | −0.141 | 1.052 | 2.204 | 58.6% |

log₁₀ µM; the 16 µM threshold is 1.204.

The library's predicted distribution is indistinguishable from the one the
same oracle assigns to known antimicrobial peptides — marginally *less*
potent, not more. Training proximity supports the comparison rather than
undermining it: the library's nearest oracle-training neighbour has median
identity 0.567, against 0.609 for the reference AMPs, so the two sets sit at
comparable distance from the regressor's training data. There is no sign of
extrapolation.

Both predicted distributions are, however, shifted toward weaker potency than
the measured one and span a narrower range. That is the regression-to-the-mean
of a model with a held-out Spearman of 0.533: it compresses predictions toward
the centre. It is a property of the oracle, not of the library, and it applies
equally to real AMPs.

One consequence shapes the ranking. Predicted success rate against the panel
is not saturated across the library — only about 40% of candidates fall below
16 µM on the strain-weighted mean — but it does saturate among the top-ranked
candidates, which are the only ones competing for 100 places out of 50,000. A
key that is constant across every contender cannot order them, so the primary
ranking key is the strain-weighted mean predicted log₁₀ MIC, which retains its
variance. Success rate is still computed and reported as a diagnostic, and the
fraction of candidates pinned at 1.00 is printed at generation time so the
collapse stays visible.

### Cysteine content is a compliance exclusion

The grid requests zero-cysteine peptides only. The training corpus contains a
two-cysteine class amounting to roughly 22% of rows, and it is excluded
deliberately: two free cysteines on a linear peptide oxidise to an
intramolecular disulfide under standard synthesis and handling, which is a
cyclic molecule, and the competition requires linear peptides with free
termini. This is a compliance decision rather than a modelling one — the
excluded share is redistributed across the remaining grid.

## Filters applied to the library

Four filters run, in this order, after every sampling round
(`generate_library` in `src/ampx/generate.py`). A sequence must clear all four
to enter the library.

1. **Compliance** (`compliance_report` in `src/ampx/models/compliance.py`):
   alphabet restricted to the 20 standard residues; length 8-50; at most two
   cysteines. The cysteine bound is a compliance requirement, not a
   physicochemical preference — see [Cysteine content is a compliance
   exclusion](#cysteine-content-is-a-compliance-exclusion).
2. **Deduplication** within the library.
3. **Exact-match exclusion** against `data/antibacterial.fasta`.
4. **Novelty screen.** Levenshtein ratio against every one of the 39,448
   reference sequences, computed exhaustively rather than against a shortlist.
   The applied ceiling is `SAFETY_CEILING = 0.75`, below the 0.80 at which the
   organizers' validator fails, so the shipped library holds margin against
   the requirement rather than sitting on it.

**No physicochemical windows and no synthesizability filter are applied.**
Charge, hydrophobicity and hydrophobic moment enter the pipeline only as
*conditioning* on the generator's sampling grid, never as a post-hoc screen on
what it produced. Nothing is discarded for falling outside a property window.

### The novelty ceiling was documented as 0.80 and is applied at 0.75

An earlier revision of this document stated the applied novelty screen as
"Levenshtein ratio ≤ 0.80", matching the threshold at which the organizers'
validator fails. The shipped default is `SAFETY_CEILING = 0.75` in
`src/ampx/compliance.py`, passed as the `--identity-ceiling` default.

**0.75 is stricter than 0.80, so the discrepancy runs toward the conservative
side.** Every sequence the shipped code admitted would also have passed under
the documented number; the screen that actually ran rejected more than the
prose claimed, not fewer. The error could only ever have understated how much
margin the library holds against the compliance limit.

**The 0.75 is a deliberate margin, not drift.** `src/ampx/compliance.py` keeps
the two thresholds as separate named constants — `IDENTITY_CEILING = 0.80`,
annotated as "the validator's literal rule", and `SAFETY_CEILING = 0.75`, "what
you should actually select against" — with the reasoning recorded beside them:
a candidate at exactly 0.800 passes only because the validator's comparison is
strict, and the real Phase 1 screen is MMseqs2 alignment identity against
MarLys, a different metric over a larger database that will not agree with
Levenshtein in the third decimal place. Headroom costs almost nothing when
picking 100 from 50,000. `SAFETY_CEILING` entered in this repository's first
commit (`a818a9d`, 2026-08-26) at 0.75 and has never been modified since, so
there is no revision in which the code drifted away from a 0.80 default.

The code was correct throughout and the submitted library was never affected —
the realised maximum identity to any reference sequence is **0.7451**, which
satisfies both numbers. Only the prose was wrong. It is corrected in the filter
list above and in the selection section below. The 0.80 figure still appears
where this
document describes the *organizers'* threshold, which is the number they
published; the distinction is now explicit at both sites.

This is the same failure mode as [Tree count at prediction
time](#tree-count-at-prediction-time): a constant restated in prose drifts from
the constant the code reads, and nothing fails when it does.

## Top-100 selection procedure

**Objective — predicted breadth across the competition strain panel.**
Candidates are ranked by the **strain-weighted mean predicted log₁₀ MIC**
across the nine panel species the regressor covers (Appendix B), each species
weighted by the number of strains it contributes to the 20-strain panel.
Ranking by potency against a single organism was rejected: four of the five
award categories score success rate across strain panels, and 25 of our
hundred are drawn at random, so the list needs a floor rather than a peak. A
strain-weighted mean over nine species is a breadth key in exactly that
sense — a candidate potent against one organism and inert against the rest
cannot rank highly under it.

The thresholded form of breadth — the strain-weighted fraction of the panel
predicted at or below 16 µM, which is the Overall Success Rate the
competition scores — is **not** the key. It is computed for every candidate
and reported as a diagnostic, but it saturates at 1.00 among the top-ranked
contenders, and a quantity that is constant across everything competing for
the hundred places cannot order them. The continuous key retains its variance
where the thresholded one has none. This is recorded in full under
[The oracle is not extrapolating, and the ranking key reflects
that](#the-oracle-is-not-extrapolating-and-the-ranking-key-reflects-that),
which is the anchor for this decision; the saturating fraction is printed at
generation time so the collapse stays visible rather than implicit.

### Unresolved: the draw may be from the top 50, not the top 100

The reasoning above, and the module docstring of `src/ampx/ranking.py`, assume
the 25 tested peptides are drawn uniformly from all 100. Under that reading the
expected team score is the mean over the whole list, rank order inside it is
documentation rather than scoring, and the right objective is to raise the
floor rather than the peak. **Two organizer sources disagree on this point and
we do not know which supersedes.** The competition document states that "from
each qualifying team's top-100 list, 25 peptides are drawn uniformly at
random"; the website FAQ states that "a random subset of 25 peptides is drawn
from the **top 50** of this list". If the FAQ is authoritative, ranks 1–50 are
the scored set, ranks 51–100 are unscored, and intra-list ordering becomes a
real design variable rather than a presentational one.

We have **not** changed the selection rule on the strength of a React component
in the challenge website's source, and we flag the ambiguity rather than
silently picking the reading that flatters our design. The question is filed
with the organizers on the challenge issue tracker.

The submitted artifact appears unaffected either way. Selection is a greedy
score-first walk, so the better half already sorts to the front — the top 50
average **+0.1039** predicted log₁₀ MIC against **+0.2227** for ranks 51–100
(lower is more potent) — and the long-band cap is enforced as a running share
at every prefix, so it binds at k=50 (10 long-band members) exactly as it does
at k=100 (19). A list built for the top-100 reading is therefore also a
defensible list under the top-50 reading. What would change is this document's
stated rationale, not the hundred sequences.

In the code this is a single expression — `src/ampx/generate.py`:

```python
score = -metrics["mean_log_mic"] - np.where(eligible, 0.0, 10.0)
```

`mean_log_mic` is the only term that orders eligible candidates; the success
rate never enters `score`. The `10.0` is the eligibility floor below, applied
as a demotion rather than a deletion.

The panel is collapsed onto the species the MIC regressor covers, weighted by
strain count, summing to the full 20:

| Species | Strains | Class |
|---|---:|---|
| *E. coli* | 5 | Gram-negative |
| *P. aeruginosa* | 3 | Gram-negative |
| *K. pneumoniae* | 2 | Gram-negative |
| *A. baumannii* | 2 | Gram-negative |
| *S. enterica* | 2 | Gram-negative |
| *E. cloacae* | 1 | Gram-negative |
| *S. aureus* | 2 | Gram-positive |
| *E. faecalis* | 2 | Gram-positive |
| *B. subtilis* | 1 | Gram-positive |

Two departures from the literal panel, both deliberate:

- **Five trained species are excluded** — *C. albicans*, *S. epidermidis*,
  *M. luteus*, *B. cereus*, *L. monocytogenes*. None appears on the panel, so
  including them would dilute the ranked quantity with organisms that are
  never assayed.
- ***E. faecium* is folded into *E. faecalis*.** The panel includes one
  *E. faecium* VRE strain, which is absent from the regressor's species
  vocabulary, so its weight is assigned to *E. faecalis* — same genus, and
  also represented on the panel by a VRE isolate. This is an approximation,
  not a prediction for *E. faecium*: it assumes the two enterococci respond
  similarly, which is plausible and unverified here.

**Eligibility floor.** A candidate must be predicted below 16 µM against at
least one Gram-negative *and* at least one Gram-positive strain. Gram-Positive
Activity is its own award category over only 5 strains, and a
Gram-negative-only peptide also caps at 75% Overall, so a peptide potent in
one class alone wins nothing in either. Ineligible candidates are ranked below
every eligible one rather than removed, so the pool never silently shrinks
below 100 and the exclusion is visible in the ordering. Gram-negative and
Gram-positive success rates are reported separately at generation time as
diagnostics.

**No MDR-specific targeting.** Seven of the 20 strains are multi-drug
resistant isolates, and we make no attempt to favour them. The regressor
predicts from (sequence, species) and has no feature that distinguishes a
resistant isolate from a susceptible one of the same species, so an
MDR-directed objective would be uninformed by the model — it would express a
preference the predictor cannot actually act on. MDR strains therefore
contribute to our panel aggregate only through their species.

**Uncertainty.** None is modelled. The oracle is a single regressor, so
`Candidate.score_std` is 0 and the `kappa` risk-aversion term in `select_top`
is inert. It remains wired so an ensemble can replace the point predictor
without changing the selection code.

**Diversity constraint.** At most 4 candidates per structural cluster, where
clusters are currently a coarse length-band × charge-band bucket.

**Length: a cap on the long bands, and nothing else.** The oracle scores
longer peptides as more potent — length and a normalised length are 2 of its
9 physicochemical descriptors, and 40% of its MIC training rows are amidated
peptides — so a greedy walk down the ranking drifts long. The selection
therefore caps the two longest length bands (38–43, and 44–50 — the top band
is open-ended rather than 6 residues wide) at a
combined 20% of the list, enforced as a running share at every prefix rather
than as a total, so the bound holds at k=100 and at every k within the
overflow rather than being exhausted early by the highest-ranked candidates.

No other length constraint is applied. In particular the short bands are
given **no floor**: they are not capped, but neither are they granted quota
slots they did not earn on predicted MIC. Admitting a candidate *because* it
is short would import length into the selection rule to compensate for the
oracle having imported length into its scoring, which is the same error in
the opposite direction. Proportional stratification across all seven bands
was implemented and rejected for exactly this reason — it forces slots into
the 8–13 band where only 10.9% of the library clears the eligibility floor,
buying a distributional match by promoting candidates the oracle rates
poorly in the band where it is least reliable. One constraint, in one
direction.

The cap is fixed in code, not by invocation: `LONG_BANDS = {5, 6}` and
`LONG_SHARE = 0.20` are module constants in `src/ampx/ranking.py`, applied
inside the one selection function the entry point calls. No command-line flag
selects a different algorithm, so no invocation produces a selection rule
other than this one.

#### The submitted list is produced by the entry point, not curated

`generate/top.fasta` is whatever `uv run generate` writes. It is not, and
cannot be, a list chosen by hand and committed: the organizers' validator
clones the repository, runs the entry point, and compares the regenerated
output against a second run, so any file placed in `generate/` is overwritten
by their first command. A selection rule that lived in an analysis script
rather than in `ampx` would therefore be discarded at validation while still
appearing correct in the repository, and every downstream check would pass on
the wrong list.

Every constraint described in this section consequently lives in
`ampx.ranking.select_top_diverse`, on the entry-point path.
`scripts/stratified_reselect.py` re-selects over an already-generated library
so the choice can be examined without an hour-long GPU run, but it imports
that same function rather than reimplementing it; it produces byte-identical
output, which is checked by hash. There is one selection implementation and no
mechanism by which the shipped list and the analysed list can diverge.

This is a reproducibility property worth stating plainly: the top 100 cannot
be curated after the fact, and nothing in this submission's ranking is applied
outside the code the organizers run themselves.

**Novelty screen.** Levenshtein ratio ≤ 0.75 against every one of the 39,448
reference sequences, computed exhaustively rather than against a shortlist,
since a single sequence above the ceiling invalidates the list. The organizers'
threshold is 0.80; the applied ceiling is `SAFETY_CEILING = 0.75`
(`src/ampx/compliance.py`), chosen to leave margin. The realised maximum over
the submitted library is 0.7451.

**How far to trust the ranking.** The oracle's held-out Spearman is 0.533 on a
cluster-disjoint split, with a mean absolute error of 0.509 log₁₀ units — a
typical prediction is off by roughly threefold in concentration. It is used as
a coarse enrichment filter and a breadth estimator, not as a fine-grained
ordering of the top 100, and nothing in the procedure depends on the exact
order near the top.

### The top 100 is skewed long, and the cap did not fix it

The submitted list does not match the reference length distribution, and this
section states how far off it is rather than presenting the capped list as a
corrected one. **The cap bounded the tail; the skew remains.**

The library itself is not the problem. Its length distribution tracks the
reference closely (KS 0.0792 against `data/antibacterial.fasta`). The skew is
introduced entirely by selection.

| | median | >40 residues | length KS vs reference |
|---|---:|---:|---:|
| reference (`data/antibacterial.fasta`, n=39,448) | 18 | 3.5% | — |
| generated library (n=50,000) | 19 | 4.6% | 0.0792 |
| top 100, before the cap | 42.5 | 58% | **0.7846** |
| top 100, as submitted (capped) | 33 | 14% | **0.6308** |

The cap removed most of the tail — 58% of the list above 40 residues fell to
14% — and improved KS by about a fifth, from 0.785 to 0.631. It did not bring
the list near the reference, and it was never going to. **No candidate of 8–19
residues survived the selection, in either band, out of 5,533 that cleared the
eligibility floor.**

#### Why: the oracle's eligibility is length-dependent

The Gram-class eligibility floor passes a strongly length-dependent fraction
of the library, measured on the gated library itself:

| Band | 8–13 | 14–19 | 20–25 | 26–31 | 32–37 | 38–43 | 44–50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| library | 13,618 | 11,681 | 14,679 | 3,804 | 3,092 | 1,497 | 1,629 |
| eligible | 1,487 | 4,046 | 8,540 | 2,862 | 2,457 | 1,382 | 1,567 |
| **eligible rate** | **10.9%** | 34.6% | 58.2% | 75.2% | 79.5% | 92.3% | **96.2%** |
| selected (of 100) | **0** | **0** | 24 | 22 | 35 | 9 | 10 |

Predicted potency is monotonic in length band. The most potent eligible
candidate available in each band, in band order:

| Band | 8–13 | 14–19 | 20–25 | 26–31 | 32–37 | 38–43 | 44–50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| best predicted log₁₀ MIC | +0.479 | +0.273 | +0.090 | −0.005 | −0.284 | −0.526 | −0.748 |

The rank correlation between band index and best achievable predicted potency
is −1.00 across all seven bands. The consequence is decisive: the most potent
short peptide in the entire library (+0.273) is less potent, by the oracle's
own estimate, than the **worst** candidate admitted to the top 100 (+0.257).
Zero of the 5,533 eligible short candidates beat anything on the list, so the
number surviving a greedy walk is zero at any cap share on the long bands. The
cap can only redistribute within 20–49 residues, and that is what it did.

Two properties of the MIC regressor explain the gradient: length and a
normalised length are 2 of its 9 physicochemical descriptors, and roughly 40%
of its training rows are amidated peptides, which are systematically more
potent than the unmodified analogues we generate (see
[The MIC labels include modified
peptides](#the-mic-labels-include-modified-peptides-our-designs-are-unmodified)).

#### What the cap cost, and why that supports it

Predicted potency of the selected 100, before and after the cap:

| | mean | median | sd |
|---|---:|---:|---:|
| before the cap | −0.0177 | +0.0644 | 0.197 |
| as submitted (capped) | +0.1633 | +0.1801 | 0.086 |
| change | +0.181 | +0.116 | — |

Positive is less potent. In concentration the mean moves from 0.96 µM to
1.46 µM, a 52% increase, which sounds large until it is placed against the
oracle's own accuracy: the shift is **0.36× the regressor's held-out mean
absolute error of 0.509 log₁₀ units**. The entire predicted cost of discarding
the long tail is well inside the noise floor of the model that predicted it.

That is the argument for the cap. If bounding the long tail — 58% of the
uncapped list sat above 40 residues, against 14% of the submitted one — cost
real activity, it would show up as a shift large relative to the predictor's
resolution; it does not. The length preference is better read
as a scoring artifact of a model with length in its features and amidated
peptides in its labels than as a genuine biological signal, so declining to
follow it costs little that the model can actually resolve.

#### A label defect in the band tables, caught and corrected

The two band tables above were at one point headed `44–49` in their rightmost
column. That label was wrong. `length_band()` clamps with
`min((length - 8) // 6, N_BANDS - 1)`, so the top band is open-ended and holds
everything from 44 residues to the 50-residue maximum, not a 6-residue window.
323 library sequences and 2 of the submitted top 100 are 50 residues and sit in
that band while falling outside its printed label.

The counts were never affected: every figure in these tables was computed
through `length_band()`, and they re-derive exactly from
`generate/library.fasta` and `generate/top.fasta`. The cap is likewise applied
on `length_band()` and always governed the true band, so the submitted list is
unchanged. Only the printed label was wrong.

The label came from `band_label()` in `scripts/stratified_reselect.py`, which
computed its upper bound as `min(lo + BAND_WIDTH - 1, MAX_PEPTIDE_LENGTH)` —
correct for bands 0–5 and off by one for the top band. It is corrected to
return `MAX_PEPTIDE_LENGTH` for band `N_BANDS - 1`. The script had also
restated `BAND_WIDTH` and `N_BANDS` locally instead of importing them from
`ampx.ranking`, which is how the two definitions were able to drift apart at
all; it now imports both. This is recorded rather than quietly fixed because
the same class of defect appears twice elsewhere in this document, at [Tree
count at prediction time](#tree-count-at-prediction-time) and at [The novelty
ceiling was documented as 0.80](#the-novelty-ceiling-was-documented-as-080-and-is-applied-at-075).

#### What we did not do

We did not relax the eligibility floor for short peptides, and we did not
allocate quota slots to the short bands. Both would have produced a list
matching the reference distribution, and both would have done it by admitting
candidates *because* they are short — importing length into the selection rule
to offset the oracle having imported length into its scoring. The residual
skew is reported here instead. It is a limitation of the ranking model, stated
as one, and it is not corrected by the procedure that produced the submitted
list.

### The overflow list did not continue the top ranking

We ship a ranked overflow list beyond the submitted 100 so the ranking can be
extended without a second selection. That is only meaningful if the overflow
continues the *same* ranking, and for a period it did not.

The re-selection script called its allocation routine twice — once for k=100
and once for k=500 — and treated the two results as one list and its
extension. They were not. The passes allocated independently, so the first 100
of the overflow were not the top 100. Measured on the affected artifacts:
**only 40 of the top 100 appear in the overflow's first 100**, and 96 of the
100 ranks hold a different sequence, the first disagreement at rank 5. The
lists were not unrelated — 86 of the 100 appear somewhere in the overflow's
500 — which is precisely why the defect was survivable: the overflow looked
like a superset, and only its *order* was wrong. Nothing caught it, because
each file was internally valid on its own — correct length, no duplicates,
every member drawn from the library — and no check compared them to each
other.

The fix is by construction rather than by assertion. Selection now runs once,
at the full overflow length, and the top list is the first `--k` entries of
that single result, so the prefix property cannot be violated without the
selector being wrong about its own output.

The check that guards it deliberately lives in `scripts/verify_submission.py`
and compares the two files on disk. An earlier attempt asserted the invariant
inside the selection script, one line after assigning `top = ranked[:k]` —
comparing a value against itself, so it passed unconditionally and would have
gone on passing however the files were later written. An assertion that cannot
fail is worse than none, because it reads like coverage.

## Manual interventions

**None.**

No sequence in either submitted file was hand-picked, hand-written, edited,
added or removed. Both files are whatever `uv run generate` writes on a clean
clone; the organizers' validator regenerates them from the entry point, and it
reproduced our hashes byte-for-byte on a different machine.

The selection rules themselves are human design decisions — the strain-weighted
ranking objective, the ≤2-cysteine compliance bound, the novelty ceiling, and
the cap on the two longest length bands described in [Length: a cap on the long
bands](#length-a-cap-on-the-long-bands-and-nothing-else). All of them live in
`src/ampx/ranking.py` and `src/ampx/generate.py`, on the path the organizers
execute. None was applied to the artifact after the fact. We draw the line
there deliberately: a rule that runs inside the entry point is part of the
method and is reproducible from the seed alone, whereas a rule applied to the
output afterwards would be curation and would not survive regeneration.

## Reproducibility

Fixed seed 42. `uv sync && uv run generate` produces byte-identical output.

The sampling batch size is part of this contract, not a performance knob.
Sampling draws from one seeded generator across the whole batch, so changing
the batch size changes how that stream is consumed and produces a different
library from the same seed. `BATCH_SIZE` in `ampx/generate.py` is fixed for
this reason; retuning it for a different GPU's memory would silently change
the submitted library.

Model weights are **not** stored in git. `checkpoint/masked_diffusion_best.pt`
and `checkpoint/mic_regressor.json` ship as GitHub Release assets;
`checkpoint/SHA256SUMS` is committed, and `ampx.weights.ensure_weights` --
called at the top of `generate.main`, and exposed as
`scripts/fetch_weights.py` -- downloads and verifies them, so a clean clone
runs with no manual step. git-lfs was deliberately abandoned for this: the
organizers validate with a plain `git clone`, and on a machine without git-lfs
installed, or after the repository's LFS bandwidth is spent, an LFS-tracked
file arrives as a small pointer that fails only later, deep inside a model
load. The fetcher detects a pointer file explicitly and replaces it.

`scripts/verify_submission.py` is the organizers' own validator from the
template repository, with two additions:

1. **A step that constructs an `Oracle` and scores a sequence** in the synced
   environment. Importing the package proves much less than it appears to,
   because it builds no model and reads no checkpoint — that gap is how a
   missing scikit-learn dependency survived an earlier import check.
2. **A step that checks the overflow list against the top list**, requiring
   `generate/top.fasta` to be the first 100 entries of the ranked overflow, in
   order. This is a check between the two files as written, not inside the
   selector, because an assertion inside the selector compares two slices of
   one list and cannot fail. See
   [The overflow list did not continue the top
   ranking](#the-overflow-list-did-not-continue-the-top-ranking). The step is
   skipped when no overflow file is present, since the entry point does not
   produce one.

The file is otherwise unmodified, so its provenance remains visible in a diff
against the template.
Verified on: Rocky Linux 9.8 (kernel 5.14.0-687.12.1.el9_8.x86_64), Python
3.11, NVIDIA H100 80GB HBM3, CUDA. Cross-machine reproduction used two
different nodes of the same cluster: the full-scale determinism gate on one,
the organizers' validator on another, from a fresh clone with weights
downloaded from the release.

## Use of AI assistants
Claude was used for writing and editing content.
