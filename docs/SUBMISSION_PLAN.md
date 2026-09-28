# AMP Challenge submission plan

Working repo: `/home/skadadi/Antimicrobial_Peptide_Project/AMP_IT_UP`
Source repo:  `/home/skadadi/Antimicrobial_Peptide_Project/GRAMP`

Deadline: 3 days. Compliance failure is disqualification, not a warning. The
computational phase is a scored 0-100 gate; only 20 teams advance to wet lab.

Phases are ordered by dependency. Do not start Phase 4 until Phase 3 passes.

---

## Phase 0. Inventory what is already moved

Some files are already copied. Establish ground truth before copying more.

```bash
cd /home/skadadi/Antimicrobial_Peptide_Project
ls -la AMP_IT_UP/src/ampx AMP_IT_UP/src/ampx/models AMP_IT_UP/scripts AMP_IT_UP/checkpoints
diff -rq GRAMP/src/GRAMP AMP_IT_UP/src/ampx/models 2>&1 | head -40
```

Report which Tier 1 files below are present, which are missing, and which
differ from the GRAMP original. Do not overwrite a file that differs without
showing the diff first.

---

## Phase 1. File moves

### Tier 1: required for the entry point to run

Copy unchanged into `src/ampx/models/`. Keeping them in one directory preserves
every `from .features import ...` relative import.

| From (GRAMP) | To (AMP_IT_UP) |
|---|---|
| `src/GRAMP/masked_diffusion.py` | `src/ampx/models/masked_diffusion.py` |
| `src/GRAMP/conditioning.py` | `src/ampx/models/conditioning.py` |
| `src/GRAMP/features.py` | `src/ampx/models/features.py` |
| `src/GRAMP/sampling.py` | `src/ampx/models/sampling.py` |
| `src/GRAMP/compliance.py` | `src/ampx/models/compliance.py` |
| `src/GRAMP/novelty_fast.py` | `src/ampx/models/novelty_fast.py` |
| `src/GRAMP/encoder.py` | `src/ampx/models/encoder.py` |
| `scripts/generate_masked.py` | `scripts/generate_masked.py` |
| `scripts/train_mic_regressor.py` | `scripts/train_predictor.py` |
| `results/comparison/variants/length_calibration_faceonly.json` | `checkpoints/length_calibration.json` |

`src/ampx/models/__init__.py` stays empty. Do not copy `src/GRAMP/__init__.py`;
it imports `diffusion`/`decoder`/`data`/`ingest` and would pull the latent path
into the import graph.

Do not overwrite `src/ampx/compliance.py`. That is the organizer gate wired
into `generate.py`. GRAMP's compliance lands under `models/` as a separate
module; the two coexist.

Delete `src/ampx/featurize.py` (one-line stub, superseded by
`models/features.py`).

### Tier 2: required for training disclosure and documentation

| From (GRAMP) | To (AMP_IT_UP) |
|---|---|
| `src/GRAMP/data.py` | `src/ampx/models/data.py` |
| `src/GRAMP/ingest.py` | `src/ampx/data/ingest.py` |
| `src/GRAMP/metrics.py` | `src/ampx/models/metrics.py` |
| `scripts/train_masked_diffusion.py` | `scripts/train_generator.py` |
| `scripts/build_dataset.py` | `scripts/build_dataset.py` |
| `scripts/build_mic_table.py` | `scripts/build_mic_table.py` |
| `scripts/comparison/calibrate_length.py` | `scripts/calibrate_length.py` |
| `scripts/comparison/clean_libraries.py` | `scripts/clean_libraries.py` |
| `scripts/comparison/compute_metrics.py` | `scripts/compute_metrics.py` |
| `data/processed/activity_harmonized.csv` | `data/processed/activity_harmonized.csv` |
| `PREREGISTRATION_independent_oracle.md` | `docs/PREREGISTRATION_independent_oracle.md` |
| `PREREGISTRATION_arm_e.md` | `docs/PREREGISTRATION_arm_e.md` |

### Do not move

`.git/`, `logs/`, `scripts/hpc/**` (SLURM, cluster-specific),
`results/comparison/**`, `results/holdout_comparison/**`,
`scripts/run_refinement.py` and the refinement analysis scripts,
`src/GRAMP/diffusion.py`, `src/GRAMP/decoder.py`.

### Checkpoints

These are on the cluster, not in either repo tree. Copy into `checkpoints/`:

- `masked_diffusion_best.pt` (the selectivity stage-5b fine-tune, not the
  `conditioned/` one)
- `species_vocab.json`
- `mic_regressor.json`
- `mic_regressor_meta.json`

### Gate

```bash
cd AMP_IT_UP
uv run python -c "from ampx.predictor import Oracle; from ampx.models.masked_diffusion import MaskedDiffusionModel; print('ok')"
```

---

## Phase 2. Consolidate the predictor

`src/ampx/predictor.py` currently holds `DESCRIPTORS`, `GRAM_LEVELS`,
`build_features` and `Oracle`.

1. In `scripts/train_predictor.py`, delete the local `DESCRIPTORS` definition
   and the local `build_features` body. Replace with
   `from ampx.predictor import DESCRIPTORS, build_features`. Delete the
   `sys.path.insert` hack that made the old cross-script import work.
2. Add `esm_model` to the metadata dict written by `train_predictor.py`, and
   read it in `Oracle.__init__` as
   `ESM2Encoder(model_path=self.meta.get("esm_model", DEFAULT_ESM_MODEL), device=device)`.
   Without this the metadata records only `uses_embeddings`, so a checkpoint
   trained under a different ESM-2 silently produces a wrong-width feature
   block.
3. Give `Oracle.__init__` a device default that resolves
   `"cuda" if torch.cuda.is_available() else "cpu"`. The organizers may run
   CPU-only.

If the existing `mic_regressor_meta.json` lacks `esm_model`, do not retrain.
Add the key by hand with the value that was actually used and record that in
`docs/METHOD.md`.

---

## Phase 3. Entry point, determinism, hard constraints

### 3.1 Wire `src/ampx/generate.py`

Three replacements:

- `_fit_markov` / `_placeholder_generator` -> load `masked_diffusion_best.pt`
  and call `sample_library` from `scripts/generate_masked.py`.
- `score_candidates` (the net-charge proxy) -> `Oracle.predict`.
- The novelty screen inside `select_top` -> `ExhaustiveNovelty` against
  `data/reference/antibacterial.fasta`.

The approximate `NoveltyFilter` top-400 shortlist is acceptable for filtering
the 50,000-member library. It is not acceptable as the accept/reject gate on
the top 100: a shortlist can miss a true high-identity match, and one sequence
over 80% identity is an invalid candidate.

### 3.2 Seeding

Seed in one function called at the top of `main()`: `random.seed`,
`numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`,
`torch.use_deterministic_algorithms(True)`,
`torch.backends.cudnn.deterministic = True`. Set the seed as a module-level
default constant, not a required CLI argument, since the requirement is that
running the script twice with no arguments reproduces the output.

If any `DataLoader` is constructed at inference time, set `num_workers=0` or
pass an explicit `generator` and `worker_init_fn`.

### 3.3 Pin external dependencies

- `models/encoder.py`: pass `revision=<commit hash>` to both
  `AutoTokenizer.from_pretrained` and `AutoModel.from_pretrained`. An unpinned
  Hugging Face pull makes the organizers' run diverge from ours regardless of
  seeding.
- `pyproject.toml`: pin `xgboost` to an exact version. Whether a reloaded model
  predicts with all trees or truncates at `best_iteration` has changed across
  majors, and that reshuffles the top 100.
- Move `torch`, `transformers`, `xgboost`, `rapidfuzz`, `pandas`, `numpy`,
  `scipy`, `levenshtein` from the `train` extra into `dependencies`. Inference
  needs all of them and the organizers run only `uv sync`.

### 3.4 Write `scripts/verify_submission.py`

Loads the library and the top-100 file and asserts, exiting nonzero on any
failure:

- alphabet is exactly the 20 canonical residues
- every length in [8, 50]
- exactly 50,000 rows in the library, all unique
- exactly 100 rows in the top list
- top-100 is a subset of the library
- no top-100 sequence exceeds 80% identity to any reference sequence
  (exhaustive, not shortlisted)
- required metadata columns present and non-null

### 3.5 Determinism gate

```bash
uv run generate --out run_a/
uv run generate --out run_b/
diff run_a/library.csv run_b/library.csv && echo DETERMINISTIC
uv run python scripts/verify_submission.py --library run_a/library.csv --top run_a/top100.csv
```

Both must pass before Phase 4.

---

## Phase 4. Library composition

The aggregation score rewards the library as a population: physicochemical
distributions, predicted potency, embeddings versus known potent peptides,
synthesizability, novelty, diversity. A library collapsed onto one
high-charge mode scores badly on diversity even when every member is
individually plausible. Do not make all 50,000 look like the top 100.

1. Sample across the conditioning grid, not at a single optimum. Sweep length
   bins against the calibrated distribution in `checkpoints/length_calibration.json`,
   and sweep charge and hydrophobic moment across the range the training data
   covers.
2. Apply the 80% novelty ceiling to the entire library, not only the top 100.
   Near-exact matches against DBAASP/dbAMP/APD are explicitly scored.
3. Measure before committing. Run `scripts/compute_metrics.py` on the candidate
   library and report: pairwise identity distribution, unique k-mer coverage,
   per-axis KS gap versus the reference set, predicted MIC distribution.
4. If diversity is thin, widen the conditioning grid. Do not lower sampling
   temperature.

Report the metrics table before regenerating. Do not silently iterate.

---

## Phase 5. Top 100 selection

Rank by predicted MIC from the oracle, then apply a diversity constraint so no
two selected peptides exceed a fixed pairwise identity threshold. Twenty-five
of the hundred are drawn at random, so the list needs a floor, not a peak; one
strong sequence buys nothing.

Document the exact procedure in `docs/METHOD.md`. The ranking procedure is
itself a required deliverable, not an implementation detail.

Re-run `verify_submission.py` on the final top 100.

---

## Phase 6. Repository and submission artifacts

1. Clone `https://github.com/szczurek-lab/amp-challenge-2027` into `/tmp` and
   diff its layout, expected output filenames, and any `.gitattributes`
   against ours. The full requirements say "following the provided template."
   Match its paths exactly. Report any mismatch before changing ours.
2. `LICENSE`: MIT, referenced in `README.md`.
3. Weights ship as GitHub Release assets on a tag, not git-lfs. Commit
   `checkpoints/SHA256SUMS`; `.gitignore` the `.pt` and regressor `.json`.
   `scripts/fetch_weights.py` downloads and verifies; `generate.py` calls it at
   the top of `main()` so a clean clone works with no manual step.
4. Fix the stale layout lines: `README.md:55` still says
   `checkpoints/ trained weights (git-lfs)` and lists `featurize.py`, which no
   longer exists. `docs/PLAN.md:446` still has the git-lfs checklist item.
5. `docs/METHOD.md` needs: the method abstract, full training-data disclosure
   naming every source, confirmation that DBAASP/APD/Peptipedia terms of use
   were checked, the computational filters applied, and an explicit statement
   about manual intervention. If there was none beyond the coded filters, say
   so in those words.
6. Make the repo public. Grant read access to `@RasmusML` and `@szymczakpau`.

---

## Phase 7. Kaggle submission

The README's step 6 says "To submit, head to the Kaggle competition page:
https://www.kaggle.com/competitions/amp-challenge". This is a second delivery
channel the earlier phases did not account for: making the repository correct
is necessary but is not, by itself, submitting.

### Hard date

**October 1st 2026 AOE** is the submission deadline for both the
50,000-peptide library and the top-100 list (competition document, section
2.4 Timeline). Phase 1 computational evaluation runs through October 2026,
with qualification decisions communicated afterwards.

### What is established

- Registration requires an **institutional email address**; free webmail is
  rejected for primary registration. `skadadi@purdue.edu` qualifies.
- One team submits **one** library and **one** top-100 list. Contributing to
  more than one team requires a genuinely distinct method and must be declared
  at registration.
- The repository is a separate requirement from the Kaggle upload: public for
  co-authorship eligibility, or private with read access granted to
  `@RasmusML` and `@szymczakpau` for benchmark participation.
- If the platform is down near the deadline, organizers open a Google Forms
  backup endpoint with compliance checks applied manually.

### What is not yet known

Whether Kaggle wants the two FASTA files uploaded directly, a repository URL,
or both, and whether it enforces its own file-naming or size limits. The
Kaggle page and the competition website are both client-rendered, so neither
could be read automatically; this has to be checked by hand, signed in.

**Do this early, not at the end.** Registration may require approval of the
institutional address, and the answer may constrain the output format — which
is the one thing that is expensive to change after a generation run.

### Checklist

1. Register the team with the Purdue address; declare any multi-team
   participation.
2. Read the Kaggle submission form and record here exactly what it accepts.
3. Confirm whether the repository URL is supplied through Kaggle or by email.
4. Submit `generate/library.fasta` and `generate/top.fasta` as that form
   requires, before October 1st AOE.
5. Grant `@RasmusML` and `@szymczakpau` read access, and make the repository
   public if pursuing co-authorship.

---

## Final gate

On a clean clone, on a machine that is not the dev box:

```bash
git clone <repo> fresh && cd fresh
uv sync
uv run generate
uv run python scripts/verify_submission.py
diff outputs/library.csv <submitted library>
```

This is the exact procedure the organizers described. If it passes here, the
submission is valid.

---

## Triage

If time runs short, cut Phase 4 breadth. A compliant, reproducible, moderately
diverse library scores. A better library that fails `uv sync` scores zero.
Phases 1, 2, 3 and 6 are non-negotiable.
