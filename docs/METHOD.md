# Method

> Fill this in as you build. It is a **submission requirement**: the abstract,
> training-data disclosure, filters applied, and selection procedure are all
> required for benchmark participation, and the LLM-assistance disclosure is
> required by the NeurIPS Main Track Handbook.

## Abstract

_(150-250 words. What the model is, what it is conditioned on, how the top-100
was selected.)_

## Training data

See [../data/README.md](../data/README.md) for the full provenance table.

- **Generator corpus:** _(sources, record counts, filters)_
- **Predictor corpus:** _(EV positives/negatives, synthetic negatives, counts)_
- **Splits:** MMseqs2 clustering at 40% identity, cluster-disjoint.

### Harmonized activity table

`data/processed/activity_harmonized.csv` is committed to this repository rather
than rebuilt at run time. It is the table `scripts/build_dataset.py` writes and
the MIC regressor trains on, with columns `sequence, species, strain, endpoint,
value_um, censored, source`.

It previously existed here only as a header-only stub, which would have made
the training-data disclosure unverifiable. The real table was recovered from
the cluster at
`$D/processed/conditioned/activity_harmonized.csv`
(where `$D` is defined in the source repository's `scripts/hpc/env.sh`) and
committed whole. At 5.3 MB it is small enough to version directly, so no
fetch step or rebuild is required to check our numbers.

Row counts, verified against the committed file:

| Partition | Rows |
|---|---|
| Total | 80,473 |
| `endpoint == "mic"` | 39,117 |
| `endpoint == "hc50"` | 41,356 |

Replicate measurements are collapsed upstream by geometric mean in log10
space, so a peptide assayed repeatedly against one species contributes one
row and cannot dominate the fit.

### Version pinning of shipped checkpoints

The inference dependencies in `pyproject.toml` pin `torch`, `transformers` and
`xgboost` to exact versions, because each determines how a shipped checkpoint
is read rather than merely how fast it runs:

- **`xgboost==3.2.0`** — the version recorded in `checkpoints/mic_regressor.json`'s
  own `version` field. That model holds 1,807 trees with `best_iteration=1706`,
  so 101 trees sit past the early-stopping point. Whether a reader truncates at
  `best_iteration` or predicts with every tree has changed across XGBoost
  releases, and the two answers give different MIC estimates and therefore a
  different top 100.
- **`transformers==5.17.0`** — produced the 480-dimensional ESM-2 embedding
  block. The regressor's 510 input features are 9 descriptors, 16 species
  indicators, 5 Gram indicators and those 480 embedding dimensions, so a
  different encoder yields a feature matrix of the wrong width or the wrong
  meaning.
- **`torch==2.11.0`** — trained `checkpoints/masked_diffusion_best.pt`.

### Encoder identity, recovered after the fact

`checkpoints/mic_regressor_meta.json` now carries
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

The superseded model is kept at `checkpoints/mic_regressor_leaky_split.json`
with its own metadata, so the comparison can be re-run rather than taken on
trust. It is not loaded by anything.

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

- Architecture:
- Representation:
- Conditioning:
- Sampler:
- Checkpoint: `checkpoints/`

## Surrogate predictors

- Members of the ensemble:
- Calibration method:
- Held-out performance (clustered split): FPR, Prec@100, AUPRC

## Filters applied to the library

1. Alphabet restricted to the 20 standard residues; length 8-50.
2. Deduplication.
3. Exact-match exclusion against `data/reference/antibacterial.fasta`.
4. _(physicochemical windows, synthesizability, etc.)_

## Top-100 selection procedure

_(This is explicitly requested by the organizers. Describe the objective, the
uncertainty estimate, the diversity constraint, and the novelty screen.)_

- Objective:
- Uncertainty:
- Diversity constraint:
- Novelty screen: Levenshtein ratio <= 0.80 vs all reference sequences, plus
  MMseqs2 identity <= 80% vs the MarLys AMP database.

## Manual interventions

_(State plainly. "None" is a valid and strong answer -- the competition is
partly about separating model quality from human curation.)_

## Reproducibility

Fixed seed 42. `uv sync && uv run generate` produces byte-identical output.
Verified on: _(OS, Python, GPU)_

## Use of AI assistants

_(Required disclosure. State which tools were used and for what.)_
