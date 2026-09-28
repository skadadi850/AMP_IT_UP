# Method

> Fill this in as you build. It is a **submission requirement**: the abstract,
> training-data disclosure, filters applied, and selection procedure are all
> required for benchmark participation, and the LLM-assistance disclosure is
> required by the NeurIPS Main Track Handbook.

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
are ordered by the fraction of the panel predicted at or below 16 µM, with
mean predicted MIC as the tie-break, under a hard floor requiring activity
against at least one Gram-negative and one Gram-positive strain. The
regressor is validated on a cluster-disjoint 40%-identity split (Spearman
0.533) and used as a breadth estimator, not a fine-grained ranking.

## Training data

See [../data/README.md](../data/README.md) for the full provenance table.

- **Generator corpus:** trained in two stages over the sources tabulated in
  [Generator corpus, by stage](#generator-corpus-by-stage) below. Stage 5a
  pretrains on 309,272 sequences; stage 5b fine-tunes on 60,405 conditioned
  rows.
- **Predictor corpus:** the MIC arm of
  `data/processed/activity_harmonized.csv` — 39,117 rows over **5,602 unique
  peptides** and 659 species, from GRAMPA. The row-to-peptide ratio is the
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
  sequences with GenBank accessions, drawn from sORFdb/SmProt, length-filtered
  to 8–50 residues; 269,824 survive deduplication into the pretrain table.
  These are real translated open reading frames, **not** shuffled, random or
  mutated decoys.
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

1. Alphabet restricted to the 20 standard residues; length 8-50.
2. Deduplication.
3. Exact-match exclusion against `data/antibacterial.fasta`.
4. _(physicochemical windows, synthesizability, etc.)_

## Top-100 selection procedure

**Objective — predicted breadth across the competition strain panel.**
Candidates are ranked by the strain-weighted fraction of the 20-strain panel
(Appendix B) they are predicted to inhibit at or below 16 µM, which is the
Overall Success Rate the competition scores, with mean predicted log₁₀ MIC as
the tie-break. Ranking by potency against a single organism was rejected:
four of the five award categories score success rate across strain panels, and
25 of our hundred are drawn at random, so the list needs a floor rather than a
peak.

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

**Novelty screen.** Levenshtein ratio ≤ 0.80 against every one of the 39,448
reference sequences, computed exhaustively rather than against a shortlist,
since a single sequence above the ceiling invalidates the list.

**How far to trust the ranking.** The oracle's held-out Spearman is 0.533 on a
cluster-disjoint split, with a mean absolute error of 0.509 log₁₀ units — a
typical prediction is off by roughly threefold in concentration. It is used as
a coarse enrichment filter and a breadth estimator, not as a fine-grained
ordering of the top 100, and nothing in the procedure depends on the exact
order near the top.

## Manual interventions

_(State plainly. "None" is a valid and strong answer -- the competition is
partly about separating model quality from human curation.)_

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
template repository, with one addition: a step that constructs an `Oracle` and
scores a sequence in the synced environment. Importing the package proves much
less than it appears to, because it builds no model and reads no checkpoint —
that gap is how a missing scikit-learn dependency survived an earlier import
check. The file is otherwise unmodified, so its provenance is visible in a
diff against the template.
Verified on: _(OS, Python, GPU)_

## Use of AI assistants
Claude was used for writing and editing content. 
_(Required disclosure. State which tools were used and for what.)_
