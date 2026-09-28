# Preregistration — `deep_fusion` as a consensus filter

Written 2026-09-28, **before** the classifier was trained and before any
score, correlation or removal count was observed. The decision rules below are
fixed in advance so that a filter which turns out to be decorative is dropped
on the evidence rather than kept because it was built.

## What is being tested

The submission ranks candidates by strain-weighted mean predicted log10 MIC
from the XGBoost oracle (`checkpoint/mic_regressor.json`). The proposal is to
add `deep_fusion` — a binary AMP classifier — as a **filter, not a ranker**:
after ranking, drop any candidate whose AMP probability falls below a
threshold, before the 0.60 pairwise-identity diversity pass.

A second model is only worth shipping if it carries information the oracle
does not already have. If it reproduces the oracle's ordering, the filter
removes whatever the oracle already ranked last and buys nothing.

## The decision rules

Both are evaluated across the full 50,000-sequence library from the shipped
generation run.

**Rule 1 — independence.** Report Spearman rho between `deep_fusion` AMP
probability and oracle strain-weighted mean predicted log10 MIC.

- **|rho| >= 0.60 → drop the filter.** The classifier is echoing the oracle.
- |rho| < 0.60 → the filter may proceed to rule 2.

The sign is expected to be negative: a higher AMP probability should go with a
lower predicted MIC. The rule is on magnitude, so a strong correlation of
either sign kills it.

**Rule 2 — the filter must do something.** Report the fraction of the ranked
top 100 that falls below the threshold and would be replaced.

- **Removes 0 of 100 → drop the filter.** It is inert at the only place it
  was asked to act.
- Removes 1–100 → report the count and the identity of what was removed.

Failing either rule drops the filter. It is not rescued by adjusting the
threshold afterwards: the threshold is set by rule 3 below, from a
distribution that does not involve the library.

**Rule 3 — the threshold is set before the library is scored**, from the
classifier's probability distribution on **held-out** reference AMPs, at the
5th percentile. Held-out matters: the 39,448 reference AMPs *are* the
training positives (they are exactly the `label == 1` rows of
`mlamp_positives.csv`), so the threshold must come from fold 0, which the
classifier never trains on. Setting it on training positives would measure
memorisation and return a saturated, meaningless cut.

## Predictions, recorded before measuring

- Spearman |rho| will land between 0.25 and 0.55 — related, since both models
  see sequence and both were fitted on overlapping notions of "good peptide",
  but not redundant, because one predicts a continuous potency against
  specific strains and the other predicts membership of a class.
- The filter will remove between 0 and 15 of the top 100. The top 100 is
  already selected for low predicted MIC, which correlates with AMP-likeness,
  so most of it should clear an AMP-probability floor set at the 5th
  percentile of real AMPs.
- The most likely failure mode is rule 2, not rule 1: an inert filter rather
  than a redundant one.

If the measured values fall outside these ranges, that is worth reporting as a
surprise rather than quietly absorbing.

## What is being deviated from, and why

This is a **fresh fit**, not a reproduction. It is stated here rather than
discovered later:

1. **No serialized `deep_fusion` checkpoint has ever existed** on this
   machine (`GRAMP/results/comparison/BLOCKERS.md`, B3). The comparison
   study's scores came from a fresh fit of the notebook's architecture,
   trained under explicit instruction as a recorded exception (B7). This is a
   *second* fresh fit. Neither is comparable to the notebook's reported
   numbers, and the notebook's figures must not be quoted beside either.

2. **Negatives are real proteins, not synthetic decoys.** The existing
   `raw/splits_with_negatives.csv` is 299,756 shuffled/mutated decoys, which
   are trivially separable and produce a saturated classifier that cannot
   order anything. This fit instead uses `processed/negatives.fasta`:
   269,833 real small proteins from sORFdb/SmProt with GenBank accessions,
   already length-filtered to 8–50 residues.

   **This is sORFdb/SmProt, not UniProt.** No UniProt or SwissProt non-AMP
   set exists on this machine. The requirement that motivated the change —
   real proteins rather than decoys — is met; the specific database is not
   the one originally named, and METHOD.md must say sORFdb/SmProt.

3. **The negative split is random, not cluster-disjoint.** The 39,448
   positives carry cluster assignments (23,510 clusters) and are split
   cluster-disjoint with fold 0 held out. The sORFdb negatives have no
   cluster assignment and neither MMseqs2 nor CD-HIT is available here, so
   they are split at random. Held-out performance on the negative class is
   therefore optimistic to an unmeasured degree, and any reported specificity
   carries that caveat.

4. **LoRA adapters are merged into the base weights before saving.** The
   recipe fine-tunes ESM-2 with LoRA, which needs `peft` at load time. Merging
   means inference needs only `transformers`, so shipping the filter does not
   add `peft` to the submission's dependency set.

## If it ships

Weights in the GitHub Release alongside the existing two, inference code in
`src/ampx`, and METHOD.md carrying the training data disclosure, the
sORFdb-not-UniProt note, the random-negative-split caveat, and the
no-checkpoint-ever-existed note from point 1.

## If it does not ship

This file stays. A filter that was built, measured and rejected is part of the
method's history, and the measurement is the reason the submission ranks the
way it does.
