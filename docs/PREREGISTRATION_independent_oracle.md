# Pre-registration: re-scoring every arm with an independent oracle

Written **before** the independent regressor is trained and before any arm is
re-scored. The decision rule below is fixed at the time of writing. If the
result is unwelcome, the result stands and this document does not change.

## Why this test exists

Every number in `contrasts.csv` is the opinion of one XGBoost regressor,
`mic_regressor.json`. The seeded arms did not merely get scored by it — they
were **optimised against it**. `run_refinement.py` carries forward
`argmin(predicted_log_mic)` over 32 derivatives, five times in sequence. Taking
the minimum of 32 noisy predictions selects for favourable prediction error as
much as for real potency, and five rounds compound it.

The control arms did not receive equal treatment:

| arm | selection against the oracle |
|---|---|
| A, B, D, N | 5 sequential rounds of best-of-32 |
| C (fresh samples) | none during generation; one best-of-budget at analysis |

So the seeded arms have had strictly more opportunity to accumulate
oracle-specific error, and **that bias points toward the conclusion the project
wants** (arm B beating arm C by +0.178). A result whose principal confound
favours the hypothesis cannot be defended until the confound is removed.

## What is being tested

Whether the headline contrasts survive an oracle that **no arm selected
against**.

Primary: **B vs C** — inpainting the model's own best output against fresh
conditioned sampling at matched budget. Currently +0.178 log10, 82 wins / 7
losses. This is the claim the pipeline makes.

Secondary, reported whatever they do, not promoted to claims:
- **A vs D** — inpainting vs random mutation at matched edits (currently
  +0.107 uniform, +0.138 attribution).
- **A vs C** — inpainting a known AMP vs fresh sampling (currently **−0.206**,
  a loss; it is reported because it is a loss).
- **A vs D in the `<0.70` band** (currently **−0.122**, a reversal).

## Construction

1. Train a second regressor from `train_mic_regressor.py` with `--seed 17`,
   which redraws **both** the held-out cluster assignment and the XGBoost seed.
   Same MIC table, same 40%-identity clustering, same feature builder.
2. Re-score, with that model, every candidate already produced by the pilot,
   plus the seed peptides in `seeds_a.csv` / `seeds_b.csv`.
3. **The arms are not re-run.** Their trajectories were produced under the
   original oracle and stay exactly as they were. Only the *evaluation* changes.
   Re-running selection under the new oracle would measure a different
   experiment and would reintroduce the same confound against the new model.
4. Re-run `analyze_refinement.py` unmodified on the re-scored inputs, so every
   pairing, budget match, band stratification and resampling rule is the code
   already in use rather than a second implementation written for this test.
5. The novelty gate (`submittable`, `max_reference_identity`) is unchanged: it
   is computed by alignment against the reference corpus and no oracle is
   involved.

### Acknowledged limit

The second regressor is trained on the same MIC table, so it is *independent of
the selection*, not statistically independent of the data. It cannot detect a
bias shared by any model fit to this corpus. It can only detect error specific
to the model the arms exploited — which is exactly the confound named above.
Any claim arising from this test is stated with that scope and not beyond it.

## Decision rule

Let `d_orig` = +0.178 be the current B-vs-C advantage and `d_indep` the same
quantity under the independent oracle, both in log10 MIC (uM), both from the
same paired best-of-budget construction.

| outcome | condition | what is claimed |
|---|---|---|
| **Survives** | `d_indep >= 0.089` (at least half of `d_orig`) **and** the paired win count stays a majority (> 50% of peptide seeds) | B-vs-C is reported as a real result, with both the original and independent numbers shown side by side |
| **Weakened** | `0 < d_indep < 0.089` | reported as directional only. The headline drops the effect size and states that most of the original gap was oracle exploitation |
| **Collapses** | `d_indep <= 0` **or** the win count falls to a minority | the B-vs-C claim is **withdrawn**. `HANDOFF.md` and the figures are corrected to say the pilot does not show inpainting beating fresh sampling |

Fixed before the test:

- Thresholds are as above. They are not renegotiated after seeing `d_indep`.
- The regressor seed is **17**, chosen before training and not swept. If a
  second seed is ever run it is reported alongside, never substituted for this
  one, and never used to pick the better of two.
- The independent model must clear the same gate the original did
  (`MIN_SPEARMAN = 0.4` on its own held-out clusters, label-shuffled control
  near zero). A model that fails its own validation is not a valid referee, and
  the test is re-planned rather than reported.
- If the two oracles disagree about the *direction* of A-vs-D as well, that is
  reported as instability of the whole potency axis, not quietly dropped.

## Amendment, recorded before any re-scoring ran

The first attempt to train the seed-17 model surfaced a defect in
`train_mic_regressor.py` that invalidates the premise of this test as written
above, so the test was rebuilt before it produced any contrast.

`load_clusters` keyed its map by mmseqs2 **identifiers** (`seq1|mic+ref`) while
the MIC table is keyed by **sequence**, so `table["sequence"].map(clusters)`
matched nothing. Every one of the 28,604 rows fell through the singleton
fallback, 5,602 sequences became 5,602 one-member "clusters", and the
40%-identity cluster split silently became a random split by sequence. The true
structure, recovered by joining through `cluster_input.fasta`, is **2,824
clusters, with 3,487 of 5,602 sequences (62%) sharing one with a relative** —
one cluster holds 59.

This affects the original oracle (seed 0) exactly as much as the new one: the
same line produced the same degenerate split when the pilot's regressor was
trained. Both reported held-out Spearmans (0.6183 original, 0.6256 seed 17) are
therefore inflated by near-duplicate leakage of unknown size, and the
**label-shuffled control could not catch it** — that control detects label
leakage, and identity leakage with shuffled labels still reads ~0, so its
0.0601 was false assurance.

Changes made, all before results:

1. `load_clusters` now joins through the clustered FASTA and fails loudly if
   the cluster table and FASTA disagree.
2. A new `MAX_UNASSIGNED_FRACTION = 0.05` guard stops the run when the
   singleton fallback would define the split instead of patching it. The
   silent fallback is what hid this for the entire pilot.
3. Both models are retrained under the corrected split: seed 17 as the
   independent referee, and **seed 0 re-validated**, because the honest quality
   of the oracle the arms were optimised against is now an open question.

Consequences for this test, stated in advance:

- The gate is re-applied to the corrected seed-17 model. Its Spearman is
  expected to **fall**, because the leakage that inflated it is gone. If it
  falls below `MIN_SPEARMAN = 0.4` the model is not a valid referee and this
  test is re-planned rather than reported, exactly as the original rule says.
- The decision rule and its thresholds are **unchanged**. They were fixed
  before any number was seen and this amendment does not touch them.
- The re-validated seed-0 number is reported whatever it shows. If the pilot's
  oracle is substantially weaker than 0.618 under an honest split, that is a
  finding about every result in `contrasts.csv`, not only about this test.

## What this test cannot do

It does not establish potency. It establishes only whether a ranking survives a
change of scorer. Both models are regressors over the same corpus, and no MIC
has been measured in a laboratory anywhere in this project. Every claim remains
a claim about predicted values.
