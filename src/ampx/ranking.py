"""Selection of the top-100 list.

## Why this module is not just `sorted(scores)[:100]`

Read the Phase 2 scoring rule carefully:

    "From each qualifying team's top-100 list, **25 peptides are drawn
    uniformly at random**. ... Team-level scores for each of the five
    categories are the **arithmetic mean** of the relevant peptide metric
    across the team's 25 peptides."

Two consequences that reshape the whole selection problem:

1. **You are scored on the mean of your entire top-100, not on your best
   candidates.** The 25 drawn peptides are an unbiased sample of the list, so
   E[team score] = mean over all 100. A list with 10 spectacular peptides and
   90 mediocre ones scores the same as 100 mediocre ones. There is no upside to
   ranking *within* the list -- rank order is documentation, not scoring.

2. **Downside risk is not diversified away.** With n=25 out of 100, the
   sampling variance of your score is substantial. MIC is censored at 64 uM,
   so one inactive peptide contributes a full 64 to the mean and cannot be
   offset by a peptide at 0.5. The objective is therefore closer to
   *minimising expected censored MIC* than to *maximising predicted potency* --
   which means avoiding confident-but-wrong picks matters more than chasing the
   highest point estimates.

The practical rule: **maximise the floor of the list, not its ceiling.**
Prefer 100 peptides you are 85% sure are active over 100 peptides with higher
mean predicted potency but 25% predicted-inactive. `select_top` below implements
this as a lower-confidence-bound objective with a diversity constraint.

## Why diversity still matters inside the top-100

Correlated failure is the real risk. If all 100 candidates are minor variants
of one motif and that motif turns out to be inactive under the assay's exact
conditions, the entire team score collapses together. Clustering the candidate
pool and capping per-cluster representation buys independence between your
bets, at a small cost in mean predicted score. This is portfolio construction,
not novelty for its own sake.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compliance import SAFETY_CEILING, max_identity


@dataclass
class Candidate:
    """A generated peptide with its surrogate predictions."""

    sequence: str
    #: Mean predicted activity across the surrogate ensemble. Higher is better.
    mean_score: float
    #: Disagreement across the ensemble; a proxy for epistemic uncertainty.
    score_std: float = 0.0
    #: Cluster assignment used to enforce structural spread.
    cluster: int = -1

    def lcb(self, kappa: float) -> float:
        """Lower confidence bound on activity.

        `kappa` sets risk aversion. kappa=0 recovers naive greedy ranking by
        mean score. kappa=1-2 penalises candidates the ensemble disagrees on,
        which is exactly the regime the mean-of-25 scoring rewards.
        """
        return self.mean_score - kappa * self.score_std


def select_top(
    candidates: list[Candidate],
    references: list[str] | None = None,
    k: int = 100,
    kappa: float = 1.5,
    max_per_cluster: int = 4,
    identity_ceiling: float = SAFETY_CEILING,
    novelty=None,
) -> list[Candidate]:
    """Choose `k` candidates maximising the risk-adjusted floor of the list.

    Greedy pass over candidates sorted by lower confidence bound, subject to
    two constraints:

      * at most `max_per_cluster` candidates from any one cluster, which caps
        correlated failure; and
      * the hard novelty ceiling, checked lazily so the expensive similarity
        scan only runs on candidates that would otherwise be selected.

    Lazy novelty checking matters in practice: verifying all 50,000 library
    members against ~39k references is billions of alignments, while verifying
    only the few hundred candidates you actually consider is seconds.

    Pass `novelty` (an `ampx.models.novelty_fast.ExhaustiveNovelty`) in
    preference to `references`. Both compute the same quantity -- the maximum
    Levenshtein ratio against every reference -- but the index does it in
    batched native code behind a lossless length prefilter, while the
    `references` path is a Python loop retained for callers that have only a
    list. Neither is a shortlist: one sequence over the ceiling invalidates
    the submission, so this gate must see every reference.
    """
    ordered = sorted(candidates, key=lambda c: c.lcb(kappa), reverse=True)

    chosen: list[Candidate] = []
    per_cluster: dict[int, int] = {}

    for cand in ordered:
        if len(chosen) >= k:
            break
        if cand.cluster >= 0 and per_cluster.get(cand.cluster, 0) >= max_per_cluster:
            continue
        if novelty is not None:
            if float(novelty.max_identity([cand.sequence])[0]) > identity_ceiling:
                continue
        elif references is not None:
            if max_identity(cand.sequence, references, identity_ceiling) > identity_ceiling:
                continue
        chosen.append(cand)
        per_cluster[cand.cluster] = per_cluster.get(cand.cluster, 0) + 1

    if len(chosen) < k:
        raise ValueError(
            f"only {len(chosen)} of {k} slots filled. Relax max_per_cluster "
            f"(currently {max_per_cluster}) or widen the candidate pool."
        )

    return chosen


def expected_team_score(
    predicted: np.ndarray,
    n_draw: int = 25,
    n_boot: int = 20_000,
    seed: int = 42,
) -> dict[str, float]:
    """Simulate the Phase 2 draw to estimate score mean and spread.

    Use this to compare two candidate top-100 lists honestly. A list with a
    better mean but a much worse 5th percentile is often the wrong choice: the
    competition runs the draw exactly once, so you are exposed to that tail.

    Args:
        predicted: length-100 array of predicted per-peptide metric (e.g.
            censored MIC in uM, where lower is better).
        n_draw: peptides drawn per team.
        n_boot: bootstrap replicates.
        seed: RNG seed.

    Returns:
        Mean, standard deviation, and 5th/95th percentiles of the team score.
    """
    rng = np.random.default_rng(seed)
    predicted = np.asarray(predicted, dtype=float)
    draws = np.array(
        [rng.choice(predicted, size=n_draw, replace=False).mean() for _ in range(n_boot)]
    )
    return {
        "mean": float(draws.mean()),
        "std": float(draws.std()),
        "p05": float(np.percentile(draws, 5)),
        "p95": float(np.percentile(draws, 95)),
    }
