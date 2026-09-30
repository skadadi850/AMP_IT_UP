"""Re-score the generated library and re-select the top 100, stratified by length.

## Why this exists

The determinism gate's top 100 had a median length of 42.5 residues against a
reference median of 18: 58% of it sat above 40 residues where the library holds
4.6% and the reference 3.5%. The library itself tracks the reference closely
(length KS 0.0792 against `data/antibacterial.fasta`), so this is a selection
artifact, not a generation one.

Two mechanisms produce it, and the rank-ordered data shows both. The oracle
scores longer peptides as more potent, and the 0.60 pairwise Levenshtein
ceiling is far easier to satisfy between long sequences than short ones, so a
greedy score-first walk under a hard diversity constraint drifts long. Median
length falls monotonically with rank across the ranked 500 while charge density
rises inversely, which is the signature of exactly that pair of pressures.

## What it does

1. Re-scores every library member with the oracle. The gate job persisted only
   FASTA, so the per-candidate scores were computed in memory and discarded;
   this writes the full table so no future re-selection needs to recompute it.
2. Selects ONCE, at the overflow length, by a greedy walk down the unchanged
   ranking key (strain-weighted mean predicted log10 MIC) under the unchanged
   eligibility floor and the unchanged identity ladder, with one added
   constraint: the two longest bands are capped at a combined 20% share.
3. Takes the top list as the first `--k` entries of that single selection, so
   the overflow continues the same ranking by construction.

It needs no GPU: scoring uses only the oracle (XGBoost plus the ESM-2
encoder), never the generator.

## What it deliberately does not do

It does not allocate quotas. An earlier version stratified all seven bands in
proportion to the reference set by largest remainder, and that is the wrong
shape of fix: it forces slots into 8-13, where only 10.9% of the library clears
the eligibility floor, and so buys a distributional match by promoting
candidates the oracle rates poorly in precisely the band where it is least
reliable.

The short bands therefore get no floor. They are not capped, but neither are
they handed slots they did not earn on predicted MIC. Admitting a candidate
BECAUSE it is short would import length into the selection rule to compensate
for the oracle having imported length into its scoring -- the same error, in
the opposite direction. One constraint, one direction.

That the short bands stay thin is a finding about the oracle, not a defect in
this script, and it belongs in the method disclosure rather than being
papered over by the allocation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ampx.compliance import read_fasta, write_fasta  # noqa: E402
from ampx.models.novelty_fast import ExhaustiveNovelty  # noqa: E402
from ampx.generate import (PANEL_WEIGHTS, REFERENCE_FASTA, REGRESSOR,  # noqa: E402
                           REGRESSOR_META, panel_success, predict_panel,
                           resolve_device)
from ampx.models.features import MAX_PEPTIDE_LENGTH, MIN_PEPTIDE_LENGTH  # noqa: E402
from ampx.predictor import Oracle  # noqa: E402
from ampx.ranking import (BAND_WIDTH, LONG_BANDS, LONG_SHARE,  # noqa: E402
                          N_BANDS, SAFETY_CEILING, Candidate, length_band,
                          select_top_diverse)

#: BAND_WIDTH and N_BANDS are imported from `ampx.ranking` rather than restated
#: here. They were previously duplicated, which is how band_label() came to
#: disagree with length_band() about where the top band ends.


#: Re-exported so this script and the entry point cannot disagree about
#: where a band starts. The definition lives with the selection that uses it.
band_of = length_band


def band_label(band: int) -> str:
    lo = MIN_PEPTIDE_LENGTH + band * BAND_WIDTH
    # The top band is open-ended: length_band() clamps with min(..., N_BANDS-1),
    # so band N_BANDS-1 absorbs everything from `lo` up to MAX_PEPTIDE_LENGTH,
    # which is wider than BAND_WIDTH. Labelling it `lo`..`lo+BAND_WIDTH-1` drops
    # the longest peptides out of their own label -- 323 of the library and 2 of
    # the submitted top 100 are 50 residues and belong to this band.
    if band == N_BANDS - 1:
        return f"{lo}-{MAX_PEPTIDE_LENGTH}"
    return f"{lo}-{lo + BAND_WIDTH - 1}"


def allocate(reference_lengths: list[int], k: int) -> dict[int, int]:
    """INFORMATIONAL ONLY -- this does not drive selection.

    Slots per band, proportional to the reference, summing to exactly k, by
    largest remainder rather than rounding (rounding each band independently
    does not sum to k).

    This was the selection rule in an earlier version and was rejected; see
    "What it deliberately does not do" in the module docstring. It survives
    solely to print the counterfactual alongside the band table, so a reader
    can see what a proportional allocation would have demanded of each band
    and compare it against what the greedy walk actually returned. Nothing
    downstream reads its result -- `select_top_diverse` is the selection.
    """
    counts = np.zeros(N_BANDS, dtype=np.int64)
    for length in reference_lengths:
        counts[band_of(length)] += 1
    share = counts / counts.sum() * k
    floor = np.floor(share).astype(np.int64)
    remainder = k - int(floor.sum())
    # Ties broken by band index so the allocation is deterministic.
    order = sorted(range(N_BANDS), key=lambda b: (-(share[b] - floor[b]), b))
    for b in order[:remainder]:
        floor[b] += 1
    return {b: int(floor[b]) for b in range(N_BANDS)}


def describe(name: str, lengths: np.ndarray) -> str:
    return (f"  {name:<22} n={len(lengths):<6} median {np.median(lengths):5.1f}  "
            f"mean {lengths.mean():5.2f}  >40 {100*(lengths>40).mean():5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=str(REPO / "run_a" / "library.fasta"))
    ap.add_argument("--reference", default=str(REFERENCE_FASTA))
    ap.add_argument("--scores-out", default=str(REPO / "results" / "selection" / "candidate_scores.csv"))
    ap.add_argument("--top-out", default=str(REPO / "results" / "selection" / "top_stratified.fasta"))
    ap.add_argument("--overflow-out", default=str(REPO / "results" / "selection" / "top500_stratified.fasta"))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--overflow-k", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--from-scores", default=None,
                    help="reuse an existing candidate_scores.csv instead of re-scoring")
    args = ap.parse_args()

    _, library = read_fasta(Path(args.library))
    _, reference = read_fasta(Path(args.reference))
    print(f"library   : {len(library)} sequences from {args.library}")
    print(f"reference : {len(reference)} sequences from {args.reference}")

    oracle = None
    if not args.from_scores:
        device = args.device or resolve_device(None)
        oracle = Oracle(str(REGRESSOR), str(REGRESSOR_META), device=device)

    if args.from_scores:
        # Re-selection is free once the table exists: the 40 minutes is all
        # ESM-2 encoding, and the scores are a deterministic function of the
        # library and the pinned checkpoint.
        print(f"\n== reusing cached scores: {args.from_scores} ==")
        import pandas as pd
        cached = pd.read_csv(args.from_scores).set_index("sequence")
        cached = cached.loc[library]
        species = list(PANEL_WEIGHTS)
        predictions = {s: cached[f"log_mic_{s.replace(' ', '_')}"].to_numpy()
                       for s in species}
    else:
        print("\n== re-scoring the library ==")
        predictions = predict_panel(library, oracle)
    metrics = panel_success(predictions)
    eligible = (metrics["gram_negative"] > 0) & (metrics["gram_positive"] > 0)
    score = -metrics["mean_log_mic"] - np.where(eligible, 0.0, 10.0)
    lengths = np.array([len(s) for s in library])
    bands = np.array([band_of(int(n)) for n in lengths])
    print(f"  eligible: {int(eligible.sum())}/{len(library)} "
          f"({100*eligible.mean():.1f}%)")

    out = Path(args.scores_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.from_scores:
        print(f"  (score table unchanged: {out})")
    else:
        species = list(PANEL_WEIGHTS)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sequence", "length", "length_band", "length_band_label"]
                       + [f"log_mic_{s.replace(' ', '_')}" for s in species]
                       + ["mean_log_mic", "success_overall", "success_gram_negative",
                          "success_gram_positive", "eligible", "panel_score"])
            for i, seq in enumerate(library):
                w.writerow([seq, int(lengths[i]), int(bands[i]), band_label(int(bands[i]))]
                           + [f"{predictions[s][i]:.6f}" for s in species]
                           + [f"{metrics['mean_log_mic'][i]:.6f}",
                              f"{metrics['overall'][i]:.6f}",
                              f"{metrics['gram_negative'][i]:.6f}",
                              f"{metrics['gram_positive'][i]:.6f}",
                              int(eligible[i]), f"{score[i]:.6f}"])
        print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(library)} rows)")

    print("\n== length bands: reference, library, eligibility ==")
    print("   (the prop. column is the REJECTED proportional allocation, shown")
    print("    for comparison only -- selection is the greedy walk below)")
    ref_lengths = [len(s) for s in reference]
    quotas = allocate(ref_lengths, args.k)
    ref_counts = np.zeros(N_BANDS, dtype=np.int64)
    for n in ref_lengths:
        ref_counts[band_of(n)] += 1
    lib_counts = np.bincount(bands, minlength=N_BANDS)
    elig_counts = np.bincount(bands[eligible], minlength=N_BANDS)
    print(f"  {'band':<8}{'reference':>12}{'prop.':>7}{'library':>10}{'eligible':>10}")
    for b in range(N_BANDS):
        print(f"  {band_label(b):<8}{ref_counts[b]:>8} {100*ref_counts[b]/len(ref_lengths):>4.1f}%"
              f"{quotas[b]:>7}{lib_counts[b]:>10}{elig_counts[b]:>10}")

    # Computed once for the whole library, exactly as generate.py does it:
    # per-candidate inside the greedy loop it would dominate the cost.
    print("\n== reference identity (precomputed once) ==")
    novelty = ExhaustiveNovelty(reference, threshold=SAFETY_CEILING)
    ref_identity = novelty.max_identity(library)
    print(f"  max {ref_identity.max():.4f}  median {np.median(ref_identity):.4f}")

    candidates = [Candidate(sequence=s, mean_score=float(score[i]))
                  for i, s in enumerate(library)]

    # ONE selection, at the overflow length. The top list is its first k
    # entries, so the overflow continues the same ranking by construction.
    # THE SAME FUNCTION THE ENTRY POINT USES. This script re-selects over an
    # already-generated library so the choice can be examined without an
    # hour-long GPU run, but it must never become a second implementation of
    # shipping logic: `generate/top.fasta` is regenerated by the organizers'
    # validator, so any selection rule living only here would be discarded on
    # their first command while appearing correct in the repository.
    print(f"\n== selection (ampx.ranking.select_top_diverse), {args.overflow_k} once ==")
    selected, info = select_top_diverse(
        candidates,
        novelty=None,
        reference_identity=ref_identity,
        k=args.overflow_k,
        min_required=args.k,
    )
    print(f"  ceiling {info['internal_ceiling']:.2f}  "
          f"relaxed={info['ladder_relaxed']}  selected {info['selected']}  "
          f"eligible {info['eligible']}  long-band members {info['long_band']}")

    ranked = [c.sequence for c in selected]
    top = ranked[:args.k]

    # The prefix invariant is NOT asserted here. `top` is `ranked[:k]` one
    # line above, so any check at this point compares a value against itself
    # and passes unconditionally -- it reads like a guard while guarding
    # nothing. What can actually go wrong is the pair of FILES disagreeing
    # after they are written, which is what the earlier two-pass version
    # produced, so the invariant is checked there instead: see
    # `_verify_overflow_prefix` in scripts/verify_submission.py.
    n_long_top = sum(band_of(len(s)) in LONG_BANDS for s in top)
    print(f"  long bands within the top {args.k}: {n_long_top} "
          f"(cap {int(LONG_SHARE*args.k)})")
    assert n_long_top <= LONG_SHARE * args.k + 1, "long-band cap violated in the top list"

    write_fasta(top, Path(args.top_out), prefix="rank")
    write_fasta(ranked, Path(args.overflow_out), prefix="rank")
    print(f"  wrote {args.top_out} and {args.overflow_out}")

    idx = {s: i for i, s in enumerate(library)}
    for name, seqs in ((f"top-{args.k}", top), (f"overflow-{len(ranked)}", ranked)):
        sl = np.array([len(s) for s in seqs])
        picked = [idx[s] for s in seqs]
        print(f"\n  -- {name}")
        print(describe("   selected", sl))
        print(describe("   reference", np.array(ref_lengths)))
        from scipy.stats import ks_2samp
        print(f"     length KS vs reference : {ks_2samp(sl, ref_lengths).statistic:.4f}")
        bands_sel = np.array([band_of(int(n)) for n in sl])
        comp = {band_label(b): int((bands_sel == b).sum()) for b in range(N_BANDS)}
        print(f"     band composition       : {comp}")
        ml = metrics["mean_log_mic"][picked]; so = metrics["overall"][picked]
        print(f"     mean_log_mic median    : {np.median(ml):+.3f}  range {ml.min():+.3f} to {ml.max():+.3f}")
        print(f"     success overall mean   : {so.mean():.4f}")


if __name__ == "__main__":
    main()
