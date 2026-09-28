"""Competition entry point. `uv run generate` must produce identical output every time.

Writes two files into `generate/`:

    generate/library.fasta   50,000 unique compliant sequences
    generate/top.fasta       100 ranked candidates

## Reproducibility contract

The organizers verify by running `uv sync`, then the entry point, twice, and
byte-comparing the outputs. Sources of nondeterminism that will fail you:

  * unseeded RNG anywhere in the sampling path (including dataloader shuffling
    if you generate inside a training loop);
  * `set` / `dict` iteration order feeding into output order -- insertion order
    is stable in modern Python but set order is *not*, so never iterate a set
    to build the library;
  * non-deterministic GPU kernels. Call `torch.use_deterministic_algorithms(True)`
    and set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before any CUDA work;
  * multi-worker generation where results are collected in completion order
    rather than sorted back into a canonical order;
  * **model construction drawing from the global RNG**. Loading a checkpoint
    can initialise layers absent from the weights, consuming RNG draws and
    shifting everything sampled afterwards. `main()` therefore loads every
    model *before* seeding, never after.

The safest architecture, and the one used here: generate to a list, sort
deterministically, then write. Never let concurrency touch output ordering.

## Pipeline

1. `load_models` builds the masked-diffusion generator, the MIC oracle and the
   exhaustive novelty index. No RNG is touched here.
2. `set_global_determinism` pins every stream, once, afterwards.
3. `generate_library` samples in rounds against a length grid, and after each
   round drops sequences that are non-compliant, duplicated, an exact
   reference match, or above the 80% identity ceiling. Rounds continue until
   the library is full, each with its own derived seed so the result is a
   function of the base seed alone.
4. `score_candidates` scores survivors with the oracle.
5. `select_top` picks the hundred, re-checking novelty exhaustively.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from .compliance import (
    SAFETY_CEILING,
    LIBRARY_SIZE,
    TOP_SIZE,
    read_fasta,
    verify_library,
    verify_top,
    write_fasta,
)
from .models.compliance import compliance_report
from .models.features import LENGTH_SPAN, MAX_PEPTIDE_LENGTH, MIN_PEPTIDE_LENGTH
from .models.masked_diffusion import MaskedDiffusionModel
from .models.novelty_fast import ExhaustiveNovelty
from .models.sampling import (build_condition, empirical_length_counts,
                             sample_library)
from .predictor import Oracle, resolve_device
from .ranking import (INTERNAL_IDENTITY_LADDER, Candidate,
                      select_top_diverse)
from .weights import ensure_weights

#: Every stochastic component must derive from this one value. It is a
#: module-level default rather than a required argument because the
#: reproducibility requirement is that running the script *with no arguments*
#: twice produces identical output.
DEFAULT_SEED = 42

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
REFERENCE_FASTA = _REPO_ROOT / "data" / "antibacterial.fasta"
CHECKPOINTS = _REPO_ROOT / "checkpoint"
GENERATOR_CKPT = CHECKPOINTS / "masked_diffusion_best.pt"
SPECIES_VOCAB = CHECKPOINTS / "species_vocab.json"
REGRESSOR = CHECKPOINTS / "mic_regressor.json"
REGRESSOR_META = CHECKPOINTS / "mic_regressor_meta.json"
#: Length calibration, most specific first. The stage-5b file is fitted
#: against the checkpoint we actually ship; `length_calibration.json` was
#: fitted against the face_only weights and is a fallback, not an equivalent.
#: Length distribution is scored directly in Phase 1, so using a pad bias
#: derived from different weights would skew all 50,000 sequences on a scored
#: axis. `scripts/hpc/60_calibrate_and_generate.sbatch` writes the stage-5b
#: file, and generation in that job picks it up here.
LENGTH_CALIBRATIONS = (
    CHECKPOINTS / "length_calibration_stage5b.json",
    CHECKPOINTS / "length_calibration.json",
)

#: Sampler settings. `guidance_weight`, `steps`, `temperature` and
#: `batch_size` mirror the defaults in `scripts/generate_masked.py`, so the
#: entry point and the research script sample the same way.
GUIDANCE_WEIGHT = 2.0
SAMPLING_STEPS = 128
TEMPERATURE = 1.0
BATCH_SIZE = 512

#: Oversample factor per round. Compliance, dedup and the novelty ceiling all
#: reject candidates, so a round must ask for more than it needs.
OVERSAMPLE = 1.6
MAX_ROUNDS = 12

#: The competition's strain panel (Appendix B of the competition document),
#: collapsed to the species our regressor covers and weighted by how many
#: strains of each appear in the panel. Ranking reproduces Overall Success
#: Rate -- the fraction of the 20 strains a peptide inhibits at or below
#: 16 uM -- rather than potency against any single organism.
#:
#: Two deliberate departures from the literal panel:
#:
#:   * Five species the regressor was trained on are absent here (C. albicans,
#:     S. epidermidis, M. luteus, B. cereus, L. monocytogenes). None is assayed
#:     by the competition, so scoring them would dilute the aggregate with
#:     organisms that earn nothing.
#:   * E. faecium (1 strain, VRE) is on the panel but absent from the
#:     regressor's species vocabulary. Its strain is folded into E. faecalis,
#:     the nearest covered organism -- same genus, also a VRE isolate on the
#:     panel. That is a documented approximation, not a prediction for
#:     E. faecium: it assumes the two enterococci respond similarly, which is
#:     plausible but unverified here.
#:
#: Weights therefore sum to 20, the full panel.
PANEL_WEIGHTS: dict[str, int] = {
    # Gram-negative: 15 of 20 strains
    "escherichia coli": 5,
    "pseudomonas aeruginosa": 3,
    "klebsiella pneumoniae": 2,
    "acinetobacter baumannii": 2,
    "salmonella enterica": 2,
    "enterobacter cloacae": 1,
    # Gram-positive: 5 of 20 strains, including E. faecium folded into
    # E. faecalis (1 + 1 = 2).
    "staphylococcus aureus": 2,
    "enterococcus faecalis": 2,
    "bacillus subtilis": 1,
}

#: Species above that are Gram-negative. Used for the eligibility floor and
#: for the per-class diagnostics.
PANEL_GRAM_NEGATIVE = frozenset({
    "escherichia coli", "pseudomonas aeruginosa", "klebsiella pneumoniae",
    "acinetobacter baumannii", "salmonella enterica", "enterobacter cloacae",
})

#: Potency threshold, in log10 micromolar. A strain counts toward success rate
#: when predicted MIC is at or below 16 uM.
POTENCY_THRESHOLD_UM = 16.0
LOG_POTENCY_THRESHOLD = float(np.log10(POTENCY_THRESHOLD_UM))

# --------------------------------------------------------------------------- #
# Conditioning grid
#
# The first full run sampled with an empty request, which left every axis at
# index 0 or mask 0 and made the run very nearly unconditional -- length was
# the only axis carrying signal. On `is_amp` that was worse than neutral: the
# axis is exempt from conditional dropout, so its index-0 row never receives a
# gradient anywhere in training, and the model was being fed an untrained
# vector on the one axis it always saw.
#
# Values below are quoted in the units `build_condition` expects, which are
# not uniform: `mic_um`/`hc50_um` are raw micromolar, `gram_selectivity` is a
# raw log10 ratio, `gravy` is a raw GRAVY score, while `charge_per_res` and
# `mu_h` are passed through already normalized onto [0, 1]. Ranges are taken
# from the generator's own training corpus so the grid stays on-distribution.
# --------------------------------------------------------------------------- #

#: Strain counts double as sampling weights, reproducing the panel's 15:5
#: Gram-negative:Gram-positive composition. `build_condition` derives the
#: `gram` axis from the species, so it does not need to be specified.
GRID_SPECIES = tuple(PANEL_WEIGHTS.items())

#: Target MIC. All at or below the competition's 16 uM threshold, spanning the
#: training corpus's 5th to 50th percentile (0.78 to 12.5 uM).
GRID_MIC_UM = (2.0, 4.0, 8.0)

#: Target HC50, high meaning non-hemolytic. Above the training median of
#: 105 uM and below its 95th percentile of 258 uM.
GRID_HC50_UM = (128.0, 256.0)

#: log10(MIC gram-positive / MIC gram-negative). Positive favours activity
#: against Gram-negatives, which is 15 of the 20 panel strains; the negative
#: rung is retained because Gram-Positive Activity is a separate award and the
#: eligibility floor requires activity in both classes. Training spans
#: -0.92 to +0.80.
GRID_SELECTIVITY = ((-0.5, 25), (0.0, 40), (0.5, 35))

#: Normalized charge per residue; training p5/p50/p95 are 0.460/0.591/0.750.
#: Deliberately stopping short of the top of the range: a library collapsed
#: onto one high-charge mode scores badly on diversity even when every member
#: is individually plausible.
GRID_CHARGE_PER_RES = (0.50, 0.59, 0.68)

#: Normalized hydrophobic moment; training p5/p50/p95 are 0.187/0.536/0.925.
GRID_MU_H = (0.30, 0.55, 0.80)

#: Raw GRAVY; training p5/p50/p95 are -1.94/-0.16/+1.32.
GRID_GRAVY = (-1.2, -0.2, 0.8)

#: Cysteine class 1 is zero-cysteine. Class 2 (two cysteines) is excluded on
#: compliance grounds rather than modelling grounds: two free cysteines on a
#: linear peptide oxidise to an intramolecular disulfide under standard
#: synthesis and handling, and the rules require linear peptides. Its share of
#: the training corpus is redistributed across the rest of the grid.
GRID_CYS_CLASS = 1


def set_global_determinism(seed: int) -> None:
    """Pin every RNG we might touch.

    Call this *after* all model loading, not before: `from_pretrained` and
    checkpoint loads can initialise missing layers, and those draws would
    otherwise advance the global stream between seeding and sampling.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_models(device: str):
    """Build the generator, the oracle and the novelty index.

    Deliberately free of RNG use, so `set_global_determinism` can run after it.
    """
    species_vocab = json.loads(SPECIES_VOCAB.read_text())
    model = MaskedDiffusionModel.load(str(GENERATOR_CKPT), device=device)
    model.eval()

    oracle = Oracle(str(REGRESSOR), str(REGRESSOR_META), device=device)

    _, references = read_fasta(REFERENCE_FASTA)
    novelty = ExhaustiveNovelty(references, threshold=SAFETY_CEILING)

    calibration = _load_calibration()
    return model, species_vocab, oracle, novelty, references, calibration


def _load_calibration() -> dict:
    """Load the most specific length calibration available."""
    for path in LENGTH_CALIBRATIONS:
        if path.exists():
            calibration = json.loads(path.read_text())
            print(
                f"length  : pad_bias {calibration.get('best_pad_bias', 0.0):+.2f}, "
                f"reveal {calibration.get('reveal', 'uniform')} "
                f"(from {path.name})"
            )
            return calibration
    raise SystemExit(
        "no length calibration found; expected one of: "
        + ", ".join(str(p) for p in LENGTH_CALIBRATIONS)
    )


def build_request_grid() -> list[tuple[dict, int]]:
    """The conditioning cells, each with an integer weight.

    Cells are the full cross product of the swept axes, so ordering is a
    deterministic function of the constants above and does not depend on any
    RNG. Weights multiply, and counts are allocated from them by largest
    remainder so the totals land exactly.
    """
    cells: list[tuple[dict, int]] = []
    for species, species_w in GRID_SPECIES:
        for mic in GRID_MIC_UM:
            for hc50 in GRID_HC50_UM:
                for selectivity, sel_w in GRID_SELECTIVITY:
                    for charge in GRID_CHARGE_PER_RES:
                        for mu_h in GRID_MU_H:
                            for gravy in GRID_GRAVY:
                                cells.append((
                                    {
                                        "is_amp": "amp",
                                        "species": species,
                                        "mic_um": mic,
                                        "mic_censored": False,
                                        "hc50_um": hc50,
                                        "hc50_source": "predicted",
                                        "gram_selectivity": selectivity,
                                        "charge_per_res": charge,
                                        "mu_h": mu_h,
                                        "gravy": gravy,
                                        "cys_class": GRID_CYS_CLASS,
                                    },
                                    species_w * sel_w,
                                ))
    return cells


def _allocate(weights: list[int], total: int) -> list[int]:
    """Split `total` across `weights`, summing exactly. Largest remainder."""
    weight_sum = float(sum(weights))
    exact = [total * w / weight_sum for w in weights]
    counts = [int(x) for x in exact]
    remainder = total - sum(counts)
    if remainder:
        # Deterministic tie-break on index, so the same weights always give
        # the same allocation.
        order = sorted(range(len(weights)),
                       key=lambda i: (-(exact[i] - counts[i]), i))
        for i in order[:remainder]:
            counts[i] += 1
    return counts


def conditioned_grid(
    total: int, lengths: np.ndarray, species_vocab: dict, device: str
):
    """Build one condition covering the whole grid, with per-sample lengths.

    `build_condition` is called once per cell rather than once per sample --
    4,374 calls instead of 80,000 -- and the length axis is then overwritten
    vectorized. Crossing length into the cells themselves would multiply the
    call count by the 43 distinct lengths for no benefit, since length is
    independent of the other axes here.
    """
    import torch

    cells = build_request_grid()
    counts = _allocate([w for _, w in cells], total)
    if len(lengths) != total:
        raise ValueError(f"expected {total} lengths, got {len(lengths)}")

    blocks = []
    cursor = 0
    for (request, _), count in zip(cells, counts):
        if count <= 0:
            continue
        block = build_condition(request, count, species_vocab, device=device)
        span = lengths[cursor:cursor + count]
        cursor += count
        normalized = np.clip(
            (span.astype(np.float64) - MIN_PEPTIDE_LENGTH) / LENGTH_SPAN, 0.0, 1.0
        )
        block["length_norm"] = torch.tensor(
            normalized, dtype=torch.float32, device=device
        ).view(count, 1)
        block["length_norm_mask"] = torch.ones(count, 1, device=device)
        blocks.append(block)

    keys = blocks[0].keys()
    return {k: torch.cat([b[k] for b in blocks], dim=0) for k in keys}


def _structural_cluster(sequence: str) -> int:
    """Coarse deterministic bucket used to cap correlated failure in the top 100.

    Length band crossed with net-charge band. This is a placeholder for the
    real diversity constraint, which is Phase 5's deliverable; it exists so
    `select_top`'s `max_per_cluster` does something meaningful rather than
    partitioning on an arbitrary index.
    """
    charge = sum(c in "KR" for c in sequence) - sum(c in "DE" for c in sequence)
    length_band = min((len(sequence) - MIN_PEPTIDE_LENGTH) // 6, 6)
    charge_band = int(np.clip(charge, -2, 12)) // 3
    return int(length_band * 8 + charge_band)


def generate_library(
    n_sequences: int,
    seed: int,
    model,
    species_vocab: dict,
    novelty: ExhaustiveNovelty,
    references: list[str],
    calibration: dict,
    device: str,
    request: dict | None = None,
    max_length: int = MAX_PEPTIDE_LENGTH,
) -> list[str]:
    """Sample until `n_sequences` unique, compliant, novel sequences exist."""
    request = dict(request or {})
    reference_exact = set(references)

    # Lengths are drawn from the reference distribution, so capping the
    # requested length means capping that distribution at the source rather
    # than filtering afterwards: a post-hoc filter would leave the shortfall
    # unfilled and quietly shrink the library. At the default of
    # MAX_PEPTIDE_LENGTH this is a no-op, since the reference set is already
    # bounded at 50, and generation is byte-identical to not passing it.
    max_length = min(int(max_length), MAX_PEPTIDE_LENGTH)
    if max_length < MIN_PEPTIDE_LENGTH:
        raise SystemExit(
            f"--length {max_length} is below the {MIN_PEPTIDE_LENGTH}-residue "
            "competition floor"
        )
    reference_lengths = [len(s) for s in references if len(s) <= max_length]
    if not reference_lengths:
        raise SystemExit(
            f"--length {max_length} leaves no reference sequences to draw a "
            "length distribution from"
        )

    # The calibration ships `reveal`/`best_pad_bias` fitted by
    # scripts/calibrate_length.py. pad_bias is only meaningful with
    # free_length, which is the configuration it was fitted under.
    reveal = calibration.get("reveal", "uniform")
    pad_bias = float(calibration.get("best_pad_bias", 0.0))

    accepted: list[str] = []
    seen: set[str] = set()

    for round_index in range(MAX_ROUNDS):
        remaining = n_sequences - len(accepted)
        if remaining <= 0:
            break
        ask = int(remaining * OVERSAMPLE) + 1

        # Lengths are drawn to match the reference distribution and then
        # shuffled, so every conditioning cell sees the whole length range
        # rather than a contiguous slice of it. Seeded per round, so the
        # assignment is a function of the base seed alone.
        counts = empirical_length_counts(reference_lengths, ask)
        pool = np.repeat(
            np.array(list(counts.keys()), dtype=np.int64),
            np.array(list(counts.values()), dtype=np.int64),
        )
        np.random.default_rng(seed + round_index).shuffle(pool)
        condition = conditioned_grid(len(pool), pool, species_vocab, device=device)

        raw = sample_library(
            model,
            condition,
            guidance_weight=GUIDANCE_WEIGHT,
            steps=SAMPLING_STEPS,
            temperature=TEMPERATURE,
            batch_size=BATCH_SIZE,
            # Derived, not reused: identical seeds would make every round draw
            # the same sequences and the loop would never converge.
            seed=seed + round_index,
            # The requested length is a hard constraint. With free_length the
            # model places PAD itself, and a positive pad bias then overrides
            # the request: the previous run asked for 12% of samples above 29
            # residues and produced none, losing the whole 30-50 band that the
            # reference set occupies. Verified on CPU that this reproduces the
            # requested length exactly at 10, 25 and 45.
            free_length=False,
            reveal=reveal,
            pad_bias=0.0,
        )

        # Cheap filters first, so the identity scan only sees survivors.
        fresh: list[str] = []
        for seq in raw:
            if seq in seen or seq in reference_exact:
                continue
            if not all(compliance_report(seq).values()):
                continue
            seen.add(seq)
            fresh.append(seq)

        if fresh:
            keep = novelty.is_novel(fresh)
            accepted.extend(seq for seq, ok in zip(fresh, keep) if ok)

        print(
            f"round {round_index}: sampled {len(raw)}, "
            f"{len(fresh)} new and compliant, library now {len(accepted)}"
        )

    if len(accepted) < n_sequences:
        raise RuntimeError(
            f"produced only {len(accepted)} of {n_sequences} sequences after "
            f"{MAX_ROUNDS} rounds. Raise MAX_ROUNDS or OVERSAMPLE, or widen "
            "the conditioning grid."
        )

    # Canonical ordering makes byte-identical output independent of how the
    # sequences were produced. Do not remove.
    return sorted(accepted[:n_sequences])


def predict_panel(sequences: list[str], oracle: Oracle) -> dict[str, np.ndarray]:
    """Predicted log10 MIC for every panel species, one array per species."""
    return {
        species: np.asarray(
            oracle.predict(sequences, [species] * len(sequences)), dtype=np.float64
        )
        for species in PANEL_WEIGHTS
    }


def panel_success(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Strain-weighted success rates from per-species predictions.

    Returns `overall`, `gram_negative` and `gram_positive` success rates, each
    the fraction of strains in that class predicted at or below 16 uM, plus
    `mean_log_mic` for tie-breaking.
    """
    names = list(PANEL_WEIGHTS)
    weights = np.array([PANEL_WEIGHTS[s] for s in names], dtype=np.float64)
    is_negative = np.array([s in PANEL_GRAM_NEGATIVE for s in names])
    # rows: species, columns: candidates
    stack = np.vstack([predictions[s] for s in names])
    hit = stack <= LOG_POTENCY_THRESHOLD

    def rate(mask: np.ndarray) -> np.ndarray:
        w = weights[mask]
        return (hit[mask] * w[:, None]).sum(axis=0) / w.sum()

    return {
        "overall": rate(np.ones(len(names), dtype=bool)),
        "gram_negative": rate(is_negative),
        "gram_positive": rate(~is_negative),
        "mean_log_mic": (stack * weights[:, None]).sum(axis=0) / weights.sum(),
    }


def score_candidates(sequences: list[str], oracle: Oracle) -> list[Candidate]:
    """Rank candidates by predicted breadth across the competition panel.

    Four of the five award categories score success rate across a strain
    panel, so the ranked quantity is the strain-weighted fraction of the panel
    a peptide is predicted to inhibit at or below 16 uM, with mean predicted
    log10 MIC as the tie-break. A peptide that is exceptional against one
    organism and inert against the rest scores poorly here, which is the
    intent: 25 of the hundred are drawn at random, so the list needs a floor.

    Candidates failing the eligibility floor -- at least one Gram-negative and
    at least one Gram-positive strain below threshold -- are given a score
    below every eligible candidate rather than being dropped, so
    `select_top` always has enough to fill `k` and the exclusion is visible
    in the ordering instead of silently shrinking the pool.

    `score_std` is 0.0 because the oracle is a single regressor, not an
    ensemble. That makes `Candidate.lcb` collapse to the mean and the `kappa`
    risk-aversion knob inert; it stays wired so an ensemble can be dropped in
    without touching the selection code.
    """
    metrics = panel_success(predict_panel(sequences, oracle))
    overall = metrics["overall"]
    eligible = (metrics["gram_negative"] > 0) & (metrics["gram_positive"] > 0)

    # Primary key is the strain-weighted mean predicted log10 MIC, negated so
    # that higher is better. Success rate was the natural objective -- it is
    # what four of the five award categories score -- but it does not
    # discriminate among the candidates that actually compete. It is not
    # saturated across the library: only about 40% of candidates fall below
    # 16 uM on the weighted mean. It saturates among the top-ranked, which are
    # the only ones in contention for 100 slots out of 50,000, and a key that
    # is constant over every contender orders nothing. Mean predicted MIC
    # keeps its variance and carries the same strain weighting, so the
    # ranking discriminates again without touching the grid.
    #
    # Success rate is still computed and reported, as a diagnostic on whether
    # the objective has collapsed.
    score = -metrics["mean_log_mic"] - np.where(eligible, 0.0, 10.0)

    print(
        f"panel   : {int(eligible.sum())}/{len(sequences)} candidates clear the "
        f"eligibility floor (>=1 Gram-negative and >=1 Gram-positive strain "
        f"<= {POTENCY_THRESHOLD_UM:g} uM)"
    )
    print(
        f"          mean success rate: overall {overall.mean():.3f}, "
        f"Gram-negative {metrics['gram_negative'].mean():.3f}, "
        f"Gram-positive {metrics['gram_positive'].mean():.3f}"
    )
    saturated = float((overall >= 1.0).mean())
    print(
        f"          success rate saturated at 1.00 for {100*saturated:.1f}% of "
        f"candidates; ranking on mean predicted log10 MIC "
        f"(median {np.median(metrics['mean_log_mic']):.3f})"
    )

    candidates = [
        Candidate(
            sequence=seq,
            mean_score=float(value),
            score_std=0.0,
            cluster=_structural_cluster(seq),
        )
        for seq, value in zip(sequences, score)
    ]
    metrics["eligible"] = eligible
    return candidates, metrics


def _report_selection(top, ranked, metrics, info, reference_identity, library) -> None:
    """Print the diagnostics the selection has to be judged on.

    The panel score takes only 21 distinct values, so ties are enormous. If
    the selected hundred all sit in one or two tiers then the ordering was in
    practice decided by the tie-break -- mean predicted MIC, the noisier
    quantity -- and that should be visible rather than implied.
    """
    index = {seq: i for i, seq in enumerate(library)}
    picked = [index[c.sequence] for c in top]
    overall = metrics["overall"][picked]
    ident = np.asarray(reference_identity)[picked]

    print(f"\nselection: internal identity ceiling {info['internal_ceiling']:.2f}"
          f"{' (RELAXED from %.2f)' % INTERNAL_IDENTITY_LADDER[0] if info['ladder_relaxed'] else ''}"
          f", {info['eligible']}/{len(ranked)} ranked candidates eligible")

    tiers, counts = np.unique(np.round(overall, 4), return_counts=True)
    print(f"  panel score distribution across the selected {len(top)}:")
    for tier, count in sorted(zip(tiers, counts), key=lambda x: -x[0]):
        bar = "#" * int(round(40 * count / max(counts)))
        print(f"    {tier:.2f}  n={count:3d}  {bar}")
    if len(tiers) <= 2:
        print("    NOTE: the selection occupies <=2 score tiers, so the ordering "
              "was effectively decided by mean predicted MIC, not by breadth.")

    print(f"  Gram-negative success rate: mean {metrics['gram_negative'][picked].mean():.3f}")
    print(f"  Gram-positive success rate: mean {metrics['gram_positive'][picked].mean():.3f}")
    picked_mic = metrics["mean_log_mic"][picked]
    print(f"  mean predicted log10 MIC  : median {np.median(picked_mic):.3f}  "
          f"range {picked_mic.min():.3f} to {picked_mic.max():.3f}  "
          f"(library median {np.median(metrics['mean_log_mic']):.3f})")
    print(f"  max identity vs reference : max {ident.max():.4f}  "
          f"median {np.median(ident):.4f}  min {ident.min():.4f}")


def main() -> None:
    entry_point = Path(sys.argv[0]).stem or "generate"

    parser = argparse.ArgumentParser(description="Generate an AMP Challenge submission.")
    parser.add_argument("--n-sequences", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--length", type=int, default=MAX_PEPTIDE_LENGTH,
        help="maximum peptide length, in residues (template flag; the "
             "competition ceiling and the default are both 50)")
    parser.add_argument("--device", default=None,
                        help="'cuda', 'cpu', or omitted to resolve automatically.")
    parser.add_argument("--kappa", type=float, default=1.5,
                        help="Risk aversion in top-K selection. Inert while the "
                             "oracle is a single model (score_std is 0).")
    parser.add_argument("--overflow-k", type=int, default=500,
                        help="Length of the ranked overflow list written to "
                             "docs/. The organizers replace an invalid "
                             "top-100 entry with the next valid candidate, so "
                             "the ordering past 100 is used.")
    parser.add_argument("--identity-ceiling", type=float, default=SAFETY_CEILING,
                        help="Max Levenshtein ratio to any reference sequence. "
                             "The validator fails above 0.80; default leaves margin.")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: named after the entry point).")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device  : {device}")

    # A clean clone has the checksums but not the two large artifacts, so this
    # has to happen before anything tries to load them. It is a no-op when
    # they are already present and verified.
    ensure_weights()

    # Load first, seed second. See set_global_determinism.
    model, species_vocab, oracle, novelty, references, calibration = load_models(device)
    print(f"oracle  : {oracle.n_trees} trees, esm={oracle.meta.get('esm_model')}")

    set_global_determinism(args.seed)

    out_dir = Path(args.out) if args.out else Path(entry_point)
    out_dir.mkdir(parents=True, exist_ok=True)

    library = generate_library(
        args.n_sequences, args.seed, model, species_vocab, novelty,
        references, calibration, device, max_length=args.length,
    )
    library_path = out_dir / "library.fasta"
    write_fasta(library, library_path)
    print(f"library : {len(library)} sequences -> {library_path}")

    report = verify_library(library, reference=set(references),
                            expected_size=args.n_sequences)
    print(report)
    if not report.ok:
        sys.exit(1)

    candidates, metrics = score_candidates(library, oracle)

    # Computed once for the whole library rather than per candidate inside the
    # greedy loop, where it would dominate the cost.
    reference_identity = novelty.max_identity(library)

    ranked, info = select_top_diverse(
        candidates,
        novelty=novelty,
        reference_identity=reference_identity,
        k=args.overflow_k,
        min_required=args.top_k,
        kappa=args.kappa,
        reference_ceiling=args.identity_ceiling,
    )
    top = ranked[:args.top_k]
    top_sequences = [c.sequence for c in top]

    top_path = out_dir / "top.fasta"
    write_fasta(top_sequences, top_path)
    print(f"top     : {len(top_sequences)} sequences -> {top_path}")

    # The organizers replace an invalid top-100 entry with "the next valid
    # candidate", so the ordering past rank 100 is used. Writing it out keeps
    # that replacement inside our procedure. Deliberately not in the output
    # directory, which the template expects to hold exactly library.fasta and
    # top.fasta.
    overflow_path = _REPO_ROOT / "docs" / "top500_overflow.fasta"
    write_fasta([c.sequence for c in ranked], overflow_path, prefix="rank")
    print(f"overflow: {len(ranked)} ranked sequences -> {overflow_path}")

    _report_selection(top, ranked, metrics, info, reference_identity, library)

    top_report = verify_top(top_sequences, library, references=references,
                            top_k=args.top_k)
    print(top_report)
    if not top_report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
