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
from .models.features import MAX_PEPTIDE_LENGTH, MIN_PEPTIDE_LENGTH
from .models.masked_diffusion import MaskedDiffusionModel
from .models.novelty_fast import ExhaustiveNovelty
from .models.sampling import empirical_length_counts, length_grid, sample_library
from .predictor import Oracle, resolve_device
from .ranking import Candidate, select_top
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
) -> list[str]:
    """Sample until `n_sequences` unique, compliant, novel sequences exist."""
    request = dict(request or {})
    reference_exact = set(references)
    reference_lengths = [len(s) for s in references]

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

        counts = empirical_length_counts(reference_lengths, ask)
        condition = length_grid(request, counts, species_vocab, device=device)

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
            free_length=True,
            reveal=reveal,
            pad_bias=pad_bias,
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

    # Tie-break by potency, scaled far below one strain's worth of success
    # rate so it can never outrank a genuine breadth difference.
    tiebreak = -metrics["mean_log_mic"] * 1e-3
    score = overall + tiebreak - np.where(eligible, 0.0, 10.0)

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

    return [
        Candidate(
            sequence=seq,
            mean_score=float(value),
            score_std=0.0,
            cluster=_structural_cluster(seq),
        )
        for seq, value in zip(sequences, score)
    ]


def main() -> None:
    entry_point = Path(sys.argv[0]).stem or "generate"

    parser = argparse.ArgumentParser(description="Generate an AMP Challenge submission.")
    parser.add_argument("--n-sequences", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None,
                        help="'cuda', 'cpu', or omitted to resolve automatically.")
    parser.add_argument("--kappa", type=float, default=1.5,
                        help="Risk aversion in top-K selection. Inert while the "
                             "oracle is a single model (score_std is 0).")
    parser.add_argument("--max-per-cluster", type=int, default=4)
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
        references, calibration, device,
    )
    library_path = out_dir / "library.fasta"
    write_fasta(library, library_path)
    print(f"library : {len(library)} sequences -> {library_path}")

    report = verify_library(library, reference=set(references),
                            expected_size=args.n_sequences)
    print(report)
    if not report.ok:
        sys.exit(1)

    candidates = score_candidates(library, oracle)
    top = select_top(
        candidates,
        novelty=novelty,
        k=args.top_k,
        kappa=args.kappa,
        max_per_cluster=args.max_per_cluster,
        identity_ceiling=args.identity_ceiling,
    )
    top_sequences = [c.sequence for c in top]

    top_path = out_dir / "top.fasta"
    write_fasta(top_sequences, top_path)
    print(f"top     : {len(top_sequences)} sequences -> {top_path}")

    top_report = verify_top(top_sequences, library, references=references,
                            top_k=args.top_k)
    print(top_report)
    if not top_report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
