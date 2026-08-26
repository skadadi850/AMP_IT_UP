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
    rather than sorted back into a canonical order.

The safest architecture, and the one used here: generate to a list, sort
deterministically, then write. Never let concurrency touch output ordering.

## Current state

`_placeholder_generator` is an order-2 Markov model fit on the reference AMP
set. It exists so the repository passes the official validator from day one --
swap in your trained model at the marked call site. It is not a competitive
method and should not be submitted.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .compliance import (
    SAFETY_CEILING,
    LIBRARY_SIZE,
    TOP_SIZE,
    is_valid_sequence,
    read_fasta,
    verify_library,
    verify_top,
    write_fasta,
)
from .ranking import Candidate, select_top

#: Every stochastic component must derive from this one value.
DEFAULT_SEED = 42

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
REFERENCE_FASTA = _REPO_ROOT / "data" / "reference" / "antibacterial.fasta"


def set_global_determinism(seed: int) -> None:
    """Pin every RNG we might touch. Call this first, before anything else."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional until you plug in a trained model
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Placeholder generator -- REPLACE THIS
# --------------------------------------------------------------------------- #

def _fit_markov(sequences: list[str], order: int = 2) -> dict:
    """Fit an order-k Markov model over residues, plus a length distribution."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lengths: list[int] = []

    for seq in sequences:
        lengths.append(len(seq))
        padded = "^" * order + seq
        for i in range(order, len(padded)):
            counts[padded[i - order : i]][padded[i]] += 1

    model = {}
    for ctx, nxt in counts.items():
        residues = sorted(nxt)  # sorted -> deterministic ordering
        weights = np.array([nxt[r] for r in residues], dtype=float)
        model[ctx] = (residues, weights / weights.sum())

    return {"order": order, "table": model, "lengths": sorted(lengths)}


def _placeholder_generator(
    n: int, model: dict, rng: np.random.Generator, forbidden: set[str]
) -> list[str]:
    """Sample `n` unique valid sequences from the fitted Markov model."""
    order = model["order"]
    table = model["table"]
    lengths = model["lengths"]

    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = n * 200

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        target = int(rng.choice(lengths))
        seq = ""
        ctx = "^" * order
        for _ in range(target):
            entry = table.get(ctx)
            if entry is None:
                break
            residues, probs = entry
            seq += str(rng.choice(residues, p=probs))
            ctx = (ctx + seq[-1])[-order:]

        if not is_valid_sequence(seq) or seq in seen or seq in forbidden:
            continue
        seen.add(seq)
        out.append(seq)

    if len(out) < n:
        raise RuntimeError(
            f"generated only {len(out)}/{n} unique sequences after "
            f"{attempts} attempts"
        )
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def generate_library(n_sequences: int, seed: int) -> list[str]:
    """Produce the compliant library. Swap the marked call for your model."""
    rng = np.random.default_rng(seed)

    _, references = read_fasta(REFERENCE_FASTA)
    forbidden = set(references)

    # >>> REPLACE: load your checkpoint and sample from it here. <<<
    model = _fit_markov(references, order=2)
    sequences = _placeholder_generator(n_sequences, model, rng, forbidden)

    # Canonical ordering makes byte-identical output independent of how the
    # sequences were produced. Do not remove.
    return sorted(sequences)


def score_candidates(sequences: list[str]) -> list[Candidate]:
    """Attach surrogate predictions to sequences.

    Replace with your calibrated ensemble. Return `mean_score` (higher is
    better), `score_std` (ensemble disagreement) and a `cluster` id so
    `select_top` can enforce structural spread. See docs/METHOD.md.
    """
    # Placeholder: net charge as a crude activity proxy, so the pipeline runs.
    cationic = set("KR")
    anionic = set("DE")
    out = []
    for i, seq in enumerate(sequences):
        charge = sum(c in cationic for c in seq) - sum(c in anionic for c in seq)
        out.append(
            Candidate(
                sequence=seq,
                mean_score=charge / max(len(seq), 1),
                score_std=0.0,
                cluster=i % 25,
            )
        )
    return out


def main() -> None:
    entry_point = Path(sys.argv[0]).stem or "generate"

    parser = argparse.ArgumentParser(description="Generate an AMP Challenge submission.")
    parser.add_argument("--n-sequences", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--kappa", type=float, default=1.5,
                        help="Risk aversion in top-K selection.")
    parser.add_argument("--max-per-cluster", type=int, default=4)
    parser.add_argument("--identity-ceiling", type=float, default=SAFETY_CEILING,
                        help="Max Levenshtein ratio to any reference sequence. "
                             "The validator fails above 0.80; default leaves margin.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the similarity scan (faster; not for submission).")
    args = parser.parse_args()

    set_global_determinism(args.seed)

    out_dir = Path(entry_point)
    out_dir.mkdir(parents=True, exist_ok=True)

    library = generate_library(args.n_sequences, args.seed)
    library_path = out_dir / "library.fasta"
    write_fasta(library, library_path)
    print(f"library : {len(library)} sequences -> {library_path}")

    _, references = read_fasta(REFERENCE_FASTA)
    report = verify_library(library, reference=set(references),
                            expected_size=args.n_sequences)
    print(report)
    if not report.ok:
        sys.exit(1)

    candidates = score_candidates(library)
    top = select_top(
        candidates,
        references=None if args.skip_verify else references,
        k=args.top_k,
        kappa=args.kappa,
        max_per_cluster=args.max_per_cluster,
        identity_ceiling=args.identity_ceiling,
    )
    top_sequences = [c.sequence for c in top]

    top_path = out_dir / "top.fasta"
    write_fasta(top_sequences, top_path)
    print(f"top     : {len(top_sequences)} sequences -> {top_path}")

    top_report = verify_top(
        top_sequences,
        library,
        references=None if args.skip_verify else references,
        top_k=args.top_k,
    )
    print(top_report)
    if not top_report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
