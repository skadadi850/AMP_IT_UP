"""Generate synthetic negatives for classifier training.

## Why this script matters more than your architecture choice

From OmegAMP's ablation (their Tab. 16), positive likelihood ratio LR+ as
synthetic negative types are added:

    no synthetics                      LR+   3.5
    + random                           LR+   4.3
    + shuffled                         LR+  27.5
    + mutated                          LR+ 138.1

Mutated negatives alone account for roughly 80% of the total gain. Every
published deep-learning AMP classifier they compared against sits at LR+ 1-4 --
barely better than chance at the thing that actually matters, which is not
calling an inactive peptide active. The difference is not the model. It is the
negatives.

## What each type teaches, and why you need all of them

**Random** (uniform over 20 residues, length-matched to the positives).
Length-matching is the point: without it the classifier learns "AMPs are 8-50
residues" and little else.

**Shuffled** (a permutation of a real AMP). Preserves net charge, hydrophobicity
and composition *exactly*; destroys only order. This is the pressure that stops
the model being a pure charge detector -- recall OmegAMP's feature importance
put mean charge at pH 7 in 30% of decision nodes. Baselines misclassify 27-99%
of shuffled sequences as AMPs.

**Mutated** (5 substitutions in a real AMP). Preserves context almost entirely,
so it teaches that AMP-ness is not a bag of local motifs. Highest value.

**Add/delete** (5 insertions or deletions). *Held out of training*, used only
for evaluation. These mimic near-miss outputs of a generative model -- exactly
the distribution your own generator produces. Baselines misclassify 31-99%.
Scoring well here without training on it is the real test that you learned
function rather than composition.

## The assumption, stated plainly

These are *assumed* inactive, not measured inactive. The argument is that
P(active | shuffled) <= 1/L! and actives are a vanishing fraction of sequence
space; empirically under 5% of random and shuffled sequences show activity. The
label noise is small but nonzero -- which is exactly why the weighted loss
upweights experimentally-validated rows rather than treating all rows equally.
Skip the weighting and your classifier will be overconfident.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ampx.data.ingest import STANDARD  # noqa: E402

PROCESSED = REPO / "data" / "processed"
ALPHABET = sorted(STANDARD)

DEFAULT_N = 100_000
N_MUTATIONS = 5
N_INDELS = 5


def load_positives(path: Path, min_len: int = 8, max_len: int = 50) -> list[str]:
    """Read experimentally-supported active sequences from the canonical table."""
    seqs: list[str] = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["role"] != "positive":
                continue
            s = row["sequence"]
            if min_len <= len(s) <= max_len:
                seqs.append(s)
    return sorted(set(seqs))


def make_random(n: int, lengths: list[int], rng: random.Random) -> list[str]:
    """Uniform random sequences, length-matched to the positive distribution."""
    return ["".join(rng.choices(ALPHABET, k=rng.choice(lengths))) for _ in range(n)]


def make_shuffled(n: int, positives: list[str], rng: random.Random) -> list[str]:
    """Permutations of real AMPs. Composition preserved, order destroyed."""
    out = []
    for _ in range(n):
        chars = list(rng.choice(positives))
        rng.shuffle(chars)
        out.append("".join(chars))
    return out


def make_mutated(n: int, positives: list[str], rng: random.Random,
                 k: int = N_MUTATIONS) -> list[str]:
    """k substitutions at distinct positions in a real AMP."""
    out = []
    for _ in range(n):
        seq = list(rng.choice(positives))
        for p in rng.sample(range(len(seq)), min(k, len(seq))):
            seq[p] = rng.choice([a for a in ALPHABET if a != seq[p]])
        out.append("".join(seq))
    return out


def make_add_delete(n: int, positives: list[str], rng: random.Random,
                    k: int = N_INDELS, min_len: int = 8, max_len: int = 50) -> list[str]:
    """k insertions/deletions. EVALUATION ONLY -- never train on these."""
    out = []
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        seq = list(rng.choice(positives))
        for _ in range(k):
            if rng.random() < 0.5 and len(seq) > min_len:
                del seq[rng.randrange(len(seq))]
            elif len(seq) < max_len:
                seq.insert(rng.randrange(len(seq) + 1), rng.choice(ALPHABET))
        if min_len <= len(seq) <= max_len:
            out.append("".join(seq))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-per-type", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input", type=Path, default=PROCESSED / "peptides.csv")
    ap.add_argument("--out", type=Path, default=PROCESSED / "negatives.csv")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"{args.input} not found. Run scripts/build_dataset.py first.")
        sys.exit(1)

    rng = random.Random(args.seed)
    positives = load_positives(args.input)
    if len(positives) < 100:
        print(f"WARNING: only {len(positives)} positives -- shuffled and mutated "
              f"negatives will not be diverse. Add more sources.")
    print(f"positives available : {len(positives):,}")

    lengths = [len(s) for s in positives]
    n = args.n_per_type

    groups = {
        "random": make_random(n, lengths, rng),
        "shuffled": make_shuffled(n, positives, rng),
        "mutated": make_mutated(n, positives, rng),
        "add_delete": make_add_delete(n, positives, rng),
    }

    # A synthetic negative equal to a real AMP is a mislabelled row. Rare, free
    # to remove, and exactly the sort of error that quietly caps your ceiling.
    positive_set = set(positives)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    collisions = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "length", "negative_type", "split_role"])
        for kind, seqs in groups.items():
            split_role = "eval_only" if kind == "add_delete" else "train"
            for s in dict.fromkeys(seqs):
                if s in positive_set:
                    collisions += 1
                    continue
                w.writerow([s, len(s), kind, split_role])
                counts[kind] += 1

    print("\nwritten:")
    for kind, c in counts.items():
        role = "eval only" if kind == "add_delete" else "train"
        print(f"  {kind:<12} {c:>8,}  ({role})")
    print(f"  collisions with real AMPs removed: {collisions}")
    print(f"\n  -> {args.out}")
    print("\nReminder: weight experimentally-validated rows above these when "
          "training. See docs/PLAN.md 2.1.")


if __name__ == "__main__":
    main()
