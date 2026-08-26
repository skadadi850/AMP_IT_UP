"""Cluster-disjoint train/val/test splits via MMseqs2.

## Why a random split will lie to you

AMP databases are saturated with homologs. MarLys aggregates thirteen primary
databases, so the same peptide family appears many times with one or two
substitutions -- magainin variants, cecropin variants, whole alanine-scan
series. A random split puts near-identical sequences on both sides of the
boundary. Your classifier then "generalises" by recognising sequences it has
effectively already seen, and every number you report is inflated.

This is not a minor effect and it is the single most common methodological flaw
in published AMP prediction work. It matters doubly here because the split
choice does not merely misreport your accuracy -- it *selects your top-100*. A
predictor tuned on a leaky split picks the wrong 100 peptides, and Phase 2
scores you on the mean of a random 25 of them.

So: cluster at 40% identity, assign whole clusters to splits, never individual
sequences. Report the clustered number. If it is much worse than your random
split, the gap is the leakage you would otherwise have shipped.

## Why the organizers' own threshold is different, and why that is fine

Phase 1 novelty is scored against MarLys at 80% identity, and the top-100 has
an 80% ceiling. Those are *submission constraints*. 40% is a much stricter bar
used here for a different purpose: honest generalisation measurement. You want
your evaluation split to be harder than your deployment condition, not equal to
it.

## Install MMseqs2

    conda install -c conda-forge -c bioconda mmseqs2
    # or:  brew install mmseqs2
    # or see https://github.com/soedinglab/MMseqs2

MMseqs2 is also what the organizers use for the real Phase 1 identity check, so
you need it installed regardless. Get it now.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"
INTERIM = REPO / "data" / "interim"


def require_mmseqs() -> str:
    exe = shutil.which("mmseqs")
    if exe is None:
        print(
            "mmseqs not found on PATH.\n\n"
            "  conda install -c conda-forge -c bioconda mmseqs2\n\n"
            "You need it for the Phase 1 identity check too, so install it now\n"
            "rather than working around it.",
            file=sys.stderr,
        )
        sys.exit(1)
    return exe


def load_sequences(path: Path) -> list[str]:
    with open(path) as fh:
        return sorted({row["sequence"] for row in csv.DictReader(fh)})


def cluster(sequences: list[str], identity: float, coverage: float,
            exe: str, workdir: Path) -> dict[str, int]:
    """Cluster sequences, returning sequence -> cluster id.

    Uses --cov-mode 0 with a coverage requirement so that a short peptide
    aligning inside a long one does not merge two unrelated families. For
    peptides this matters: without coverage control, an 8-mer can chain
    otherwise-distinct clusters together.
    """
    fasta = workdir / "in.fasta"
    with open(fasta, "w") as fh:
        for i, s in enumerate(sequences):
            fh.write(f">s{i}\n{s}\n")

    db = workdir / "db"
    clu = workdir / "clu"
    tsv = workdir / "clu.tsv"

    def run(cmd: list[str]) -> None:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)

    run([exe, "createdb", str(fasta), str(db)])
    run([exe, "cluster", str(db), str(clu), str(workdir / "tmp"),
         "--min-seq-id", str(identity),
         "-c", str(coverage),
         "--cov-mode", "0",
         # Short sequences need a sensitive search or you get singletons.
         "-s", "7.5",
         "--cluster-mode", "0"])
    run([exe, "createtsv", str(db), str(db), str(clu), str(tsv)])

    idx_to_seq = {f"s{i}": s for i, s in enumerate(sequences)}
    rep_to_id: dict[str, int] = {}
    assignment: dict[str, int] = {}
    with open(tsv) as fh:
        for line in fh:
            rep, member = line.split("\t")[:2]
            member = member.strip()
            if rep not in rep_to_id:
                rep_to_id[rep] = len(rep_to_id)
            seq = idx_to_seq.get(member)
            if seq is not None:
                assignment[seq] = rep_to_id[rep]
    return assignment


def assign_splits(clusters: dict[str, int], fracs: tuple[float, float, float],
                  seed: int) -> dict[str, str]:
    """Assign whole clusters to splits, largest-first for balance.

    Largest-first matters: a few AMP families are enormous, and random cluster
    assignment can put a 2000-member family entirely in test, wrecking the
    intended proportions.
    """
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for seq, cid in clusters.items():
        by_cluster[cid].append(seq)

    order = sorted(by_cluster.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rng = random.Random(seed)

    total = sum(len(v) for v in by_cluster.values())
    targets = {"train": fracs[0] * total, "val": fracs[1] * total,
               "test": fracs[2] * total}
    current = {"train": 0, "val": 0, "test": 0}

    out: dict[str, str] = {}
    for _, members in order:
        # Greedily place in whichever split is furthest below its target.
        deficits = {k: targets[k] - current[k] for k in targets}
        pick = max(deficits, key=lambda k: (deficits[k], rng.random()))
        for m in members:
            out[m] = pick
        current[pick] += len(members)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", type=Path, default=PROCESSED / "peptides.csv")
    ap.add_argument("--out", type=Path, default=PROCESSED / "splits.csv")
    ap.add_argument("--identity", type=float, default=0.40,
                    help="Clustering identity threshold (default 0.40).")
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--fracs", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"{args.input} not found. Run scripts/build_dataset.py first.")
        sys.exit(1)

    exe = require_mmseqs()
    sequences = load_sequences(args.input)
    print(f"sequences        : {len(sequences):,}")
    print(f"clustering at    : {args.identity:.0%} identity, "
          f"{args.coverage:.0%} coverage")

    INTERIM.mkdir(parents=True, exist_ok=True)
    ctx = (tempfile.TemporaryDirectory(dir=INTERIM) if not args.keep_tmp
           else None)
    workdir = Path(ctx.name) if ctx else INTERIM / "mmseqs_work"
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        clusters = cluster(sequences, args.identity, args.coverage, exe, workdir)
    except subprocess.CalledProcessError as e:
        print(f"mmseqs failed:\n{e.stderr.decode(errors='replace')}",
              file=sys.stderr)
        sys.exit(1)
    finally:
        if ctx:
            ctx.cleanup()

    n_clusters = len(set(clusters.values()))
    sizes = Counter(clusters.values())
    biggest = sizes.most_common(1)[0][1] if sizes else 0
    singletons = sum(1 for v in sizes.values() if v == 1)

    print(f"clusters         : {n_clusters:,}")
    print(f"  singletons     : {singletons:,} "
          f"({100*singletons/max(n_clusters,1):.1f}%)")
    print(f"  largest        : {biggest:,} sequences")
    print(f"redundancy       : {len(sequences)/max(n_clusters,1):.2f}x "
          f"(sequences per cluster)")

    splits = assign_splits(clusters, tuple(args.fracs), args.seed)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "cluster", "split"])
        for seq in sequences:
            if seq in splits:
                w.writerow([seq, clusters[seq], splits[seq]])

    dist = Counter(splits.values())
    print("\nsplit sizes:")
    for k in ("train", "val", "test"):
        print(f"  {k:<6} {dist[k]:>8,} ({100*dist[k]/max(len(splits),1):5.1f}%)")
    print(f"\n  -> {args.out}")
    print("\nJoin this on `sequence` when training. Report metrics on the test "
          "split only.\nIf a random split scores much better, that gap is leakage.")


if __name__ == "__main__":
    main()
