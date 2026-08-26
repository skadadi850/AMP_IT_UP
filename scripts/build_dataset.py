"""Build the canonical peptide table from whatever is in data/raw/.

Typical workflow:

    # 1. See what a download actually looks like before configuring it
    uv run python scripts/build_dataset.py --inspect dbaasp

    # 2. Fix seq_col / mic_col in src/ampx/data/sources.py to match

    # 3. Build
    uv run python scripts/build_dataset.py

    # 4. Read the report. If a source shows kept=0 or a suspiciously low keep
    #    rate, the column names are wrong -- go back to step 1.

Outputs:

    data/processed/peptides.csv     canonical long-format table
    data/processed/manifest.md      provenance, paste into data/README.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ampx.data.ingest import (  # noqa: E402
    IngestStats,
    Record,
    canonical_sequence,
    parse_mic,
    read_fasta,
    read_table,
)
from ampx.data.sources import SOURCES, SOURCES_BY_ID, Source  # noqa: E402

RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"

FIELDS = [
    "sequence", "length", "source", "role", "label",
    "mic_ugml", "mic_censored", "strain", "src_id",
]


def _files_for(src: Source) -> list[Path]:
    folder = RAW / src.id
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob(src.glob) if p.is_file())


def inspect(source_id: str) -> None:
    """Print the real structure of a raw drop. Always do this first."""
    src = SOURCES_BY_ID.get(source_id)
    if src is None:
        print(f"Unknown source '{source_id}'. Known: {sorted(SOURCES_BY_ID)}")
        sys.exit(1)

    files = _files_for(src)
    print(f"\nSource   : {src.id}  ({src.name})")
    print(f"Folder   : {RAW / src.id}")
    print(f"Declared : kind={src.kind} role={src.role}")
    if not files:
        print("\nNo files found. Download the data into the folder above.")
        return
    print(f"Files    : {len(files)}")
    for p in files[:10]:
        print(f"           {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    head = files[0]
    print(f"\n--- first records of {head.name} ---")
    if src.kind == "fasta":
        for i, (hdr, seq) in enumerate(read_fasta(head)):
            print(f">{hdr}\n{seq[:80]}")
            if i >= 2:
                break
    else:
        rows = read_table(head)
        try:
            first = next(rows)
        except StopIteration:
            print("(empty)")
            return
        print("COLUMNS:")
        for k in first:
            print(f"  {k!r}")
        print("\nSAMPLE ROWS:")
        sample = [first]
        for _, r in zip(range(2), rows):
            sample.append(r)
        for i, row in enumerate(sample):
            print(f"  [{i}] " + "  ".join(f"{k}={v!r}" for k, v in list(row.items())[:6]))
        print("\nSet seq_col / mic_col / label_col in sources.py to match the "
              "column names above.")


def ingest_source(src: Source) -> tuple[list[Record], IngestStats]:
    stats = IngestStats(source=src.id)
    records: list[Record] = []

    for path in _files_for(src):
        stats.files.append(path.name)

        if src.kind == "fasta":
            for hdr, raw in read_fasta(path):
                stats.rows_read += 1
                seq = canonical_sequence(raw)
                if seq is None:
                    if raw.strip():
                        stats.rejected_nonstandard += 1
                    else:
                        stats.rejected_empty += 1
                    continue
                records.append(Record(
                    sequence=seq, source=src.id, role=src.role,
                    label=1.0 if src.role == "positive"
                          else 0.0 if src.role == "negative" else None,
                    src_id=hdr.split()[0] if hdr else "",
                ))
                stats.kept += 1
                if src.role in ("positive", "negative"):
                    stats.with_label += 1

        else:
            if not src.seq_col:
                print(f"  ! {src.id}: kind='table' but seq_col is not set. "
                      f"Run --inspect {src.id}")
                continue
            for row in read_table(path):
                stats.rows_read += 1
                seq = canonical_sequence(row.get(src.seq_col, ""))
                if seq is None:
                    stats.rejected_nonstandard += 1
                    continue

                label = None
                if src.label_col:
                    v = str(row.get(src.label_col, "")).strip().lower()
                    if v:
                        label = 1.0 if v in src.positive_values else 0.0
                elif src.role == "positive":
                    label = 1.0
                elif src.role == "negative":
                    label = 0.0

                mic, censored = (None, False)
                if src.mic_col:
                    mic, censored = parse_mic(row.get(src.mic_col))

                records.append(Record(
                    sequence=seq, source=src.id, role=src.role, label=label,
                    mic_ugml=mic, mic_censored=censored,
                    strain=str(row.get(src.strain_col, "")) if src.strain_col else "",
                    src_id=str(row.get(src.id_col, "")) if src.id_col else "",
                ))
                stats.kept += 1
                if label is not None:
                    stats.with_label += 1
                if mic is not None:
                    stats.with_mic += 1
                    if censored:
                        stats.censored_mic += 1

    return records, stats


def write_manifest(all_stats: list[IngestStats], out: Path) -> None:
    """Emit the provenance table required for disclosure."""
    lines = [
        "# Training data manifest",
        "",
        "Auto-generated by `scripts/build_dataset.py`. Copy into "
        "`data/README.md` and fill any TODO license/version fields.",
        "",
        "| Source | Name | License | Accessed | Role | Files | Kept | Labels | MIC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for st in all_stats:
        src = SOURCES_BY_ID[st.source]
        lines.append(
            f"| `{src.id}` | {src.name} | {src.license} | {src.accessed} | "
            f"{src.role} | {len(st.files)} | {st.kept:,} | {st.with_label:,} | "
            f"{st.with_mic:,} |"
        )
    lines += ["", "## Notes", ""]
    for st in all_stats:
        src = SOURCES_BY_ID[st.source]
        if src.notes:
            lines.append(f"- **{src.id}**: {src.notes}")
    out.write_text("\n".join(lines) + "\n")


def build() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    all_records: list[Record] = []
    all_stats: list[IngestStats] = []

    print("\nIngesting\n" + "-" * 78)
    for src in SOURCES:
        if not _files_for(src):
            print(f"{src.id:<16} (no files in data/raw/{src.id}/ -- skipped)")
            continue
        recs, stats = ingest_source(src)
        all_records.extend(recs)
        all_stats.append(stats)
        print(stats.summary())

    if not all_records:
        print("\nNothing ingested. Download data into data/raw/<source_id>/ "
              "and check sources.py.")
        return

    # Deduplicate on (sequence, source) so the same peptide appearing in two
    # databases is kept twice -- agreement across independent sources is signal,
    # and collapsing it early destroys the ability to weight by it later.
    seen: set[tuple[str, str]] = set()
    unique: list[Record] = []
    for r in all_records:
        key = (r.sequence, r.source)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    out_csv = PROCESSED / "peptides.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in unique:
            w.writerow({
                "sequence": r.sequence, "length": r.length, "source": r.source,
                "role": r.role,
                "label": "" if r.label is None else r.label,
                "mic_ugml": "" if r.mic_ugml is None else r.mic_ugml,
                "mic_censored": int(r.mic_censored),
                "strain": r.strain, "src_id": r.src_id,
            })

    write_manifest(all_stats, PROCESSED / "manifest.md")

    seqs = {r.sequence for r in unique}
    roles = Counter(r.role for r in unique)
    in_range = sum(1 for s in seqs if 8 <= len(s) <= 50)
    digest = hashlib.sha256("".join(sorted(seqs)).encode()).hexdigest()[:16]

    print("-" * 78)
    print(f"rows written      : {len(unique):,}")
    print(f"unique sequences  : {len(seqs):,}")
    print(f"  within 8-50 aa  : {in_range:,} ({100*in_range/len(seqs):.1f}%)")
    print(f"by role           : {dict(roles)}")
    print(f"corpus fingerprint: {digest}")
    print(f"\n  -> {out_csv}")
    print(f"  -> {PROCESSED / 'manifest.md'}")
    print("\nNext: uv run python scripts/cluster_split.py")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--inspect", metavar="SOURCE_ID",
                    help="Print the real structure of a raw drop and exit.")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
    else:
        build()


if __name__ == "__main__":
    main()
