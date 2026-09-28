"""
Step 2 -- clean each raw library to competition scope and record attrition.

Filtering, not repairing. A sequence containing a non-standard residue is
DROPPED, not stripped down to its standard residues, and a sequence outside
8-50 is dropped rather than trimmed. Repairing would hide exactly what this
step exists to expose: if a model emits mostly out-of-scope output, that is a
result about the model.

Nothing is padded back up to 10,000 after filtering.

Raw generator output is preserved under libraries_raw/ so the counts in
attrition.csv can be re-derived from the files they came from.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from ampx.models.compliance import write_fasta
from ampx.models.features import MAX_PEPTIDE_LENGTH, MIN_PEPTIDE_LENGTH, VALID_AA


def read_fasta_raw(path):
    """
    Read a FASTA without touching the residues.

    ampx.models.compliance.read_fasta applies clean_sequence(), which silently STRIPS
    characters outside the twenty standard residues -- 'ACDXEFG' comes back as
    'ACDEFG'. Using it here would make after_alphabet identical to raw by
    construction and hide exactly the out-of-scope output Step 2 exists to
    measure, so attrition is counted from the bytes on disk instead.

    One record per '>' header, so a generator that emitted an empty string
    still counts as one draw rather than vanishing.
    """
    records, current, seen_header = [], [], False
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if seen_header:
                    records.append("".join(current))
                current, seen_header = [], True
            elif line:
                current.append(line)
    if seen_header:
        records.append("".join(current))
    return records

MODELS = ["gramp_md", "gramp_face", "gramp_latent", "gramp_latent_ft",
          "omegamp", "ampdiffusion"]
SEEDS = [0, 1, 2]


def clean(sequences):
    """Returns (after_alphabet, after_length, deduped list)."""
    upper = [str(s).upper().strip() for s in sequences]
    alphabet = [s for s in upper if s and all(c in VALID_AA for c in s)]
    length = [s for s in alphabet
              if MIN_PEPTIDE_LENGTH <= len(s) <= MAX_PEPTIDE_LENGTH]
    deduped = list(dict.fromkeys(length))
    return len(alphabet), len(length), deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libraries", default="results/comparison/libraries")
    parser.add_argument("--raw-dir", default="results/comparison/libraries_raw")
    parser.add_argument("--output", default="results/comparison/attrition.csv")
    args = parser.parse_args()

    lib = Path(args.libraries)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODELS:
        for seed in SEEDS:
            name = f"{model}_seed{seed}.fasta"
            raw_path = raw_dir / name
            live_path = lib / name

            # First run: the generator wrote raw output into libraries/.
            # Move it aside once, so re-running this script is idempotent.
            if not raw_path.exists():
                if not live_path.exists():
                    print(f"MISSING {name} -- skipped")
                    continue
                shutil.move(str(live_path), str(raw_path))

            sequences = read_fasta_raw(str(raw_path))
            n_alpha, n_len, kept = clean(sequences)
            write_fasta(str(live_path), kept)

            rows.append({
                "model": model, "seed": seed,
                "raw": len(sequences),
                "after_alphabet": n_alpha,
                "after_length": n_len,
                "after_dedup": len(kept),
            })
            print(f"{model:18s} seed {seed}  raw {len(sequences):6d}  "
                  f"alphabet {n_alpha:6d}  length {n_len:6d}  dedup {len(kept):6d}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["model", "seed", "raw", "after_alphabet",
                            "after_length", "after_dedup"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
