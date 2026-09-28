"""
Collapse the harmonized activity table into one MIC row per
(sequence, species), the form the regressor trains on.

Also emits the unique-sequence FASTA the ESM-2 embedding job consumes, in a
fixed order, so the .npy rows and this table stay aligned.

Usage:
    python scripts/build_mic_table.py \
        --activity $D/processed/conditioned/activity_harmonized.csv \
        --output $D/processed/refinement/mic_table.csv \
        --sequences-out $D/processed/refinement/mic_sequences.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ampx.models.features import clean_sequence, compute_sequence_features, gram_for_species, is_valid_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequences-out", required=True)
    args = parser.parse_args()

    activity = pd.read_csv(args.activity)
    mic = activity[(activity["endpoint"] == "mic") & activity["value_um"].gt(0)].copy()
    mic = mic.dropna(subset=["sequence", "species", "value_um"])
    mic["sequence"] = mic["sequence"].astype(str).map(clean_sequence)
    mic = mic[mic["sequence"].map(is_valid_sequence)]
    print(f"MIC rows after cleaning: {len(mic)}")

    mic["log_mic"] = np.log10(mic["value_um"].astype(float))
    collapsed = (
        mic.groupby(["sequence", "species"], as_index=False)
        .agg(log_mic=("log_mic", "mean"), n_measurements=("log_mic", "size"))
    )
    collapsed["gram"] = collapsed["species"].map(gram_for_species)
    print(f"collapsed to {len(collapsed)} (sequence, species) rows "
          f"over {collapsed['sequence'].nunique()} unique sequences")

    # Descriptors are a pure function of the sequence, so compute them once
    # per unique sequence and broadcast rather than per row.
    unique = pd.DataFrame({"sequence": sorted(collapsed["sequence"].unique())})
    features = compute_sequence_features(unique["sequence"])
    for column in features.columns:
        unique[column] = features[column].values
    unique["row"] = np.arange(len(unique))

    merged = collapsed.merge(unique, on="sequence", how="left")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    unique.to_csv(args.sequences_out, index=False)

    print(f"log_mic: min {merged['log_mic'].min():.3f} median "
          f"{merged['log_mic'].median():.3f} max {merged['log_mic'].max():.3f}")
    print(f"species: {merged['species'].nunique()} distinct")
    print(f"wrote {args.output}")
    print(f"wrote {args.sequences_out} ({len(unique)} unique sequences, "
          f"'row' is the ESM embedding index)")


if __name__ == "__main__":
    main()
