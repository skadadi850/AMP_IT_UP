"""
Build the training metadata table.

Adding a new MIC or HC50 source is a JSON entry, not a code change. The
source spec is a list of objects:

[
  {
    "path": "data/raw/dbaasp_activity.csv",
    "endpoint": "mic",
    "source": "dbaasp",
    "default_unit": "ug/ml",
    "columns": {
      "sequence": "SEQUENCE",
      "value": "CONCENTRATION",
      "unit": "UNIT",
      "species": "TARGET_SPECIES",
      "strain": "TARGET_STRAIN",
      "relation": "CONCENTRATION_RELATION"
    }
  },
  {
    "path": "data/raw/dbaasp_hemolysis.csv",
    "endpoint": "hc50",
    "source": "dbaasp",
    "default_unit": "ug/ml",
    "columns": {"sequence": "SEQUENCE", "value": "CONCENTRATION", "unit": "UNIT"}
  }
]

Only "sequence" and "value" are required in "columns". Unit handling,
comparison operators, ranges and species normalization are all in
ampgen.ingest, so a new database only needs its column names.

Usage:
    python scripts/build_dataset.py \
        --positives data/raw/mlamp_positives.csv \
        --sources data/raw/activity_sources.json \
        --output data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ampx.models.compliance import read_fasta
from ampx.models.features import (
    attach_conditioning_features,
    build_species_vocabulary,
    clean_sequence,
    is_valid_sequence,
)
from ampx.data.ingest import harmonize_activity


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def read_positives(path: str, sequence_column: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".fasta", ".fa", ".faa"}:
        return pd.DataFrame({sequence_column: read_fasta(path)})
    frame = read_table(path)
    if sequence_column not in frame.columns:
        raise KeyError(f"{path} has no column {sequence_column!r}; columns are {list(frame.columns)}")
    return frame


def gram_selectivity(activity, metadata, sequence_column: str = "sequence"):
    """
    Per-sequence log10(MIC gram-positive / MIC gram-negative).

    Positive values mean the peptide is more potent against gram-negatives
    (it takes more of it to inhibit a gram-positive). NaN where the peptide
    lacks a MIC against either class, which leaves the axis masked for that
    row exactly like any other missing conditioning value.
    """
    from ampx.models.features import gram_for_species

    mic = activity[(activity["endpoint"] == "mic") & activity["value_um"].gt(0)]
    mic = mic.dropna(subset=["sequence", "species", "value_um"])
    if mic.empty:
        return pd.Series(np.nan, index=metadata.index)

    mic = mic.assign(
        log_mic=np.log10(mic["value_um"].astype(float)),
        gram=mic["species"].apply(gram_for_species),
    )
    mic = mic[mic["gram"].isin(["negative", "positive"])]

    per_class = mic.groupby(["sequence", "gram"])["log_mic"].mean().unstack("gram")
    if not {"negative", "positive"}.issubset(per_class.columns):
        return pd.Series(np.nan, index=metadata.index)

    ratio = (per_class["positive"] - per_class["negative"]).dropna()
    print(f"gram selectivity     : {len(ratio)} sequences measured against both classes")
    return metadata[sequence_column].map(ratio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", required=True)
    parser.add_argument(
        "--negatives", default=None,
        help="FASTA of general (non-AMP) short peptides, e.g. filtered sORFdb",
    )
    parser.add_argument("--sources", default=None, help="JSON activity source spec")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--min-species-count", type=int, default=20)
    parser.add_argument(
        "--label-column", default="label",
        help="column marking AMP membership; ignored if absent",
    )
    parser.add_argument("--label-value", type=int, default=1)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    positives = read_positives(args.positives, args.sequence_column)

    # The MLAMP export carries its own negatives (tier == non_amp_function).
    # Training a generator on those teaches it to produce non-AMPs, so they
    # are dropped here rather than relied on to be absent.
    if args.label_column in positives.columns:
        before = len(positives)
        positives = positives[positives[args.label_column] == args.label_value]
        print(
            f"positives: kept {len(positives)} of {before} rows with "
            f"{args.label_column} == {args.label_value}"
        )

    positives[args.sequence_column] = positives[args.sequence_column].apply(clean_sequence)
    before = len(positives)
    positives = positives[positives[args.sequence_column].apply(is_valid_sequence)]
    print(f"positives: {len(positives)} of {before} inside the 8-50 residue, standard-residue window")

    # General (non-AMP) short peptides for the is_amp contrast. Every sparse
    # axis (species, MIC, HC50) stays absent for these rows -- there is no
    # MIC for a sORFdb sequence and none is asserted -- only is_amp_idx=1 is
    # set here; is_amp_idx=2 for the AMP rows is filled in by
    # attach_conditioning_features rather than set explicitly on this side,
    # so a plain positives-only build (no --negatives) is unaffected.
    if args.negatives:
        negatives = pd.DataFrame({args.sequence_column: read_fasta(args.negatives)})
        negatives[args.sequence_column] = negatives[args.sequence_column].apply(clean_sequence)
        before = len(negatives)
        negatives = negatives[negatives[args.sequence_column].apply(is_valid_sequence)]
        negatives = negatives.drop_duplicates(subset=[args.sequence_column])
        negatives["is_amp_idx"] = 1
        print(
            f"negatives: {len(negatives)} of {before} inside the 8-50 residue, "
            f"standard-residue window, from {args.negatives}"
        )
        positives = pd.concat([positives, negatives], ignore_index=True, sort=False)

    blocks = []
    if args.sources:
        spec = json.loads(Path(args.sources).read_text())
        for entry in spec:
            raw = read_table(entry["path"])
            block = harmonize_activity(
                raw,
                column_map=entry["columns"],
                endpoint=entry["endpoint"],
                source=entry.get("source", Path(entry["path"]).stem),
                default_unit=entry.get("default_unit"),
                row_filters=entry.get("row_filters"),
            )
            print(
                f"{entry['path']}: {len(block)} usable {entry['endpoint']} rows "
                f"from {len(raw)} records"
            )
            blocks.append(block)

    activity = (
        pd.concat(blocks, ignore_index=True)
        if blocks
        else pd.DataFrame(columns=["sequence", "species", "strain", "endpoint",
                                   "value_um", "censored", "source"])
    )
    activity.to_csv(output / "activity_harmonized.csv", index=False)

    from ampx.data.ingest import build_metadata

    metadata = build_metadata(positives, activity, sequence_column=args.sequence_column)

    # Gram selectivity, per sequence: log10 of the geometric-mean MIC against
    # gram-positive species over the geometric-mean MIC against gram-negative
    # ones. Defined only for peptides measured against both classes, which is
    # many more peptides than any single species pair supplies.
    metadata["gram_selectivity_raw"] = gram_selectivity(
        activity, metadata, sequence_column=args.sequence_column
    )

    metadata = attach_conditioning_features(metadata, sequence_column=args.sequence_column)

    species_vocab = build_species_vocabulary(metadata, min_count=args.min_species_count)
    (output / "species_vocab.json").write_text(json.dumps(species_vocab, indent=2))

    metadata.to_csv(output / "metadata.csv", index=False)

    print()
    print(f"metadata rows        : {len(metadata)}")
    print(f"unique sequences     : {metadata[args.sequence_column].nunique()}")
    print(f"rows with MIC        : {int(metadata['log_mic'].notna().sum())}")
    print(f"rows with HC50       : {int(metadata['log_hc50'].notna().sum())}")
    print(f"rows with gram sel.  : {int(metadata['gram_selectivity'].notna().sum())}")
    print(f"right-censored MIC   : {int((metadata['mic_censored_idx'] == 2).sum())}")
    print(f"species in vocabulary: {len(species_vocab)}")
    print(f"cys_class > 2 Cys    : {int((metadata['cys_class_idx'] == 3).sum())}")
    print()
    print(f"wrote {output / 'metadata.csv'}")
    print(f"wrote {output / 'species_vocab.json'}")
    print(f"wrote {output / 'activity_harmonized.csv'}")


if __name__ == "__main__":
    main()
