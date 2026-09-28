"""
Step 3 -- core metrics, per seed and aggregated, plus the paired face-vs-iid test.

Every metric comes from ampx.models.metrics; none is reimplemented here.

Note on the primary metric. ampx.models.metrics hard-codes
PRIMARY_METRIC = "expert_range_pct", but this task pre-registers ks_mu_h as
primary with expert_range_pct and submittable_pct as secondaries. The task's
pre-registration governs; the module constant is left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ampx.models.compliance import NoveltyFilter, read_fasta
from ampx.models.metrics import (aggregate_seeds, paired_delta, reference_set,
                           sequence_properties, set_metrics)

MODELS = ["gramp_md", "gramp_face", "gramp_latent", "gramp_latent_ft",
          "omegamp", "ampdiffusion"]
SEEDS = [0, 1, 2]

PRIMARY = "ks_mu_h"
SECONDARY = ["expert_range_pct", "submittable_pct"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/processed/metadata.csv")
    parser.add_argument("--reference", default="data/antibacterial.fasta")
    parser.add_argument("--libraries", default="results/comparison/libraries")
    parser.add_argument("--outdir", default="results/comparison")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    real_sequences = metadata.loc[metadata["label"] == 1, "sequence"].astype(str).tolist()
    print(f"MLAMP label==1 reference: {len(real_sequences)} sequences")

    real_frame = sequence_properties(real_sequences)
    real = reference_set(real_frame)
    novelty = NoveltyFilter(read_fasta(args.reference))
    print(f"exclusion set: {len(novelty.references)} sequences")

    per_seed = []
    records_by_model = {}

    for model in MODELS:
        records = []
        for seed in SEEDS:
            path = Path(args.libraries) / f"{model}_seed{seed}.fasta"
            if not path.exists():
                print(f"MISSING {path.name} -- skipped")
                continue
            sequences = read_fasta(str(path))
            record = set_metrics(sequences, real, novelty=novelty, seed=seed)
            records.append(record)
            row = {"model": model, "seed": seed, **record}
            per_seed.append(row)
            print(f"{model:18s} seed {seed}  {PRIMARY} {record.get(PRIMARY, float('nan')):.4f}  "
                  f"expert {record.get('expert_range_pct', float('nan')):.2f}  "
                  f"submittable {record.get('submittable_pct', float('nan')):.2f}")
        records_by_model[model] = records

    # real_amps: the target, computed on MLAMP with exactly the same code.
    real_record = set_metrics(real_sequences, real, novelty=novelty, seed=0)
    per_seed.append({"model": "real_amps", "seed": "-", **real_record})
    print(f"\nreal_amps  {PRIMARY} {real_record.get(PRIMARY, float('nan')):.4f}  "
          f"expert {real_record.get('expert_range_pct', float('nan')):.2f}  "
          f"submittable {real_record.get('submittable_pct', float('nan')):.2f}")

    pd.DataFrame(per_seed).to_csv(outdir / "metrics_per_seed.csv", index=False)
    print(f"wrote {outdir / 'metrics_per_seed.csv'}")

    summary = []
    for model in MODELS:
        if not records_by_model.get(model):
            continue
        summary.append({"model": model, **aggregate_seeds(records_by_model[model])})
    summary.append({"model": "real_amps", "n_seeds": 1, **real_record})
    pd.DataFrame(summary).to_csv(outdir / "metrics_summary.csv", index=False)
    print(f"wrote {outdir / 'metrics_summary.csv'}")

    # Paired gramp_face vs gramp_md over the three shared seeds.
    base, var = records_by_model.get("gramp_md", []), records_by_model.get("gramp_face", [])
    rows = []
    if len(base) == len(var) and base:
        for metric in [PRIMARY] + SECONDARY:
            delta = paired_delta(base, var, metric)
            rows.append({
                "metric": metric,
                "role": "primary" if metric == PRIMARY else "secondary",
                "gramp_md_mean": float(pd.Series([r[metric] for r in base if metric in r]).mean()),
                "gramp_face_mean": float(pd.Series([r[metric] for r in var if metric in r]).mean()),
                "n_seeds": len(base),
                **delta,
            })
            print(f"paired {metric}: delta {delta.get('delta', float('nan')):+.4f} "
                  f"(face - md), p {delta.get('p', float('nan'))}")
    else:
        print("cannot pair gramp_face against gramp_md -- unequal or missing seed sets")
    pd.DataFrame(rows).to_csv(outdir / "face_vs_iid.csv", index=False)
    print(f"wrote {outdir / 'face_vs_iid.csv'}")


if __name__ == "__main__":
    main()
