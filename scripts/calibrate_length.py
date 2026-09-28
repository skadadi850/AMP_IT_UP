"""
Calibrate the PAD-logit bias so the learned length distribution matches the data.

The free-length sampler produces a learned but short-biased length distribution
(median 16 against MLAMP's 18). This fits ONE scalar -- a constant added to the
PAD logit at sampling time -- by minimising the KS distance between generated
and reference lengths.

Tuned on the TRAINING folds only (fold != holdout). Fitting it against the
held-out fold would spend that fold on calibration and invalidate every
held-out statistic computed afterwards, which is the whole point of having it.
One scalar on 31,661 sequences is a negligible fit, but it still belongs on
data the model was already allowed to see.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ampx.models.data import cluster_split
from ampx.models.masked_diffusion import MaskedDiffusionModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--metadata", default="data/processed/metadata.csv")
    ap.add_argument("--splits", default="data/raw/splits_with_negatives.csv")
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--reveal", default="structured", choices=["uniform", "structured"])
    ap.add_argument("--mode-weights", type=float, nargs=3, default=None)
    ap.add_argument("--seed", type=int, default=7,
                    help="calibration seed, deliberately not 0/1/2 so the fit "
                         "is not made on the same draws used for evaluation")
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0])
    ap.add_argument("--out", default="results/comparison/variants/length_calibration.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata)
    splits = pd.read_csv(args.splits)
    train_idx, _ = cluster_split(meta, splits=splits, holdout_fold=args.holdout_fold)
    train_len = meta["sequence"].astype(str).str.len().to_numpy()[np.asarray(train_idx)]
    train_len = train_len[(train_len >= 8) & (train_len <= 50)]
    print(f"calibration reference: {len(train_len)} training-fold sequences, "
          f"median {np.median(train_len):.0f}, mean {train_len.mean():.2f}")

    model = MaskedDiffusionModel.load(args.checkpoint, device=args.device)
    if args.mode_weights is not None:
        model.mask_mode_weights = tuple(args.mode_weights)

    rows = []
    for bias in args.grid:
        gen = torch.Generator(device=args.device)
        gen.manual_seed(args.seed)
        seqs = []
        for start in range(0, args.num_samples, args.batch_size):
            n = min(args.batch_size, args.num_samples - start)
            seqs.extend(model.sample(
                batch_size=n, condition=None, guidance_weight=0.0,
                num_steps=args.steps, generator=gen,
                free_length=True, reveal=args.reveal, pad_bias=bias))
        L = np.array([len(s) for s in seqs])
        scoped = L[(L >= 8) & (L <= 50)]
        ks = stats.ks_2samp(scoped, train_len).statistic if len(scoped) else np.nan
        rows.append({
            "pad_bias": bias, "n": len(L), "n_in_scope": int(len(scoped)),
            "pct_in_scope": 100.0 * len(scoped) / max(len(L), 1),
            "median_length": float(np.median(scoped)) if len(scoped) else np.nan,
            "mean_length": float(scoped.mean()) if len(scoped) else np.nan,
            "ks_length_vs_train": float(ks),
        })
        print(f"  pad_bias {bias:+.2f}  median {rows[-1]['median_length']:5.1f}  "
              f"mean {rows[-1]['mean_length']:5.2f}  in-scope {rows[-1]['pct_in_scope']:5.1f}%  "
              f"ks_length {ks:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    best = frame.loc[frame["ks_length_vs_train"].idxmin()]
    print(f"\nbest pad_bias = {best['pad_bias']:+.2f} "
          f"(ks_length {best['ks_length_vs_train']:.4f}, median {best['median_length']:.1f})")
    (out.with_suffix(".json")).write_text(json.dumps({
        "best_pad_bias": float(best["pad_bias"]),
        "ks_length_vs_train": float(best["ks_length_vs_train"]),
        "calibration_reference": "training folds only (fold != %d)" % args.holdout_fold,
        "calibration_seed": args.seed,
        "reveal": args.reveal,
        "checkpoint": args.checkpoint,
    }, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
