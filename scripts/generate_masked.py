"""
Sample from the masked discrete diffusion model.

Usage:
    python scripts/generate_masked.py \
        --model checkpoint/masked_diffusion_best.pt \
        --species-vocab checkpoint/species_vocab.json \
        --sweep --num-samples 1000

    python scripts/generate_masked.py \
        --model checkpoint/masked_diffusion_best.pt \
        --species-vocab checkpoint/species_vocab.json \
        --request request.json \
        --reference data/antibacterial.fasta \
        --num-samples 50000 \
        --guidance-weight 2.0 --steps 128 \
        --output results/candidates/library.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ampx.models.compliance import (
    MAX_REFERENCE_IDENTITY,
    NoveltyFilter,
    compliance_report,
    read_fasta,
    write_fasta,
)
from ampx.models.features import (
    MAX_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    normalize_length,
)
from ampx.models.masked_diffusion import MaskedDiffusionModel
from ampx.models.sampling import build_condition, candidate_frame, length_grid
from ampx.models.sampling import (build_condition, candidate_frame,
                                  empirical_length_counts, length_grid,
                                  sample_library)

# Requested axis -> realized column, and the transform needed to compare.
FIDELITY_AXES = {
    "length_norm": ("length", normalize_length),
    "charge_per_res": ("charge_per_res", None),
    "mu_h": ("mu_h", None),
    "gravy": ("gravy", None),
}


def build_length_counts(total: int, min_length: int, max_length: int) -> dict:
    lengths = list(range(min_length, max_length + 1))
    base = total // len(lengths)
    counts = {length: base for length in lengths}
    for index in range(total - base * len(lengths)):
        counts[lengths[index % len(lengths)]] += 1
    return counts


@torch.no_grad()
def axis_error(frame: pd.DataFrame, condition, axis: str) -> float:
    if float(condition[f"{axis}_mask"][0]) == 0.0:
        return float("nan")
    target = float(condition[axis][0])
    column, transform = FIDELITY_AXES[axis]
    realized = frame[column].astype(float)
    if transform is not None:
        realized = realized.apply(transform)
    return float(np.abs(realized - target).mean())


def run_sweep(model, request, species_vocab, args) -> pd.DataFrame:
    if "length" not in request:
        request["length"] = 20
    condition = build_condition(request, args.num_samples, species_vocab, device=args.device)

    rows = []
    for steps in args.sweep_steps:
        for weight in args.sweep_weights:
            sequences = sample_library(
                model, condition, weight, steps, args.temperature,
                args.batch_size, args.seed,
                free_length=args.free_length, reveal=args.reveal, pad_bias=args.pad_bias,
            )
            frame = candidate_frame(sequences)
            if frame.empty:
                continue

            record = {
                "steps": steps,
                "guidance_weight": weight,
                "generated": len(sequences),
                "unique": len(frame),
                "uniqueness": len(frame) / max(len(sequences), 1),
                "compliant_fraction": float(
                    frame["sequence"].apply(lambda s: all(compliance_report(s).values())).mean()
                ),
                "median_length": float(frame["length"].median()),
                "median_net_charge": float(frame["net_charge"].median()),
                "median_mu_h": float(frame["mu_h"].median()),
            }
            errors = []
            for axis in FIDELITY_AXES:
                error = axis_error(frame, condition, axis)
                record[f"error_{axis}"] = error
                if np.isfinite(error):
                    errors.append(error)
            record["mean_axis_error"] = float(np.mean(errors)) if errors else float("nan")
            rows.append(record)

            print(
                f"steps {steps:>4}  w {weight:>4.1f}  unique {record['uniqueness']:.3f}  "
                f"compliant {record['compliant_fraction']:.3f}  "
                f"axis error {record['mean_axis_error']:.4f}"
            )

    table = pd.DataFrame(rows)
    if not table.empty:
        # Guidance always trades diversity for control, so a setting that
        # hits the target by emitting near-duplicates is not usable for a
        # library. Select on fidelity subject to a uniqueness floor.
        eligible = table[table["uniqueness"] >= 0.5]
        if eligible.empty:
            eligible = table
        best = eligible.loc[eligible["mean_axis_error"].idxmin()]
        print()
        print(f"best setting: steps {int(best['steps'])}, w {best['guidance_weight']:.1f}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    # Both default to the checkpoint pair shipped in checkpoint/. They belong
    # together: the vocab fixes the species index the condition encoder was
    # trained against, so a vocab from a different corpus silently shifts every
    # species id.
    parser.add_argument("--model", default="checkpoint/masked_diffusion_best.pt")
    parser.add_argument("--species-vocab", default="checkpoint/species_vocab.json")
    parser.add_argument("--request", default=None)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--guidance-weight", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-length", type=int, default=MIN_PEPTIDE_LENGTH)
    parser.add_argument("--max-length", type=int, default=MAX_PEPTIDE_LENGTH)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--identity-threshold", type=float, default=MAX_REFERENCE_IDENTITY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-weights", type=float, nargs="+",
                        default=[0.0, 1.0, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--sweep-steps", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument(
        "--free-length", action="store_true",
        help="let the model place PAD itself instead of pinning the requested length "
             "(the 'free' in gramp_face_free_struct_cal)",
    )
    parser.add_argument(
        "--reveal", choices=["uniform", "structured"], default="uniform",
        help="'structured' unmasks in an order matching the training corruption mixture "
             "(the 'struct' in gramp_face_free_struct_cal); default 'uniform' reproduces "
             "the original sampler",
    )
    parser.add_argument(
        "--pad-bias", type=float, default=0.0,
        help="constant added to the PAD logit at sampling time, meaningful only with "
             "--free-length (the 'cal' in gramp_face_free_struct_cal, fitted to -0.15 for "
             "the reference checkpoint via scripts/comparison/calibrate_length.py)",
    )
    parser.add_argument("--output", default="results/candidates/library.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--real-lengths", default=None,
                        help="metadata csv whose length column sets the sampling distribution")
    args = parser.parse_args()

    species_vocab = json.loads(Path(args.species_vocab).read_text())
    request = json.loads(Path(args.request).read_text()) if args.request else {}
    print(f"request: {json.dumps(request)}")

    model = MaskedDiffusionModel.load(args.model, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        table = run_sweep(model, request, species_vocab, args)
        sweep_path = output.with_name("guidance_sweep.csv")
        table.to_csv(sweep_path, index=False)
        print(f"wrote {sweep_path}")
        return

    real_lengths = pd.read_csv(args.real_lengths)["length"] if args.real_lengths else None
    counts = (
        empirical_length_counts(real_lengths, args.num_samples)
        if real_lengths is not None
        else build_length_counts(args.num_samples, args.min_length, args.max_length)
    )
    condition = length_grid(request, counts, species_vocab, device=args.device)

    sequences = sample_library(
        model, condition, args.guidance_weight, args.steps,
        args.temperature, args.batch_size, args.seed,
        free_length=args.free_length, reveal=args.reveal, pad_bias=args.pad_bias,
    )
    print(f"generated {len(sequences)} sequences")

    frame = candidate_frame(sequences)
    print(f"unique after dedup: {len(frame)}")
    if frame.empty:
        print("no sequences produced")
        return

    checks = frame["sequence"].apply(compliance_report).apply(pd.Series)
    for column in checks.columns:
        frame[column] = checks[column]
    frame["compliant"] = checks.all(axis=1)
    print(f"compliant: {int(frame['compliant'].sum())}")
    frame = frame[frame["compliant"]].reset_index(drop=True)

    if args.reference:
        # The validator rejects the library for any exact match to the
        # reference set, so these are dropped before novelty scoring rather
        # than being ranked and then found to be unsubmittable. This is also
        # the honest measure of verbatim memorization: the count here is the
        # fraction of output that reproduces a training sequence exactly.
        reference_exact = set(read_fasta(args.reference))
        duplicated = frame["sequence"].isin(reference_exact)
        print(
            f"exact reference matches removed: {int(duplicated.sum())} "
            f"({100.0 * duplicated.mean():.2f}% of unique output)"
        )
        frame = frame[~duplicated].reset_index(drop=True)

    if args.reference:
        references = read_fasta(args.reference)
        print(f"reference set: {len(references)} sequences")
        novelty = NoveltyFilter(references)
        identities = []
        nearest = []
        for sequence in tqdm(frame["sequence"], desc="novelty"):
            identity, reference = novelty.max_identity(sequence)
            identities.append(identity)
            nearest.append(reference or "")
        frame["max_reference_identity"] = identities
        frame["nearest_reference"] = nearest
        frame["novel"] = frame["max_reference_identity"] < args.identity_threshold
        print(
            f"novel below {args.identity_threshold:.0%} identity: "
            f"{int(frame['novel'].sum())} of {len(frame)}"
        )
    else:
        frame["max_reference_identity"] = np.nan
        frame["novel"] = True

    frame.to_csv(output, index=False)
    submittable = frame[frame["novel"]]["sequence"].tolist()
    write_fasta(str(output.with_suffix(".fasta")), submittable)

    print()
    print(f"median length     : {frame['length'].median():.1f}")
    print(f"median net charge : {frame['net_charge'].median():.1f}")
    print(f"median mu_h       : {frame['mu_h'].median():.3f}")
    print(f"median GRAVY      : {frame['gravy_raw'].median():.3f}")
    print(f"fraction with Cys : {float((frame['cys_count'] > 0).mean()):.3f}")
    print()
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.fasta')} with {len(submittable)} submittable sequences")


if __name__ == "__main__":
    main()