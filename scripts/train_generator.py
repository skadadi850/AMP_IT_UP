"""
Train the masked discrete diffusion model

Usage:
    python scripts/train_masked_diffusion.py \
        --metadata data/processed/metadata.csv \
        --species-vocab checkpoints/species_vocab.json \
        --splits data/raw/splits_with_negatives.csv \
        --holdout-fold 0 \
        --output checkpoints
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from ampx.models.conditioning import LEGACY_CONTINUOUS_AXES, condition_from_batch
from ampx.models.data import (
    TokenDataset,
    cluster_split,
    is_amp_sampler_weights,
    stratified_sampler_weights,
)
from ampx.models.masked_diffusion import MaskedDiffusionModel

# Shape mismatches --init-from is allowed to hit. The species embedding
# table is data-dependent (pretrain and conditioned stages build their own
# species_vocab.json from different corpora). The condition encoder's output
# projection changes width whenever a conditioning axis is added, since it
# consumes one embedding block per axis; the new axis's own parameters are
# absent from an older checkpoint rather than mismatched, and land in
# `missing` instead.
EXPECTED_INIT_FROM_SKIPS = {
    "condition_encoder.categorical.species.weight",
    "condition_encoder.output_proj.0.weight",
}


def load_compatible_state_dict(
    model: MaskedDiffusionModel,
    checkpoint_path: str,
    device: str,
    reinit_condition_proj: bool = False,
) -> None:
    """
    Load a checkpoint's state dict, keeping only keys whose shape matches
    the freshly constructed model. Used for stage 5b's --init-from, where
    num_species (and therefore the species embedding shape) differs between
    the pretrain and conditioned species vocabularies.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = dict(checkpoint["model_state_dict"])
    target_state = model.state_dict()

    if reinit_condition_proj:
        # Stage 5a pretrains with every sparse axis absent, so the condition
        # encoder's output projection learns to ignore those input slots.
        # Inheriting it into stage 5b starts the fine-tune from weights that
        # already discard MIC/HC50/species, and a short low-LR fine-tune does
        # not undo that. Dropping it forces the projection to relearn the
        # mapping with all axes present.
        dropped = [k for k in source_state if k.startswith("condition_encoder.output_proj")]
        for key in dropped:
            source_state.pop(key)
        print(f"--reinit-condition-proj: dropped {len(dropped)} keys: {dropped}")

    compatible = {}
    skipped = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing = [key for key in target_state if key not in compatible]

    model.load_state_dict(compatible, strict=False)
    print(f"--init-from {checkpoint_path}: loaded {len(compatible)} keys, skipped {len(skipped)}")
    if skipped:
        print(f"skipped keys (shape mismatch or absent in checkpoint): {skipped}")
    print(f"keys left at fresh initialization (not present in checkpoint): {missing}")

    if reinit_condition_proj:
        skipped = [k for k in skipped if not k.startswith("condition_encoder.output_proj")]
    unexpected = set(skipped) - EXPECTED_INIT_FROM_SKIPS
    if unexpected:
        raise SystemExit(
            "STOP: --init-from skipped keys other than the species embedding: "
            f"{sorted(unexpected)}. This is the flagged stop-and-report condition -- "
            "investigate before training from this checkpoint."
        )


def run_epoch(model, loader, optimizer, device, p_axis, p_all, train):
    model.train(train)
    total = 0.0
    batches = 0

    for batch in tqdm(loader, desc="train" if train else "val", leave=False):
        tokens = batch["tokens"].to(device)
        condition = condition_from_batch(batch, device)

        if train:
            loss = model(tokens, condition, p_axis=p_axis, p_all=p_all)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                loss = model(tokens, condition, p_axis=0.0, p_all=0.0)

        total += float(loss.item())
        batches += 1

    return total / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/processed/metadata.csv")
    parser.add_argument("--species-vocab", default="checkpoints/species_vocab.json")
    parser.add_argument("--splits", default=None)
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument("--output", default="checkpoints",
                        help="written into on every run, so training with the "
                             "default overwrites the shipped "
                             "masked_diffusion_best.pt and species_vocab.json. "
                             "Pass an experiment directory to keep them.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--p-axis", type=float, default=0.15)
    parser.add_argument("--p-all", type=float, default=0.10)
    parser.add_argument(
        "--mask-mode-weights",
        type=float,
        nargs=3,
        default=[0.5, 0.25, 0.25],
        metavar=("IID", "SPAN", "FACE"),
        help="mixture weights for iid / span / helical-face corruption",
    )
    parser.add_argument("--balance-species", action="store_true")
    parser.add_argument(
        "--balance-is-amp", action="store_true",
        help="~3:1 negative:positive sampler by is_amp_idx, for pretraining",
    )
    parser.add_argument(
        "--legacy-continuous-axes", action="store_true",
        help="train with the continuous axis list as it stood before "
             "gram_selectivity was added, so a metadata table built before "
             "that axis existed can still be used",
    )
    parser.add_argument(
        "--reinit-condition-proj", action="store_true",
        help="do not inherit the condition encoder's output projection from "
             "--init-from; reinitialize it so it must relearn how to use the "
             "sparse axes that were absent during pretraining",
    )
    parser.add_argument(
        "--init-from", default=None,
        help="checkpoint .pt to initialize from; shape-mismatched keys "
             "(expected: only the species embedding) are skipped",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    species_vocab = json.loads(Path(args.species_vocab).read_text())

    splits = pd.read_csv(args.splits) if args.splits else None
    train_index, val_index = cluster_split(
        metadata, splits=splits, holdout_fold=args.holdout_fold
    )
    print(f"train {len(train_index)} rows, val {len(val_index)} rows")

    continuous_axes = LEGACY_CONTINUOUS_AXES if args.legacy_continuous_axes else None
    dataset = TokenDataset(metadata, species_vocab, continuous_axes=continuous_axes)

    sampler = None
    shuffle = True
    if args.balance_species:
        weights = stratified_sampler_weights(dataset.metadata)[train_index]
        sampler = WeightedRandomSampler(
            weights=weights.tolist(), num_samples=len(train_index), replacement=True
        )
        shuffle = False
    elif args.balance_is_amp:
        weights = is_amp_sampler_weights(dataset.metadata)[train_index]
        sampler = WeightedRandomSampler(
            weights=weights.tolist(), num_samples=len(train_index), replacement=True
        )
        shuffle = False

    train_loader = DataLoader(
        Subset(dataset, train_index.tolist()),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_index.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = MaskedDiffusionModel(
        num_species=len(species_vocab) + 1,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        mask_mode_weights=tuple(args.mask_mode_weights),
        continuous_axes=continuous_axes,
        device=args.device,
    )
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"mask mode weights (iid, span, face): {model.mask_mode_weights}")

    if args.init_from:
        load_compatible_state_dict(
            model, args.init_from, args.device,
            reinit_condition_proj=args.reinit_condition_proj,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(args.warmup_epochs, 1)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-6
    )

    history = {"train": [], "val": []}
    best = float("inf")
    stale = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, args.device, args.p_axis, args.p_all, True
        )
        val_loss = run_epoch(model, val_loader, optimizer, args.device, 0.0, 0.0, False)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if epoch <= args.warmup_epochs:
            warmup.step()
        else:
            cosine.step()

        print(
            f"epoch {epoch}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best:
            best = val_loss
            stale = 0
            model.save(str(output / "masked_diffusion_best.pt"))
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    model.save(str(output / "masked_diffusion_final.pt"))
    (output / "species_vocab.json").write_text(json.dumps(species_vocab, indent=2))
    with open(output / "masked_diffusion_history.pkl", "wb") as handle:
        pickle.dump({"history": history, "args": vars(args), "best_val": best}, handle)

    print(f"best val loss {best:.4f}")
    print(f"wrote {output / 'masked_diffusion_best.pt'}")


if __name__ == "__main__":
    main()