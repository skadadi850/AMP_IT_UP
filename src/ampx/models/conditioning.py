"""
Conditioning for AMP generation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

# Categorical axes. Index 0 always means unknown, so masking an axis is
# writing a zero and nothing else has to change.
CATEGORICAL_AXES = (
    "is_amp", "species", "gram", "cys_class", "mic_censored", "hc50_source",
)

# Axes exempt from conditional dropout. Their index-0 ("unknown") embedding
# row therefore never receives a gradient update, which makes the all-unknown
# state an untrained vector for these axes -- so anything that builds an
# unconditional reference (composition checks, the classifier-free guidance
# denominator) has to carry the requested value through instead of zeroing it.
DROPOUT_EXEMPT_AXES = ("is_amp",)

# The continuous axis list as it stood before `gram_selectivity` was added.
# Checkpoints saved before that change do not record their axis list, so this
# is what they are loaded with.
LEGACY_CONTINUOUS_AXES = (
    "log_mic", "log_hc50", "length_norm", "charge_per_res", "mu_h", "gravy",
)

# Fixed sizes; species is data dependent and passed in at construction.
CATEGORICAL_SIZES = {
    "is_amp": 3,         # 0 unknown, 1 general peptide, 2 validated AMP
    "gram": 5,           # unknown, negative, positive, fungal, mycobacterial
    "cys_class": 4,      # unknown, zero Cys, two Cys, more than two Cys
    "mic_censored": 3,   # unknown, exact, right-censored
    "hc50_source": 3,    # 0 unknown, 1 measured, 2 predicted
}

# Continuous axes, each carrying a companion "<axis>_mask" tensor.
CONTINUOUS_AXES = (
    "log_mic",
    "log_hc50",
    "length_norm",
    "charge_per_res",
    "mu_h",
    "gravy",
    # Gram selectivity: log10(MIC against gram-positives / MIC against
    # gram-negatives), normalized. Species identity turned out to be nearly
    # unlearnable -- ~90% of peptides with a MIC against one species also have
    # one against the other, and their MICs correlate at r = 0.67, so
    # broad-spectrum activity dominates and "which species" is barely
    # identifiable from sequence. The differential between the two classes is
    # the part that is identifiable, and it is dense over every peptide
    # measured against both classes rather than sparse per species.
    "gram_selectivity",
)

# Axes computable from the bare sequence. These are never missing in
# training data, so they carry the bulk of the guidance signal.
DENSE_AXES = ("length_norm", "charge_per_res", "mu_h", "gravy", "cys_class")

CONDITION_KEYS = (
    tuple(f"{axis}_idx" for axis in CATEGORICAL_AXES)
    + CONTINUOUS_AXES
    + tuple(f"{axis}_mask" for axis in CONTINUOUS_AXES)
)


def categorical_sizes(num_species: int) -> Dict[str, int]:
    sizes = {"species": int(num_species)}
    sizes.update(CATEGORICAL_SIZES)
    return {axis: sizes[axis] for axis in CATEGORICAL_AXES}


def normalize_condition(
    condition: Dict[str, torch.Tensor],
    batch_size: int,
    device,
) -> Dict[str, torch.Tensor]:
    """
    Coerce a user-supplied condition dict to canonical shapes and fill in
    anything absent as unknown. Categorical axes become (B,) long,
    continuous axes and masks become (B, 1) float.
    """
    out: Dict[str, torch.Tensor] = {}

    for axis in CATEGORICAL_AXES:
        key = f"{axis}_idx"
        value = condition.get(key)
        if value is None:
            out[key] = torch.zeros(batch_size, dtype=torch.long, device=device)
            continue
        value = torch.as_tensor(value, device=device).long().reshape(-1)
        if value.numel() == 1 and batch_size > 1:
            value = value.expand(batch_size)
        out[key] = value

    for axis in CONTINUOUS_AXES:
        value = condition.get(axis)
        mask = condition.get(f"{axis}_mask")
        if value is None:
            out[axis] = torch.zeros(batch_size, 1, device=device)
            out[f"{axis}_mask"] = torch.zeros(batch_size, 1, device=device)
            continue
        value = torch.as_tensor(value, device=device).float().reshape(-1, 1)
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, 1)
        if mask is None:
            mask = torch.ones_like(value)
        else:
            mask = torch.as_tensor(mask, device=device).float().reshape(-1, 1)
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size, 1)
        out[axis] = value.contiguous()
        out[f"{axis}_mask"] = mask.contiguous()

    return out


class AMPConditionEncoder(nn.Module):
    """
    Encodes the AMP conditioning axes into one vector.

    Categorical axes use an embedding table whose row 0 is the unknown
    state, zero-initialized so an unknown axis starts as no contribution.
    Continuous axes use a small MLP gated by their observation mask, with a
    learned null vector substituted when the mask is zero. Nothing is ever
    imputed: a missing MIC is represented as missing, not as the dataset
    median.
    """

    def __init__(
        self,
        num_species: int,
        embed_dim: int = 128,
        output_dim: int = 1024,
        continuous_axes=None,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.output_dim = int(output_dim)
        self.num_species = int(num_species)
        self.categorical_axes = CATEGORICAL_AXES
        # Adding a continuous axis changes the encoder's parameter shapes, so
        # a checkpoint trained before the axis existed cannot be loaded under
        # the new list. The axis list is therefore part of the saved config
        # and is read back on load, rather than being whatever the module
        # constant happens to say today.
        self.continuous_axes = tuple(
            CONTINUOUS_AXES if continuous_axes is None else continuous_axes
        )
        self.axis_sizes = categorical_sizes(num_species)

        self.categorical = nn.ModuleDict({
            axis: nn.Embedding(size, self.embed_dim)
            for axis, size in self.axis_sizes.items()
        })
        self.continuous = nn.ModuleDict({
            axis: nn.Sequential(
                nn.Linear(1, self.embed_dim // 2),
                nn.GELU(),
                nn.Linear(self.embed_dim // 2, self.embed_dim),
            )
            for axis in self.continuous_axes
        })
        self.null = nn.ParameterDict({
            axis: nn.Parameter(torch.zeros(1, self.embed_dim))
            for axis in self.continuous_axes
        })

        concat_dim = self.embed_dim * (len(self.categorical_axes) + len(self.continuous_axes))
        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

        for embedding in self.categorical.values():
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        for parameter in self.null.values():
            nn.init.normal_(parameter, std=0.02)

    def null_condition(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        """The all-unknown condition, for the guidance denominator."""
        condition: Dict[str, torch.Tensor] = {}
        for axis in self.categorical_axes:
            condition[f"{axis}_idx"] = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
        for axis in self.continuous_axes:
            condition[axis] = torch.zeros(batch_size, 1, device=device)
            condition[f"{axis}_mask"] = torch.zeros(batch_size, 1, device=device)
        return condition

    def forward(self, condition: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for axis in self.categorical_axes:
            index = condition[f"{axis}_idx"].long().reshape(-1)
            parts.append(self.categorical[axis](index))
        for axis in self.continuous_axes:
            value = condition[axis].float().reshape(-1, 1)
            mask = condition.get(f"{axis}_mask")
            mask = torch.ones_like(value) if mask is None else mask.float().reshape(-1, 1)
            projected = self.continuous[axis](value * mask)
            parts.append(projected * mask + self.null[axis] * (1.0 - mask))
        return self.output_proj(torch.cat(parts, dim=-1))


def drop_condition_axes(
    condition: Dict[str, torch.Tensor],
    p_axis: float = 0.15,
    p_all: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """
    Per-sample, per-axis conditional dropout.

    p_all is the chance a sample becomes fully unconditional, which is what
    calibrates the guidance denominator. p_axis is the independent chance
    each surviving axis is dropped, which is what teaches the model to
    handle arbitrary subsets of specified axes at sampling time. Both are
    drawn per sample.

    is_amp is exempt from both p_axis and p_all dropout: it is the axis
    separating the pretraining corpus (general peptides vs. validated AMPs)
    from each other, so hiding it would make the two training distributions
    indistinguishable to the model.
    """
    reference = condition[f"{CATEGORICAL_AXES[0]}_idx"]
    batch_size = reference.shape[0]
    device = reference.device

    keep_all = (torch.rand(batch_size, device=device) >= p_all).float()
    out: Dict[str, torch.Tensor] = {}

    for axis in CATEGORICAL_AXES:
        key = f"{axis}_idx"
        if axis in DROPOUT_EXEMPT_AXES:
            out[key] = condition[key].long().reshape(-1)
            continue
        keep = (torch.rand(batch_size, device=device) >= p_axis).float() * keep_all
        out[key] = (condition[key].long().reshape(-1) * keep.long())

    for axis in CONTINUOUS_AXES:
        value = condition[axis].float().reshape(-1, 1)
        mask = condition.get(f"{axis}_mask")
        mask = torch.ones_like(value) if mask is None else mask.float().reshape(-1, 1)
        keep = (torch.rand(batch_size, 1, device=device) >= p_axis).float()
        out[axis] = value
        out[f"{axis}_mask"] = mask * keep * keep_all.reshape(-1, 1)

    return out


def concat_conditions(
    first: Dict[str, torch.Tensor],
    second: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Stack two conditions along the batch axis, for one-pass guidance."""
    return {key: torch.cat([first[key], second[key]], dim=0) for key in first}


def select_condition(
    condition: Dict[str, torch.Tensor],
    start: int,
    stop: int,
) -> Dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in condition.items()}


def condition_from_batch(
    batch: Dict[str, torch.Tensor],
    device,
) -> Dict[str, torch.Tensor]:
    """Pull the condition tensors out of a dataloader batch."""
    return {key: batch[key].to(device) for key in CONDITION_KEYS if key in batch}


def describe_condition(condition: Dict[str, torch.Tensor], index: int = 0) -> str:
    parts = []
    for axis in CATEGORICAL_AXES:
        parts.append(f"{axis}={int(condition[f'{axis}_idx'][index])}")
    for axis in CONTINUOUS_AXES:
        mask = float(condition[f"{axis}_mask"][index])
        if mask > 0:
            parts.append(f"{axis}={float(condition[axis][index]):.3f}")
        else:
            parts.append(f"{axis}=unknown")
    return " ".join(parts)


__all__ = [
    "CATEGORICAL_AXES",
    "CATEGORICAL_SIZES",
    "CONTINUOUS_AXES",
    "CONDITION_KEYS",
    "DENSE_AXES",
    "AMPConditionEncoder",
    "categorical_sizes",
    "concat_conditions",
    "condition_from_batch",
    "describe_condition",
    "drop_condition_axes",
    "normalize_condition",
    "select_condition",
]
