"""
Turning a human-readable design request into a condition tensor, and a
condition tensor into candidate sequences.

Concentrations are given in uM and converted to the normalized log scale
internally, so the request reads in the units the assay reports.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .conditioning import CATEGORICAL_AXES, CONTINUOUS_AXES
from .features import (
    GRAM_INDEX,
    MAX_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    compute_sequence_features,
    gram_for_species,
    normalize_charge_per_residue,
    normalize_gravy,
    normalize_length,
    normalize_log_concentration,
    normalize_selectivity_ratio,
)

# Human-facing request keys and how they map onto axes.
REQUEST_KEYS = (
    "is_amp",
    "species",
    "gram",
    "hc50_source",
    "mic_um",
    "mic_censored",
    "hc50_um",
    "gram_selectivity",
    "length",
    "charge_per_res",
    "net_charge",
    "mu_h",
    "gravy",
    "cys_class",
)

# String spellings accepted for the two provenance axes, alongside a plain
# integer index.
IS_AMP_VALUES = {"unknown": 0, "general": 1, "amp": 2}
HC50_SOURCE_VALUES = {"unknown": 0, "measured": 1, "predicted": 2}


def build_condition(
    request: Dict,
    batch_size: int,
    species_vocab: Dict[str, int],
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Expand a design request into a full condition dict.

    Unspecified categorical axes get index 0 (unknown); unspecified
    continuous axes get mask 0. The length axis is always specified, since
    the decoder needs a target length to shape toward, and defaults to the
    midpoint of the competition range.
    """
    condition: Dict[str, torch.Tensor] = {}

    def categorical(axis: str, value: int) -> None:
        condition[f"{axis}_idx"] = torch.full(
            (batch_size,), int(value), dtype=torch.long, device=device
        )

    def continuous(axis: str, value: Optional[float]) -> None:
        if value is None or not np.isfinite(value):
            condition[axis] = torch.zeros(batch_size, 1, device=device)
            condition[f"{axis}_mask"] = torch.zeros(batch_size, 1, device=device)
        else:
            condition[axis] = torch.full((batch_size, 1), float(value), device=device)
            condition[f"{axis}_mask"] = torch.ones(batch_size, 1, device=device)

    def provenance(axis: str, values_map: Dict[str, int]) -> None:
        raw = request.get(axis)
        if raw is None:
            categorical(axis, 0)
        elif isinstance(raw, (int, float)):
            categorical(axis, int(raw))
        else:
            categorical(axis, values_map.get(str(raw).strip().lower(), 0))

    provenance("is_amp", IS_AMP_VALUES)
    provenance("hc50_source", HC50_SOURCE_VALUES)

    species = request.get("species")
    if species is None:
        categorical("species", 0)
        gram_name = request.get("gram", "unknown")
    else:
        key = str(species).strip().lower()
        categorical("species", species_vocab.get(key, 0))
        gram_name = request.get("gram") or gram_for_species(key)
    categorical("gram", GRAM_INDEX.get(str(gram_name).strip().lower(), 0))

    cys_class = request.get("cys_class")
    categorical("cys_class", 0 if cys_class is None else int(cys_class))

    mic_um = request.get("mic_um")
    if mic_um is None:
        categorical("mic_censored", 0)
        continuous("log_mic", None)
    else:
        censored = bool(request.get("mic_censored", False))
        categorical("mic_censored", 2 if censored else 1)
        continuous("log_mic", normalize_log_concentration(mic_um))

    hc50_um = request.get("hc50_um")
    continuous("log_hc50", None if hc50_um is None else normalize_log_concentration(hc50_um))

    # Requested as a raw log10 ratio: +1 means 10x more potent against
    # gram-negatives, -1 means 10x more potent against gram-positives.
    selectivity = request.get("gram_selectivity")
    continuous(
        "gram_selectivity",
        None if selectivity is None else normalize_selectivity_ratio(selectivity),
    )

    length = request.get("length")
    if length is None:
        continuous("length_norm", 0.5)
    else:
        continuous("length_norm", normalize_length(length))

    charge_per_res = request.get("charge_per_res")
    if charge_per_res is None and request.get("net_charge") is not None and length:
        charge_per_res = normalize_charge_per_residue(
            float(request["net_charge"]), int(length)
        )
    continuous("charge_per_res", charge_per_res)

    continuous("mu_h", request.get("mu_h"))

    gravy_value = request.get("gravy")
    continuous(
        "gravy",
        None if gravy_value is None else normalize_gravy(float(gravy_value)),
    )

    for axis in CATEGORICAL_AXES:
        condition.setdefault(
            f"{axis}_idx", torch.zeros(batch_size, dtype=torch.long, device=device)
        )
    for axis in CONTINUOUS_AXES:
        condition.setdefault(axis, torch.zeros(batch_size, 1, device=device))
        condition.setdefault(f"{axis}_mask", torch.zeros(batch_size, 1, device=device))

    return condition


def length_grid(
    request: Dict,
    counts: Dict[int, int],
    species_vocab: Dict[str, int],
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    One condition covering several requested lengths.

    Length is the axis most worth sweeping rather than fixing: the decoder
    enforces it as a hard constraint, so pinning a single value collapses
    the library onto one length band and throws away the motif families
    that only work at other lengths.
    """
    blocks = []
    for length, count in counts.items():
        if count <= 0:
            continue
        local = dict(request)
        local["length"] = int(np.clip(length, MIN_PEPTIDE_LENGTH, MAX_PEPTIDE_LENGTH))
        blocks.append(build_condition(local, count, species_vocab, device=device))

    if not blocks:
        raise ValueError("length_grid received no positive counts")

    keys = blocks[0].keys()
    return {key: torch.cat([block[key] for block in blocks], dim=0) for key in keys}


@torch.no_grad()
def generate_candidates(
    diffusion_model,
    decoder,
    condition: Dict[str, torch.Tensor],
    guidance_weight: float = 3.0,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    temperature: float = 0.8,
    top_k: int = 5,
    sample_batch_size: int = 512,
    decode_batch_size: int = 256,
    seed: Optional[int] = None,
) -> List[str]:
    """Sample latents under the condition, then decode them."""
    device = next(diffusion_model.parameters()).device
    total = condition[f"{CATEGORICAL_AXES[0]}_idx"].shape[0]

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    sequences: List[str] = []
    for start in range(0, total, sample_batch_size):
        stop = min(start + sample_batch_size, total)
        chunk = {key: value[start:stop].to(device) for key, value in condition.items()}
        embeddings = diffusion_model.sample(
            batch_size=stop - start,
            condition=chunk,
            guidance_weight=guidance_weight,
            num_inference_steps=num_inference_steps,
            eta=eta,
            generator=generator,
        )
        sequences.extend(
            decoder.decode(
                embeddings,
                chunk,
                temperature=temperature,
                top_k=top_k,
                batch_size=decode_batch_size,
            )
        )
    return sequences


def candidate_frame(sequences: Iterable[str]) -> pd.DataFrame:
    """Candidate table with the dense properties attached, deduplicated."""
    unique = list(dict.fromkeys(str(s).upper().strip() for s in sequences if s))
    frame = pd.DataFrame({"sequence": unique})
    if frame.empty:
        return frame
    features = compute_sequence_features(frame["sequence"])
    for column in features.columns:
        frame[column] = features[column].values
    return frame



def sample_library(
    model,
    condition,
    guidance_weight: float,
    steps: int,
    temperature: float,
    batch_size: int,
    seed: int | None,
    free_length: bool = False,
    reveal: str = "uniform",
    pad_bias: float = 0.0,
) -> list:
    """Batched sampling from a masked-diffusion checkpoint.

    Lives here rather than in `scripts/generate_masked.py` because the
    competition entry point needs it: `uv` installs only `src/ampx`, so
    anything under `scripts/` is absent from an installed environment and
    importing it from `ampx.generate` would work on the dev box and fail on a
    clean clone. The script imports it back from here, so there is one
    implementation.

    `seed` seeds a dedicated `torch.Generator` on the sampling device rather
    than the global RNG, so sampling is reproducible independently of whatever
    else has drawn from the global stream.
    """
    device = next(model.parameters()).device
    total = condition["species_idx"].shape[0]

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    sequences = []
    for start in tqdm(range(0, total, batch_size), desc="sampling"):
        stop = min(start + batch_size, total)
        chunk = {key: value[start:stop].to(device) for key, value in condition.items()}
        sequences.extend(
            model.sample(
                batch_size=stop - start,
                condition=chunk,
                guidance_weight=guidance_weight,
                num_steps=steps,
                temperature=temperature,
                generator=generator,
                free_length=free_length,
                reveal=reveal,
                pad_bias=pad_bias,
            )
        )
    return sequences

__all__ = [
    "sample_library",
    "REQUEST_KEYS",
    "build_condition",
    "candidate_frame",
    "generate_candidates",
    "length_grid",
]

def empirical_length_counts(
    lengths,
    total: int,
    min_length: int = MIN_PEPTIDE_LENGTH,
    max_length: int = MAX_PEPTIDE_LENGTH,
) -> Dict[int, int]:
    """
    Length allocation matched to an observed distribution.

    A uniform grid over 8 to 50 is right for a candidate library, where
    breadth across length bands is the point. It is wrong for a
    distribution-match benchmark: real AMPs have a median length of 18 and
    77 percent fall in 10 to 30, so uniform sampling disqualifies half the
    output on length alone and depresses every composite property metric
    regardless of model quality.
    """
    observed = np.asarray(list(lengths), dtype=int)
    observed = observed[(observed >= min_length) & (observed <= max_length)]
    if observed.size == 0:
        raise ValueError("no lengths inside the competition window")

    values, counts = np.unique(observed, return_counts=True)
    share = counts / counts.sum()
    allocation = np.floor(share * total).astype(int)

    remainder = total - int(allocation.sum())
    if remainder > 0:
        order = np.argsort(-(share * total - allocation))
        for index in order[:remainder]:
            allocation[index] += 1

    return {int(v): int(c) for v, c in zip(values, allocation) if c > 0}
