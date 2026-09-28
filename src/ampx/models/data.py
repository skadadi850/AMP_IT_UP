"""
Datasets over precomputed pooled embeddings.

Both datasets emit the conditioning axes under their canonical keys, so a
dataloader batch can be handed to condition_from_batch with no renaming.
Sparse axes come out with a mask of zero rather than an imputed value; the
diffusion model then applies its own per-sample dropout on top, and the two
compose: an axis that is missing in the data stays missing, and an axis
that is present is sometimes hidden so the model learns to work without it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .conditioning import CATEGORICAL_AXES, CONTINUOUS_AXES
from .features import species_index

# Metadata column for each axis. Column names are the axis names, except
# species which is indexed through the vocabulary at load time.
CATEGORICAL_COLUMNS = {axis: f"{axis}_idx" for axis in CATEGORICAL_AXES}
CONTINUOUS_COLUMNS = {axis: axis for axis in CONTINUOUS_AXES}


def prepare_metadata(
    metadata: pd.DataFrame,
    species_vocab: Dict[str, int],
    continuous_axes=None,
) -> pd.DataFrame:
    """
    Add species_idx and check every axis column is present.

    continuous_axes lets a caller train against an older axis list than the
    module constant, so a metadata table built before an axis existed is not
    rejected for lacking its column.
    """
    frame = metadata.copy()
    frame["species_idx"] = species_index(frame, species_vocab)
    continuous_columns = (
        CONTINUOUS_COLUMNS if continuous_axes is None
        else {axis: axis for axis in continuous_axes}
    )

    for axis, column in CATEGORICAL_COLUMNS.items():
        if column not in frame.columns:
            raise KeyError(f"metadata is missing categorical column {column!r} for axis {axis!r}")
        frame[column] = frame[column].fillna(0).astype(int)

    for axis, column in continuous_columns.items():
        if column not in frame.columns:
            raise KeyError(f"metadata is missing continuous column {column!r} for axis {axis!r}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.reset_index(drop=True)


def condition_arrays(metadata: pd.DataFrame, continuous_axes=None) -> Dict[str, np.ndarray]:
    """Materialize the whole condition as dense arrays, once."""
    arrays: Dict[str, np.ndarray] = {}
    continuous_columns = (
        CONTINUOUS_COLUMNS if continuous_axes is None
        else {axis: axis for axis in continuous_axes}
    )
    for axis, column in CATEGORICAL_COLUMNS.items():
        arrays[f"{axis}_idx"] = metadata[column].to_numpy(dtype=np.int64)
    for axis, column in continuous_columns.items():
        values = metadata[column].to_numpy(dtype=np.float64)
        observed = np.isfinite(values)
        arrays[axis] = np.nan_to_num(values, nan=0.0).astype(np.float32).reshape(-1, 1)
        arrays[f"{axis}_mask"] = observed.astype(np.float32).reshape(-1, 1)
    return arrays


class AMPDataset(Dataset):
    """Embedding plus condition, for diffusion training."""

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        species_vocab: Dict[str, int],
        continuous_axes=None,
    ):
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch"
            )
        self.embeddings = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
        # See TokenDataset: __getitem__ must iterate the same axis list
        # condition_arrays built, not the module global.
        self.continuous_axes = tuple(
            CONTINUOUS_AXES if continuous_axes is None else continuous_axes
        )
        self.metadata = prepare_metadata(metadata, species_vocab, continuous_axes)
        self.condition = condition_arrays(self.metadata, continuous_axes)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {"embedding": self.embeddings[index]}
        for axis in CATEGORICAL_AXES:
            key = f"{axis}_idx"
            item[key] = torch.tensor(self.condition[key][index], dtype=torch.long)
        for axis in self.continuous_axes:
            item[axis] = torch.from_numpy(self.condition[axis][index])
            item[f"{axis}_mask"] = torch.from_numpy(self.condition[f"{axis}_mask"][index])
        return item


class DecoderDataset(AMPDataset):
    """Embedding plus condition plus teacher-forcing tokens."""

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        species_vocab: Dict[str, int],
        decoder,
        sequence_column: str = "sequence",
    ):
        super().__init__(embeddings, metadata, species_vocab)
        token_length = decoder.max_peptide_length + 2
        self.tokens = torch.tensor(
            [
                decoder.encode_tokens(str(sequence), max_length=token_length)
                for sequence in self.metadata[sequence_column]
            ],
            dtype=torch.long,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        item["tokens"] = self.tokens[index]
        return item


def cluster_split(
    metadata: pd.DataFrame,
    splits: Optional[pd.DataFrame] = None,
    sequence_column: str = "sequence",
    cluster_column: str = "cluster",
    fold_column: str = "fold",
    holdout_fold: int = 0,
    random_seed: int = 42,
) -> tuple:
    """
    Train and validation indices.

    If a splits frame is supplied it is joined on sequence and the given
    fold is held out, which keeps the cluster-disjoint property from the
    MMseqs2 clustering already in the project. A random split leaks
    homologues across the boundary and inflates validation loss agreement,
    so it is only the fallback when no splits file exists.
    """
    frame = metadata.reset_index(drop=True)

    if splits is not None and sequence_column in splits.columns:
        keyed = frame[[sequence_column]].merge(
            splits[[sequence_column, fold_column]].drop_duplicates(subset=[sequence_column]),
            on=sequence_column,
            how="left",
        )
        folds = keyed[fold_column]
        if folds.notna().any():
            # Sequences with no assigned fold stay in training rather than
            # being silently dropped.
            is_val = (folds == holdout_fold).fillna(False).to_numpy()
            val_indices = np.where(is_val)[0]
            train_indices = np.where(~is_val)[0]
            return train_indices, val_indices

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(frame))
    cut = int(0.9 * len(order))
    return order[:cut], order[cut:]


def stratified_sampler_weights(metadata: pd.DataFrame) -> np.ndarray:
    """
    Sampling weights that stop the unannotated majority from drowning out
    the MIC-annotated rows.

    Weight is inverse to the size of the (has-MIC, species) stratum, so a
    species with forty measurements is seen about as often as one with four
    hundred and the potency axis gets gradient from all of them.
    """
    has_mic = metadata["log_mic"].notna()
    stratum = np.where(
        has_mic,
        "mic:" + metadata["species_idx"].astype(str),
        "nomic",
    )
    counts = pd.Series(stratum).value_counts()
    return (1.0 / counts.reindex(stratum).to_numpy()).astype(np.float64)


def is_amp_sampler_weights(
    metadata: pd.DataFrame,
    negative_ratio: float = 3.0,
    positive_ratio: float = 1.0,
) -> np.ndarray:
    """
    Sampling weights that draw general peptides (is_amp_idx == 1) and
    validated AMPs (is_amp_idx == 2) in roughly negative_ratio:positive_ratio,
    not the raw corpus ratio -- pretraining mixes ~300k general peptides
    with ~39k AMPs, and sampling at the raw ratio would swamp the AMP signal
    the is_amp axis is supposed to teach.
    """
    is_amp = metadata["is_amp_idx"].to_numpy()
    weights = np.zeros(len(metadata), dtype=np.float64)
    n_pos = int((is_amp == 2).sum())
    n_neg = int((is_amp == 1).sum())
    if n_pos > 0:
        weights[is_amp == 2] = positive_ratio / n_pos
    if n_neg > 0:
        weights[is_amp == 1] = negative_ratio / n_neg
    return weights


__all__ = [
    "AMPDataset",
    "DecoderDataset",
    "cluster_split",
    "condition_arrays",
    "is_amp_sampler_weights",
    "prepare_metadata",
    "stratified_sampler_weights",
]


class TokenDataset(Dataset):
    """
    Amino acid tokens plus condition, for masked discrete diffusion.

    No embeddings: the model reads sequences directly, so the ESM-2
    encoding step and embeddings.npy drop out of the pipeline.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        species_vocab: Dict[str, int],
        sequence_column: str = "sequence",
        canvas: int = None,
        continuous_axes=None,
    ):
        from .masked_diffusion import encode_sequence
        from .features import MAX_PEPTIDE_LENGTH

        self.canvas = int(canvas or MAX_PEPTIDE_LENGTH)
        # Held on the instance because __getitem__ must iterate the same axis
        # list condition_arrays built. Reading the module global there instead
        # raises KeyError for any axis added after this checkpoint's metadata
        # was written.
        self.continuous_axes = tuple(
            CONTINUOUS_AXES if continuous_axes is None else continuous_axes
        )
        self.metadata = prepare_metadata(metadata, species_vocab, continuous_axes)
        self.condition = condition_arrays(self.metadata, continuous_axes)
        self.tokens = torch.tensor(
            [
                encode_sequence(str(sequence), canvas=self.canvas)
                for sequence in self.metadata[sequence_column]
            ],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {"tokens": self.tokens[index]}
        for axis in CATEGORICAL_AXES:
            key = f"{axis}_idx"
            item[key] = torch.tensor(self.condition[key][index], dtype=torch.long)
        for axis in self.continuous_axes:
            item[axis] = torch.from_numpy(self.condition[axis][index])
            item[f"{axis}_mask"] = torch.from_numpy(self.condition[f"{axis}_mask"][index])
        return item