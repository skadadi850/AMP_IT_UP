"""Calibrated activity and MIC surrogates. See docs/PLAN.md 2.2-2.3.

The MIC regressor predicts log10 MIC (uM) for a (sequence, species) pair.
Training lives in `scripts/train_predictor.py`; everything needed to *use* a
trained model lives here, so inference does not depend on the training script
being importable.

`build_features` is shared by both sides deliberately. A feature matrix
assembled one way at training time and another way at inference is the classic
silent skew, so there is exactly one implementation and the training script
imports it from here rather than defining its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models.encoder import DEFAULT_ESM_MODEL, ESM2Encoder
from .models.features import compute_sequence_features, gram_for_species

#: Descriptor columns, in the order the regressor was trained on. Order is part
#: of the model contract: XGBoost indexes columns positionally, so reordering
#: this list silently invalidates every serialized checkpoint.
DESCRIPTORS = [
    "length", "length_norm", "net_charge", "charge_per_res",
    "mu_h", "gravy_raw", "gravy", "cys_count", "cys_class_idx",
]

#: Gram levels, one-hot in this order. Same positional contract as DESCRIPTORS.
GRAM_LEVELS = ["negative", "positive", "fungal", "mycobacterial", "unknown"]


def build_features(table: pd.DataFrame, embeddings, species_levels):
    """Assemble the regressor's feature matrix.

    Blocks, in order: descriptors, one-hot species (plus an "other" catch-all
    for anything outside `species_levels`), one-hot gram, then the ESM-2
    embedding if one is supplied. Returns `(matrix, column_names)`.

    `table` needs the DESCRIPTORS columns plus `species` and `gram`; when
    `embeddings` is given it also needs `row`, indexing into that array.
    """
    blocks = [table[DESCRIPTORS].to_numpy(dtype=np.float32)]
    names = list(DESCRIPTORS)

    species = table["species"].where(table["species"].isin(species_levels), "other")
    for level in list(species_levels) + ["other"]:
        blocks.append((species == level).to_numpy(dtype=np.float32).reshape(-1, 1))
        names.append(f"species={level}")

    gram = table["gram"].fillna("unknown")
    for level in GRAM_LEVELS:
        blocks.append((gram == level).to_numpy(dtype=np.float32).reshape(-1, 1))
        names.append(f"gram={level}")

    if embeddings is not None:
        rows = table["row"].to_numpy(dtype=int)
        blocks.append(embeddings[rows].astype(np.float32))
        names.extend(f"esm{i}" for i in range(embeddings.shape[1]))

    return np.hstack(blocks), names


def resolve_device(device: str | None = None) -> str:
    """Pick a torch device. `None` or "auto" resolves to CUDA when available.

    Callers that need bit-reproducible output should pass "cpu" explicitly:
    CPU and GPU kernels do not agree to the last bit, and a ranking built on
    near-tied scores can reorder between them.
    """
    if device not in (None, "auto"):
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Oracle:
    """MIC regressor plus the ESM-2 encoder it needs, held together.

    The encoder is only built when the checkpoint's metadata says the model was
    trained with embeddings, so a descriptor-only regressor does not pay for
    loading ESM-2.
    """

    def __init__(self, model_path, meta_path, device: str | None = None):
        import xgboost as xgb
        self.meta = json.loads(Path(meta_path).read_text())
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.encoder = None
        if self.meta["uses_embeddings"]:
            # The embedding block must come from the same ESM-2 the regressor
            # was trained against. Checkpoints written before `esm_model` was
            # recorded fall back to the default, which is what they used.
            self.encoder = ESM2Encoder(
                model_path=self.meta.get("esm_model", DEFAULT_ESM_MODEL),
                device=resolve_device(device),
            )
            # Width is the part XGBoost cannot check for us: a wrong-but-equal
            # width would be accepted silently and predict nonsense.
            expected = self.meta.get("embedding_dim")
            if expected and self.encoder.embedding_dim != expected:
                raise ValueError(
                    f"{self.meta.get('esm_model', DEFAULT_ESM_MODEL)} produces "
                    f"{self.encoder.embedding_dim}-dim embeddings but the "
                    f"regressor was trained on {expected}. The checkpoint and "
                    "the encoder disagree; re-train or point at the right model."
                )

    def predict(self, sequences, species):
        """Predicted log10 MIC (uM). Lower is more potent."""
        if not len(sequences):
            return np.zeros(0, dtype=np.float32)
        frame = pd.DataFrame({"sequence": list(sequences), "species": list(species)})
        features_frame = compute_sequence_features(frame["sequence"])
        for column in features_frame.columns:
            frame[column] = features_frame[column].values
        frame["gram"] = frame["species"].map(gram_for_species)
        embeddings = None
        if self.encoder is not None:
            embeddings = self.encoder.encode(frame["sequence"].tolist(), batch_size=256)
            frame["row"] = np.arange(len(frame))
        features, _ = build_features(frame, embeddings, self.meta["species_levels"])
        return self.model.predict(features)
