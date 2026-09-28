"""
Pooled ESM-2 embedding, matching the anticancer pipeline exactly.

The mask-weighted mean over the last hidden state of esm2_t12_35M_UR50D
gives the 480-dimensional latent that the diffusion model works in. The
pooling is over real tokens only, so padding does not dilute short
peptides.

This is a lossy bottleneck: one 480-vector for up to fifty residues, which
is why the decoder has to be conditioned rather than left to reconstruct
from the latent alone. It is kept unchanged here because it is the
representation the March-10 model was validated on; replacing it with a
per-residue latent is a separate change to the latent space, not to the
conditioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

DEFAULT_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"

#: Pinned commit of the Hugging Face repo above. An unpinned pull resolves to
#: whatever `main` points at on the day the organizers run us, which would make
#: their embeddings -- and therefore their ranking -- diverge from ours no
#: matter how carefully everything else is seeded. Verified against the stored
#: training embeddings: re-encoding sequences from the Sep 13 run at this
#: revision reproduces `mic_esm.npy` to 2.4e-06, i.e. float32 noise.
DEFAULT_ESM_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"


class ESM2Encoder:
    def __init__(
        self,
        model_path: str = DEFAULT_ESM_MODEL,
        device: str = "cuda",
        max_length: int = 64,
        revision: str | None = DEFAULT_ESM_REVISION,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.revision = revision
        # `revision` is only meaningful for a Hub id; a local directory has no
        # commits, so it is dropped rather than raising for that case.
        hub = {} if Path(model_path).exists() else {"revision": revision}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **hub)
        # add_pooling_layer=False: pooling here is the mask-weighted mean over
        # last_hidden_state below, so EsmModel's own pooler is never read. It
        # is also absent from these weights, so constructing it draws from
        # torch's global RNG at load time -- which would shift every downstream
        # sample if a seed were set before the encoder was built.
        self.model = AutoModel.from_pretrained(
            model_path, add_pooling_layer=False, **hub
        ).to(device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, sequences: Iterable[str], batch_size: int = 128) -> np.ndarray:
        sequences = list(sequences)
        outputs: List[np.ndarray] = []

        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            spaced = [" ".join(list(str(s).upper())) for s in chunk]
            encoded = self.tokenizer(
                spaced,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            hidden = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

            expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * expanded, dim=1)
            counts = torch.clamp(expanded.sum(1), min=1e-9)
            pooled = summed / counts
            outputs.append(pooled.float().cpu().numpy())

        if not outputs:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_one(self, sequence: str) -> np.ndarray:
        return self.encode([sequence])[0]


__all__ = ["ESM2Encoder", "DEFAULT_ESM_MODEL"]
