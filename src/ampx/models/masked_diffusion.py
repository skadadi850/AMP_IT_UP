"""
Masked discrete diffusion over amino acid sequences.

The modeloperates directly on tokens over a fixed 50-position canvas.

Forward process. Each position is independently replaced by a MASK token
with probability 1 - alpha_t, using the linear schedule alpha_t = 1 - t for
t in (0, 1]. At t = 1 the canvas is fully masked; at t = 0 it is the data.

Objective. The continuous-time masked-diffusion bound reduces to a
reweighted cross entropy over masked positions only, with weight 1/t. No
noise prediction, no continuous relaxation of a discrete variable.

Reverse process. Iterative unmasking. Moving from t to s < t, each still
masked position is committed with probability (t - s) / t, sampled from the
model's posterior over amino acids. Committed positions are never remasked,
which makes the sampler a strict refinement and lets hard constraints be
enforced against partial sequences.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import (
    AMPConditionEncoder,
    CONTINUOUS_AXES,
    LEGACY_CONTINUOUS_AXES,
    concat_conditions,
    drop_condition_axes,
    normalize_condition,
)
from .features import (
    MAX_CYSTEINES,
    MAX_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    STANDARD_AA,
)

# Vocabulary: twenty residues, then PAD for positions past the peptide
# length, then MASK for the diffusion's absorbing state. PAD is a real
# prediction target, which is how the model learns to place the terminus.
PAD_TOKEN = len(STANDARD_AA)
MASK_TOKEN = len(STANDARD_AA) + 1
VOCAB_SIZE = len(STANDARD_AA) + 2

AA_TO_TOKEN = {aa: i for i, aa in enumerate(STANDARD_AA)}
TOKEN_TO_AA = {i: aa for i, aa in enumerate(STANDARD_AA)}
CYS_TOKEN = AA_TO_TOKEN["C"]


def encode_sequence(sequence: str, canvas: int = MAX_PEPTIDE_LENGTH) -> List[int]:
    """Residues then PAD out to the canvas width."""
    tokens = [AA_TO_TOKEN[a] for a in str(sequence).upper() if a in AA_TO_TOKEN]
    tokens = tokens[:canvas]
    return tokens + [PAD_TOKEN] * (canvas - len(tokens))


def decode_tokens(tokens) -> str:
    """Residues up to the first PAD; MASK is treated as an empty position."""
    out = []
    for token in tokens:
        token = int(token)
        if token == PAD_TOKEN:
            break
        if token == MASK_TOKEN:
            continue
        out.append(TOKEN_TO_AA[token])
    return "".join(out)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / (half - 1)
        frequencies = torch.exp(torch.arange(half, device=t.device) * -scale)
        angles = t.float()[:, None] * 1000.0 * frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class AdaLNTransformerBlock(nn.Module):
    """
    Bidirectional self-attention and feedforward, both adaLN-Zero modulated.

    Attention is over the 50 sequence positions, so unlike the pooled-latent
    denoiser it is doing real work: this is where ordering information lives.
    """

    def __init__(self, dim: int, num_heads: int, ff_dim: int, cond_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.modulation(c).unsqueeze(1).chunk(6, dim=-1)
        )
        h = self.norm1(x) * (1.0 + scale1) + shift1
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * attended
        h = self.norm2(x) * (1.0 + scale2) + shift2
        return x + gate2 * self.ff(h)


class MaskedDiffusionModel(nn.Module):
    def __init__(
        self,
        num_species: int,
        canvas: int = MAX_PEPTIDE_LENGTH,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        cond_embed_dim: int = 128,
        min_length: int = MIN_PEPTIDE_LENGTH,
        max_length: int = MAX_PEPTIDE_LENGTH,
        max_cysteines: int = MAX_CYSTEINES,
        mask_mode_weights=(0.5, 0.25, 0.25),
        continuous_axes=None,
        device: str = "cuda",
    ):
        super().__init__()

        if continuous_axes is None:
            continuous_axes = CONTINUOUS_AXES

        self.config = {
            "continuous_axes": tuple(continuous_axes),
            "num_species": int(num_species),
            "canvas": int(canvas),
            "d_model": int(d_model),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "ff_dim": int(ff_dim),
            "dropout": float(dropout),
            "cond_embed_dim": int(cond_embed_dim),
            "min_length": int(min_length),
            "max_length": int(max_length),
            "max_cysteines": int(max_cysteines),
            "mask_mode_weights": tuple(float(w) for w in mask_mode_weights),
        }

        self.canvas = int(canvas)
        self.d_model = int(d_model)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.max_cysteines = int(max_cysteines)
        self.mask_mode_weights = tuple(float(w) for w in mask_mode_weights)
        self.vocab_size = VOCAB_SIZE

        self.token_embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.canvas, self.d_model))
        nn.init.normal_(self.position_embedding, std=0.02)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(256),
            nn.Linear(256, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.condition_encoder = AMPConditionEncoder(
            num_species=num_species,
            embed_dim=cond_embed_dim,
            output_dim=self.d_model,
            continuous_axes=continuous_axes,
        )

        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(self.d_model, num_heads, ff_dim, self.d_model, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.d_model, 2 * self.d_model)
        )
        self.output_proj = nn.Linear(self.d_model, VOCAB_SIZE)

        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.device = device
        self.to(device)

    def logits(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        if condition is None:
            condition = self.condition_encoder.null_condition(batch_size, tokens.device)

        device_type = tokens.device.type

        # Conditioning path and embeddings stay in fp32.
        with torch.autocast(device_type=device_type, enabled=False):
            c = self.time_embedding(t) + self.condition_encoder(condition)
            x = self.token_embedding(tokens) + self.position_embedding

        # Blocks run under autocast. LayerNorm is promoted to fp32 by autocast
        # and the residual stream stays fp32 because bf16 branch outputs are
        # added into an fp32 tensor.
        with torch.autocast(
            device_type=device_type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            for block in self.blocks:
                x = block(x, c)

        # Output head in fp32 so guidance and sampling see full-precision logits.
        with torch.autocast(device_type=device_type, enabled=False):
            x = x.float()
            shift, scale = self.final_modulation(c).unsqueeze(1).chunk(2, dim=-1)
            x = self.final_norm(x) * (1.0 + scale) + shift
            out = self.output_proj(x)
        # MASK is an absorbing state, never a prediction target.
        out[..., MASK_TOKEN] = torch.finfo(out.dtype).min
        return out

    def sample_mask(
        self,
        t: torch.Tensor,
        canvas: int,
        mode_weights=None,
    ) -> torch.Tensor:
        """
        Corruption mask with three structures, mixed per sample.

        iid       independent per position, the standard masked-diffusion
                  process and the only one for which the 1/t-weighted loss
                  is the exact NELBO
        span      one contiguous block, SpanBERT-style corruption
        face      one helical face: positions whose angle (i * 100 degrees)
                  mod 360 falls inside a window. Amphipathic AMPs segregate
                  hydrophobic residues onto one helical face at roughly 100
                  degrees per residue, so this corruption forces the model to
                  reconstruct one membrane-facing surface from the
                  complementary one, which contiguous masking does not.

        Every mode holds the per-position marginal masking rate at t, so the
        diffusion schedule is untouched and only the joint changes.
        """
        if mode_weights is None:
            mode_weights = self.mask_mode_weights

        batch_size = t.shape[0]
        device = t.device
        positions = torch.arange(canvas, device=device).float()

        iid = torch.rand(batch_size, canvas, device=device) < t[:, None]

        width = (t * canvas).clamp(min=1.0)
        start = torch.rand(batch_size, device=device) * (canvas - width).clamp(min=0.0)
        span = (positions[None, :] >= start[:, None]) & (
            positions[None, :] < (start + width)[:, None]
        )

        angle = (positions * 100.0) % 360.0
        phase = torch.rand(batch_size, device=device) * 360.0
        window = t * 360.0
        offset = (angle[None, :] - phase[:, None]) % 360.0
        face = offset < window[:, None]

        choice = torch.multinomial(
            torch.tensor(mode_weights, device=device).expand(batch_size, 3),
            1,
        ).squeeze(1)
        return torch.where(
            (choice == 0)[:, None], iid,
            torch.where((choice == 1)[:, None], span, face),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        p_axis: float = 0.15,
        p_all: float = 0.10,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """
        Masked-diffusion negative log likelihood bound.

        Only masked positions contribute, weighted by 1/t. The eps floor
        keeps the weight finite as t approaches zero, where almost nothing
        is masked and the estimator would otherwise blow up.
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        t = torch.rand(batch_size, device=device) * (1.0 - eps) + eps
        masked = self.sample_mask(t, self.canvas)

        # A fully clean canvas gives no gradient, so force one masked
        # position per sample rather than wasting the example.
        empty = ~masked.any(dim=1)
        if bool(empty.any()):
            forced = torch.randint(0, self.canvas, (batch_size,), device=device)
            masked[empty, forced[empty]] = True

        corrupted = torch.where(masked, torch.full_like(tokens, MASK_TOKEN), tokens)

        condition = normalize_condition(condition, batch_size, device)
        if p_axis > 0.0 or p_all > 0.0:
            condition = drop_condition_axes(condition, p_axis=p_axis, p_all=p_all)

        logits = self.logits(corrupted, t, condition)
        token_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            reduction="none",
        ).reshape(batch_size, self.canvas)

        weight = (1.0 / t)[:, None]
        numerator = (token_loss * masked.float() * weight).sum()
        return numerator / masked.float().sum().clamp(min=1.0)

    def guided_logits(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]],
        guidance_weight: float,
        exempt_axes: Sequence[str] = (),
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Classifier-free guidance in logit space, one forward pass.

        The unconditional branch is the encoder's all-unknown state, which
        per-axis dropout trains on directly -- except for any axis exempted
        from dropout at training time. `null_condition` sets every categorical
        axis to index 0 ("unknown"), but an axis exempt from both p_axis and
        p_all dropout never takes index 0 in training, so its embedding row is
        never updated and the guidance denominator reads an untrained vector.
        Pass those axis names in `exempt_axes` to carry their requested value
        into the unconditional branch instead, so the guidance direction
        isolates the axes actually being varied.

        `logits` returns fp32 regardless of `autocast_dtype`, so the guidance
        combination below is always computed in fp32.
        """
        batch_size = tokens.shape[0]
        if condition is None or guidance_weight == 0.0:
            return self.logits(tokens, t, condition, autocast_dtype=autocast_dtype)

        null = self.condition_encoder.null_condition(batch_size, tokens.device)
        for axis in exempt_axes:
            key = f"{axis}_idx"
            if key in condition and key in null:
                null[key] = condition[key].long().reshape(-1).clone()
        # `build_condition` fills every axis in the current CONTINUOUS_AXES,
        # which can be a superset of what this checkpoint's encoder was built
        # with. `null_condition` comes from the encoder itself, so its key set
        # is authoritative; restrict the conditional branch to match before
        # concatenating, or the merge raises on an axis the encoder ignores.
        aligned = {key: condition[key] for key in null if key in condition}
        merged = concat_conditions(aligned, null)
        both = self.logits(
            torch.cat([tokens, tokens], dim=0),
            torch.cat([t, t], dim=0),
            merged,
            autocast_dtype=autocast_dtype,
        )
        cond, uncond = both[:batch_size], both[batch_size:]
        return uncond + guidance_weight * (cond - uncond)

    def constrain(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        target_length: torch.Tensor,
        pin_length: bool = True,
        pad_bias: float = 0.0,
    ) -> torch.Tensor:
        """
        Hard competition constraints applied to the whole canvas at once.

        With pin_length=True (the default, and the behaviour every result
        before this change was produced with) positions past the requested
        length are forced to PAD and positions inside it are forbidden from
        taking PAD, which pins length exactly.

        With pin_length=False the PAD column is left alone and the model
        places the terminus itself. PAD is a real prediction target during
        training, so the model has already learned where sequences end; the
        pin overrides that. Turning it off is what lets the sampler produce a
        LEARNED length distribution instead of one imposed by the caller,
        which is the only way this architecture can sample unconditionally.

        Cysteine is banned once a sample has already committed its quota, so
        no generated peptide can exceed the organizers' disulfide limit. That
        constraint applies in both modes.
        """
        neg_inf = torch.finfo(logits.dtype).min
        batch_size = logits.shape[0]

        # Length calibration. A constant added to the PAD logit shifts where the
        # model places the terminus: negative lengthens, positive shortens. It
        # is one scalar, applied at sampling time only, and it does not touch
        # the residue distribution at any position -- it only moves the stopping
        # point. Meaningless when pin_length is True, since PAD is then forced.
        if pad_bias != 0.0 and not pin_length:
            logits[..., PAD_TOKEN] = logits[..., PAD_TOKEN] + float(pad_bias)

        if pin_length:
            positions = torch.arange(self.canvas, device=logits.device)[None, :]
            inside = positions < target_length[:, None]

            logits[..., PAD_TOKEN] = torch.where(
                inside, torch.full_like(logits[..., PAD_TOKEN], neg_inf),
                logits[..., PAD_TOKEN]
            )
            outside = ~inside
            forced = torch.full_like(logits, neg_inf)
            forced[..., PAD_TOKEN] = 0.0
            logits = torch.where(outside[..., None], forced, logits)

        committed_cys = (tokens == CYS_TOKEN).sum(dim=1)
        exhausted = committed_cys >= self.max_cysteines
        if bool(exhausted.any()):
            column = logits[..., CYS_TOKEN]
            logits[..., CYS_TOKEN] = torch.where(
                exhausted[:, None], torch.full_like(column, neg_inf), column
            )
        return logits

    def reveal_priority(
        self,
        batch_size: int,
        device,
        generator: Optional[torch.Generator] = None,
        mode_weights=None,
    ) -> torch.Tensor:
        """
        Per-position priority governing the ORDER the reverse process unmasks.

        Higher priority stays masked longer. The still-masked set therefore
        keeps the shape of the corruption the model was trained on, which
        fixes the forward/reverse mismatch recorded in BLOCKERS.md B9: the
        default sampler unmasks uniformly at random, so for the span and face
        share of training samples the reverse process never sees the joint the
        forward process produced.

        iid    uniform priority, which reproduces the original uniform reveal
        span   priority falls with circular distance from a random anchor, so
               the masked remainder is one contiguous block
        face   priority falls with angular distance from a random phase at
               100 degrees per residue, so the masked remainder is one
               helical face
        """
        if mode_weights is None:
            mode_weights = self.mask_mode_weights

        positions = torch.arange(self.canvas, device=device).float()

        uniform = torch.rand(batch_size, self.canvas, device=device, generator=generator)

        anchor = torch.randint(
            0, self.canvas, (batch_size,), device=device, generator=generator
        ).float()
        gap = (positions[None, :] - anchor[:, None]).abs()
        gap = torch.minimum(gap, self.canvas - gap)
        span = 1.0 - gap / (0.5 * self.canvas)

        angle = (positions * 100.0) % 360.0
        phase = torch.rand(batch_size, device=device, generator=generator) * 360.0
        offset = (angle[None, :] - phase[:, None]).abs() % 360.0
        offset = torch.minimum(offset, 360.0 - offset)
        face = 1.0 - offset / 180.0

        choice = torch.multinomial(
            torch.tensor([float(w) for w in mode_weights], device=device)
            .expand(batch_size, 3),
            1,
            generator=generator,
        ).squeeze(1)
        return torch.where(
            (choice == 0)[:, None], uniform,
            torch.where((choice == 1)[:, None], span, face),
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        guidance_weight: float = 3.0,
        num_steps: int = 128,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        free_length: bool = False,
        reveal: str = "uniform",
        pad_bias: float = 0.0,
        guidance_exempt_axes: Sequence[str] = (),
        init_tokens: Optional[torch.Tensor] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> List[str]:
        """
        Iterative unmasking, from a fully masked canvas or a partial one.

        free_length=False and reveal="uniform" reproduce the original sampler
        exactly. free_length=True lets the model place PAD itself instead of
        having length pinned by the condition; reveal="structured" unmasks in
        an order matching the training corruption mixture.

        init_tokens turns this into an inpainting sampler: pass a
        (batch, canvas) tensor in which the positions to regenerate hold
        MASK_TOKEN and every other position holds the residue (or PAD) to
        keep. batch_size is then taken from the tensor. Pinned positions are
        never overwritten -- both reveal branches intersect with still_masked
        and the commit is a torch.where against it -- so the guarantee comes
        from the existing loop rather than from new bookkeeping.

        autocast_dtype runs the transformer blocks under autocast at that
        dtype on CUDA; the output head and guidance stay fp32. It is ignored
        on other devices.

        With reveal="structured" the set of positions committed at each step
        is a function of the priority ranking and the schedule alone, not of
        the model output. Steps that commit nothing are skipped without a
        forward pass, since their output would be discarded.
        """
        self.eval()
        device = self.position_embedding.device
        if device.type != "cuda":
            autocast_dtype = None

        if init_tokens is not None:
            tokens = init_tokens.to(device=device, dtype=torch.long).clone()
            if tokens.dim() != 2 or tokens.shape[1] != self.canvas:
                raise ValueError(
                    f"init_tokens must be (batch, {self.canvas}), got {tuple(tokens.shape)}"
                )
            batch_size = tokens.shape[0]
        else:
            tokens = torch.full(
                (batch_size, self.canvas), MASK_TOKEN, dtype=torch.long, device=device
            )

        if condition is not None:
            condition = normalize_condition(condition, batch_size, device)

        if init_tokens is not None and not free_length:
            # constrain() forces PAD outside target_length and bans it inside,
            # which fights a pinned scaffold unless the two agree. Take the
            # length from the scaffold's own PAD boundary rather than from the
            # condition, which describes the requested length, not this one.
            # A masked position is still a peptide position -- it is awaiting a
            # residue, not absent. Excluding MASK here truncated the peptide
            # whenever a C-terminal residue happened to be masked, which is
            # ~k/l of all seeds.
            occupied = tokens != PAD_TOKEN
            positions = torch.arange(self.canvas, device=device).expand(batch_size, -1)
            target_length = torch.where(
                occupied, positions + 1, torch.zeros_like(positions)
            ).amax(dim=1).clamp(min=1)
        else:
            target_length = self.target_lengths(condition, batch_size, device)

        # The schedule's t is the corruption level the model is told to expect.
        # A fully masked canvas starts at 1.0; a canvas that is only 30% masked
        # starts at 0.3, or the model is denoising against a distribution it
        # never saw in training.
        initial_masked = (tokens == MASK_TOKEN).sum(dim=1)
        max_masked = int(initial_masked.max().item())
        t_start = 1.0
        if init_tokens is not None:
            t_start = float(max_masked) / float(self.canvas)
            t_start = min(1.0, max(t_start, 1.0 / float(num_steps)))
        schedule = torch.linspace(t_start, 0.0, num_steps + 1, device=device)

        priority = None
        if reveal == "structured":
            priority = self.reveal_priority(batch_size, device, generator)
        elif reveal != "uniform":
            raise ValueError(f"reveal must be 'uniform' or 'structured', got {reveal!r}")

        for step in range(num_steps):
            t_now, t_next = schedule[step], schedule[step + 1]
            still_masked = tokens == MASK_TOKEN
            if not bool(still_masked.any()):
                break

            to_reveal = None
            if priority is not None:
                # Keep the k highest-priority positions masked, where k is the
                # count the schedule calls for at t_next. The masked remainder
                # therefore retains the span or face shape of this sample.
                # Fraction of the ORIGINALLY masked positions still to hold
                # back. Scaling by the full canvas instead would exceed the
                # masked count for most of the schedule when inpainting, so
                # reveal would stall and then commit everything at once.
                remaining = float(t_next) / max(t_start, 1e-6)
                keep = int(round(remaining * float(max_masked)))
                if keep <= 0 or t_next <= 0:
                    to_reveal = still_masked
                else:
                    masked_priority = torch.where(
                        still_masked, priority,
                        torch.full_like(priority, -float("inf")),
                    )
                    order = masked_priority.argsort(dim=1, descending=True)
                    rank = torch.empty_like(order)
                    rank.scatter_(
                        1, order,
                        torch.arange(self.canvas, device=device)
                        .expand(batch_size, self.canvas),
                    )
                    to_reveal = still_masked & (rank >= keep)
                if not bool(to_reveal.any()):
                    continue

            logits = self.guided_logits(
                tokens, t_now.expand(batch_size), condition, guidance_weight,
                exempt_axes=guidance_exempt_axes,
                autocast_dtype=autocast_dtype,
            )
            logits = self.constrain(
                logits, tokens, target_length,
                pin_length=not free_length, pad_bias=pad_bias,
            )

            probabilities = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            sampled = torch.multinomial(
                probabilities.reshape(-1, self.vocab_size), 1, generator=generator
            ).reshape(batch_size, self.canvas)

            if priority is None:
                # Commit probability for the linear schedule alpha_t = 1 - t.
                commit = float((t_now - t_next) / t_now.clamp(min=1e-6))
                draw = torch.rand(tokens.shape, device=device, generator=generator)
                to_reveal = still_masked & ((draw < commit) | (t_next <= 0))
            tokens = torch.where(to_reveal, sampled, tokens)

        return [decode_tokens(row.tolist()) for row in tokens]

    def target_lengths(
        self,
        condition: Optional[Dict[str, torch.Tensor]],
        batch_size: int,
        device,
    ) -> torch.Tensor:
        span = float(self.max_length - self.min_length)
        if condition is None or "length_norm" not in condition:
            value = torch.full((batch_size,), 0.5, device=device)
        else:
            value = condition["length_norm"].float().reshape(-1)
            mask = condition.get("length_norm_mask")
            if mask is not None:
                mask = mask.float().reshape(-1)
                value = value * mask + 0.5 * (1.0 - mask)
        lengths = (value * span + self.min_length).round().long()
        return lengths.clamp(self.min_length, self.max_length)

    def save(self, path: str) -> None:
        import os
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(target.suffix + ".tmp")
        torch.save({"model_state_dict": self.state_dict(), "config": self.config}, staging)
        os.replace(staging, target)

    @classmethod
    def load(cls, path: str, device: str = "cuda"):
        from ..weights import fetch_if_missing

        fetch_if_missing(path)
        checkpoint = torch.load(path, map_location=device)
        config = dict(checkpoint["config"])
        config["device"] = device
        # Checkpoints saved before the axis list was recorded predate
        # gram_selectivity, so load them with the axis list they were built
        # under rather than whatever CONTINUOUS_AXES says now.
        config.setdefault("continuous_axes", LEGACY_CONTINUOUS_AXES)
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


__all__ = [
    "AA_TO_TOKEN",
    "MASK_TOKEN",
    "MaskedDiffusionModel",
    "PAD_TOKEN",
    "TOKEN_TO_AA",
    "VOCAB_SIZE",
    "decode_tokens",
    "encode_sequence",
]