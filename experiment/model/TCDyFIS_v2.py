"""
TCDyFIS-v2: Learnable Temporal Aggregation + Mamba Temporal Modeling.

Key improvements over v1:
  1. Replace F.interpolate with learnable cross-attention temporal aggregation
     (Perceiver-style latent queries attend to raw sequences).
  2. Replace temporal conv / global attention stack with Mamba sequence modeling
     following the compact FISNet-style temporal backbone.
  3. Bidirectional cross-modal fusion (each modality can attend to others).
  4. Simplified dyadic head: concat + MLP instead of 6-route gating.
  5. Keep a compact parameter budget for the small-data regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


@dataclass
class TCDyFISv2Config:
    task: int = 1
    n_labels: int = 9

    video_dim: int = 235
    audio_wav2vec_dim: int = 768
    audio_librosa_dim: int = 93
    text_dim: int = 1024

    d_model: int = 192
    d_audio: int = 192
    n_heads: int = 4
    n_temporal_blocks: int = 2
    compressed_len: int = 64
    text_compressed_len: int = 32
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    head_hidden: int = 192
    dropout: float = 0.3
    share_encoders: bool = True
    dyadic_layers: int = 1

    use_visual: bool = True
    use_audio: bool = True
    use_text: bool = True

    label_smoothing: float = 0.0
    input_noise_std: float = 0.0


class LearnableTemporalAggregation(nn.Module):
    """Perceiver-style: learnable latent queries cross-attend to variable-length input."""

    def __init__(self, d_model: int, n_latents: int, n_heads: int, dropout: float):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x.size(0)
        q = self.norm_q(self.latents.expand(bsz, -1, -1))
        kv = self.norm_kv(x)
        key_pad = ~mask if mask is not None else None
        h, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_pad)
        h = q + h
        h = self.norm_ffn(h + self.ffn(h))
        n_latents = self.latents.size(1)
        out_mask = torch.ones(bsz, n_latents, dtype=torch.bool, device=x.device)
        return h, out_mask


class MambaBlock(nn.Module):
    """Pre-norm Mamba block for compressed temporal modeling."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class MambaEncoder(nn.Module):
    """Stacked Mamba encoder over compressed temporal tokens."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class ModalityEncoder(nn.Module):
    """Unified encoder: input_proj -> learnable aggregation -> Mamba temporal modeling."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        compressed_len: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.aggregation = LearnableTemporalAggregation(
            d_model, compressed_len, n_heads, dropout,
        )
        self.temporal_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_blocks,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h, comp_mask = self.aggregation(h, mask)
        h = self.temporal_encoder(h)
        return self.norm(h), comp_mask


class AudioEncoder(nn.Module):
    def __init__(self, cfg: TCDyFISv2Config):
        super().__init__()
        self.wav_proj = nn.Linear(cfg.audio_wav2vec_dim, cfg.d_audio)
        self.librosa_proj = nn.Linear(cfg.audio_librosa_dim, cfg.d_audio)
        self.encoder = ModalityEncoder(
            d_in=2 * cfg.d_audio,
            d_model=cfg.d_model,
            compressed_len=cfg.compressed_len,
            n_blocks=cfg.n_temporal_blocks,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            mamba_d_state=cfg.mamba_d_state,
            mamba_d_conv=cfg.mamba_d_conv,
            mamba_expand=cfg.mamba_expand,
        )

    def forward(
        self,
        wav2vec_feat: torch.Tensor,
        librosa_feat: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if librosa_feat is None:
            librosa_feat = wav2vec_feat.new_zeros(
                wav2vec_feat.size(0), wav2vec_feat.size(1), self.librosa_proj.in_features,
            )
        fused = torch.cat([self.wav_proj(wav2vec_feat), self.librosa_proj(librosa_feat)], dim=-1)
        return self.encoder(fused, mask)


class AttentivePooling(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class BidirectionalCrossModalFusion(nn.Module):
    """Each modality attends to all others bidirectionally, then gated merge."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        modalities: list[tuple[str, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not modalities:
            raise ValueError("At least one modality required")
        if len(modalities) == 1:
            _, seq, mask = modalities[0]
            return self.out_norm(seq), mask

        all_seqs = [seq for _, seq, _ in modalities]
        all_masks = [mask for _, _, mask in modalities]

        concat_seq = torch.cat(all_seqs, dim=1)
        concat_mask = torch.cat(all_masks, dim=1)

        updated: list[torch.Tensor] = []
        for seq, mask in zip(all_seqs, all_masks):
            attn_out, _ = self.cross_attn(
                seq, concat_seq, concat_seq,
                key_padding_mask=~concat_mask,
            )
            g = torch.sigmoid(self.gate(torch.cat([seq, attn_out], dim=-1)))
            fused = self.norm(seq + g * attn_out)
            updated.append(fused)

        primary = updated[0]
        primary_mask = all_masks[0]
        for other, other_mask in zip(updated[1:], all_masks[1:]):
            if other.size(1) != primary.size(1):
                other_pooled = (other * other_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True)
                other_pooled = other_pooled / other_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
                primary = primary + other_pooled.expand_as(primary)
            else:
                primary = primary + other

        return self.out_norm(primary), primary_mask


class SimpleDyadicBlock(nn.Module):
    """Lightweight cross-role interaction."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        query_seq: torch.Tensor,
        context_seq: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query_seq, context_seq, context_seq,
            key_padding_mask=~context_mask,
        )
        h = self.norm(query_seq + attn_out)
        h = self.norm_ffn(h + self.ffn(h))
        return h * query_mask.unsqueeze(-1).to(h.dtype)


class TCDyFISv2(nn.Module):
    def __init__(self, cfg: TCDyFISv2Config):
        super().__init__()
        self.cfg = cfg

        if not any([cfg.use_visual, cfg.use_audio, cfg.use_text]):
            raise ValueError("At least one modality must be enabled")

        self.visual_encoder = (
            ModalityEncoder(
                d_in=cfg.video_dim, d_model=cfg.d_model,
                compressed_len=cfg.compressed_len,
                n_blocks=cfg.n_temporal_blocks, n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                mamba_d_state=cfg.mamba_d_state,
                mamba_d_conv=cfg.mamba_d_conv,
                mamba_expand=cfg.mamba_expand,
            )
            if cfg.use_visual else None
        )
        self.audio_encoder = AudioEncoder(cfg) if cfg.use_audio else None
        self.text_encoder = (
            ModalityEncoder(
                d_in=cfg.text_dim, d_model=cfg.d_model,
                compressed_len=cfg.text_compressed_len,
                n_blocks=max(1, cfg.n_temporal_blocks - 1), n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                mamba_d_state=cfg.mamba_d_state,
                mamba_d_conv=cfg.mamba_d_conv,
                mamba_expand=cfg.mamba_expand,
            )
            if cfg.use_text else None
        )

        if cfg.task == 2 and not cfg.share_encoders:
            self.patient_visual_encoder = (
                ModalityEncoder(
                    d_in=cfg.video_dim, d_model=cfg.d_model,
                    compressed_len=cfg.compressed_len,
                    n_blocks=cfg.n_temporal_blocks, n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    mamba_d_state=cfg.mamba_d_state,
                    mamba_d_conv=cfg.mamba_d_conv,
                    mamba_expand=cfg.mamba_expand,
                )
                if cfg.use_visual else None
            )
            self.patient_audio_encoder = AudioEncoder(cfg) if cfg.use_audio else None
            self.patient_text_encoder = (
                ModalityEncoder(
                    d_in=cfg.text_dim, d_model=cfg.d_model,
                    compressed_len=cfg.text_compressed_len,
                    n_blocks=max(1, cfg.n_temporal_blocks - 1), n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    mamba_d_state=cfg.mamba_d_state,
                    mamba_d_conv=cfg.mamba_d_conv,
                    mamba_expand=cfg.mamba_expand,
                )
                if cfg.use_text else None
            )
        else:
            self.patient_visual_encoder = self.visual_encoder
            self.patient_audio_encoder = self.audio_encoder
            self.patient_text_encoder = self.text_encoder

        self.fusion = BidirectionalCrossModalFusion(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.pool = AttentivePooling(cfg.d_model, d_hidden=max(32, cfg.d_model // 4))

        if cfg.task == 1:
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_hidden, cfg.n_labels),
            )
        else:
            self.dyadic_blocks_c = nn.ModuleList([
                SimpleDyadicBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
                for _ in range(max(1, cfg.dyadic_layers))
            ])
            self.dyadic_blocks_p = nn.ModuleList([
                SimpleDyadicBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
                for _ in range(max(1, cfg.dyadic_layers))
            ])
            self.head = nn.Sequential(
                nn.LayerNorm(3 * cfg.d_model),
                nn.Linear(3 * cfg.d_model, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_hidden, cfg.n_labels),
            )

    def _encode_role(
        self,
        role_data: dict[str, Any],
        seq_mask: torch.Tensor,
        tok_mask: torch.Tensor | None,
        visual_encoder: nn.Module | None,
        audio_encoder: nn.Module | None,
        text_encoder: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modalities: list[tuple[str, torch.Tensor, torch.Tensor]] = []

        if visual_encoder is not None and role_data.get("video_openface3") is not None:
            feat = role_data["video_openface3"]
            if self.training and self.cfg.input_noise_std > 0:
                feat = feat + torch.randn_like(feat) * self.cfg.input_noise_std
            visual_seq, visual_mask = visual_encoder(feat, seq_mask)
            modalities.append(("visual", visual_seq, visual_mask))

        if audio_encoder is not None and role_data.get("audio_wav2vec") is not None:
            audio_seq, audio_mask = audio_encoder(
                role_data["audio_wav2vec"],
                role_data.get("audio_librosa"),
                seq_mask,
            )
            modalities.append(("audio", audio_seq, audio_mask))

        if text_encoder is not None and role_data.get("text_embedding") is not None:
            text_seq, text_mask = text_encoder(role_data["text_embedding"], tok_mask)
            modalities.append(("text", text_seq, text_mask))

        if not modalities:
            raise ValueError("No valid modalities for current role")

        fused_seq, fused_mask = self.fusion(modalities)
        fused_global = self.pool(fused_seq, fused_mask)
        return fused_seq, fused_mask, fused_global

    def _forward_task1(self, batch: dict[str, Any]) -> torch.Tensor:
        _, _, role_repr = self._encode_role(
            batch["counselor"],
            batch["counselor_word_mask"],
            batch.get("counselor_tok_mask"),
            self.visual_encoder,
            self.audio_encoder,
            self.text_encoder,
        )
        return self.head(role_repr)

    def _forward_task2(self, batch: dict[str, Any]) -> torch.Tensor:
        c_seq, c_mask, c_global = self._encode_role(
            batch["counselor"],
            batch["counselor_word_mask"],
            batch.get("counselor_tok_mask"),
            self.visual_encoder,
            self.audio_encoder,
            self.text_encoder,
        )
        p_seq, p_mask, p_global = self._encode_role(
            batch["patient"],
            batch["patient_word_mask"],
            batch.get("patient_tok_mask"),
            self.patient_visual_encoder,
            self.patient_audio_encoder,
            self.patient_text_encoder,
        )

        c_ctx, p_ctx = c_seq, p_seq
        for c_block, p_block in zip(self.dyadic_blocks_c, self.dyadic_blocks_p):
            c_ctx = c_block(c_ctx, p_ctx, c_mask, p_mask)
            p_ctx = p_block(p_ctx, c_ctx, p_mask, c_mask)

        c_interact = self.pool(c_ctx, c_mask)
        p_interact = self.pool(p_ctx, p_mask)

        diff = c_interact - p_interact
        fused = torch.cat([c_interact, p_interact, diff], dim=-1)
        return self.head(fused)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.cfg.task == 2:
            return self._forward_task2(batch)
        return self._forward_task1(batch)
