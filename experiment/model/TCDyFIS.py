"""
TCDyFIS: Temporal Compression and Dynamic Fusion for raw FIS features.

This model is designed for raw audio/video/text inputs where sequence lengths can
be much longer than aligned word-level representations. It keeps the overall
multi-modal / task1-task2 design spirit of FISNet, but moves temporal reduction
before cross-modal fusion:

1. Compress raw modality sequences to short latent timelines.
2. Encode each compressed sequence with lightweight temporal conv blocks.
3. Fuse modalities with compressed cross attention and dynamic route gating.
4. For task2, build dyadic features from counselor/patient global embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TCDyFISConfig:
    task: int = 1
    n_labels: int = 9

    video_dim: int = 235
    audio_wav2vec_dim: int = 768
    audio_librosa_dim: int = 93
    text_dim: int = 1024

    d_model: int = 256
    d_audio: int = 256
    n_heads: int = 4
    conv_kernel: int = 5
    n_temporal_blocks: int = 3
    compressed_len: int = 96
    text_compressed_len: int = 48

    head_hidden: int = 256
    dropout: float = 0.2
    share_encoders: bool = False
    dyadic_layers: int = 1

    use_visual: bool = True
    use_audio: bool = True
    use_text: bool = True


def _lengths_from_mask(mask: torch.Tensor | None, default_len: int) -> list[int]:
    if mask is None:
        return [default_len]
    return mask.long().sum(dim=1).tolist()


def _compress_sequence(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress valid prefix of each sample to a short latent timeline."""
    bsz, _, feat_dim = x.shape
    out = x.new_zeros(bsz, target_len, feat_dim)
    out_mask = torch.zeros(bsz, target_len, dtype=torch.bool, device=x.device)

    if target_len <= 0:
        raise ValueError("target_len must be positive")

    lengths = _lengths_from_mask(mask, x.size(1)) if mask is not None else [x.size(1)] * bsz
    for idx, raw_len in enumerate(lengths):
        valid_len = int(max(raw_len, 0))
        if valid_len <= 0:
            continue
        seq = x[idx, :valid_len].transpose(0, 1).unsqueeze(0)  # [1, D, T]
        if valid_len == target_len:
            compressed = seq
        elif valid_len == 1:
            compressed = seq.expand(-1, -1, target_len)
        else:
            compressed = F.interpolate(seq, size=target_len, mode="linear", align_corners=False)
        out[idx] = compressed.squeeze(0).transpose(0, 1)
        out_mask[idx] = True
    return out, out_mask


class AttentivePooling(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 128):
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


class TemporalConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )
        self.pointwise_in = nn.Linear(d_model, 2 * d_model)
        self.pointwise_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.depthwise(h.transpose(1, 2)).transpose(1, 2)
        gate, value = self.pointwise_in(h).chunk(2, dim=-1)
        h = torch.sigmoid(gate) * F.gelu(value)
        h = self.dropout(self.pointwise_out(h))
        return residual + h


class RawSequenceEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        compressed_len: int,
        n_blocks: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.compressed_len = compressed_len
        self.input_proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList(
            [TemporalConvBlock(d_model, kernel_size=kernel_size, dropout=dropout) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_comp, comp_mask = _compress_sequence(x, mask, self.compressed_len)
        h = self.input_proj(x_comp)
        for block in self.blocks:
            h = block(h)
        return self.norm(h), comp_mask


class RawAudioEncoder(nn.Module):
    def __init__(self, cfg: TCDyFISConfig):
        super().__init__()
        self.wav_proj = nn.Linear(cfg.audio_wav2vec_dim, cfg.d_audio)
        self.librosa_proj = nn.Linear(cfg.audio_librosa_dim, cfg.d_audio)
        self.encoder = RawSequenceEncoder(
            d_in=2 * cfg.d_audio,
            d_model=cfg.d_model,
            compressed_len=cfg.compressed_len,
            n_blocks=cfg.n_temporal_blocks,
            kernel_size=cfg.conv_kernel,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        wav2vec_feat: torch.Tensor,
        librosa_feat: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if librosa_feat is None:
            librosa_feat = wav2vec_feat.new_zeros(
                wav2vec_feat.size(0), wav2vec_feat.size(1), self.librosa_proj.in_features
            )
        fused = torch.cat([self.wav_proj(wav2vec_feat), self.librosa_proj(librosa_feat)], dim=-1)
        return self.encoder(fused, mask)


class CompressedCrossModalFusion(nn.Module):
    """Cross attention after temporal compression to keep memory bounded."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.cross_1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        modalities: list[tuple[str, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not modalities:
            raise ValueError("At least one modality is required for fusion")
        if len(modalities) == 1:
            _, seq, mask = modalities[0]
            return self.out_norm(seq), mask

        priority = {"text": 0, "visual": 1, "audio": 2}
        modalities = sorted(modalities, key=lambda item: priority.get(item[0], 99))
        _, primary_seq, primary_mask = modalities[0]
        fused = primary_seq

        for idx, (_, other_seq, other_mask) in enumerate(modalities[1:], start=1):
            cross = self.cross_1 if idx == 1 else self.cross_2
            norm = self.norm_1 if idx == 1 else self.norm_2
            attn_out, _ = cross(primary_seq, other_seq, other_seq, key_padding_mask=~other_mask)
            fused = norm(fused + attn_out)

        return self.out_norm(fused), primary_mask


class DynamicRouteHead(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        fused_global: torch.Tensor,
        pooled_modals: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        available = [vec for vec, flag in pooled_modals if bool(flag.any())]
        if not available:
            return fused_global

        mean_modal = torch.stack(available, dim=1).mean(dim=1)
        max_modal = torch.stack(available, dim=1).amax(dim=1)
        mix_input = torch.cat([fused_global, mean_modal, max_modal, fused_global - mean_modal], dim=-1)
        route_weights = F.softmax(self.gate(mix_input), dim=-1)
        routes = torch.stack(
            [fused_global, mean_modal, max_modal, fused_global * mean_modal],
            dim=1,
        )
        mixed = (route_weights.unsqueeze(-1) * routes).sum(dim=1)
        return self.norm(mixed)


class DyadicInteractionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.counselor_to_patient = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.patient_to_counselor = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.c_gate = nn.Linear(2 * d_model, d_model)
        self.p_gate = nn.Linear(2 * d_model, d_model)
        self.c_norm_1 = nn.LayerNorm(d_model)
        self.p_norm_1 = nn.LayerNorm(d_model)
        self.c_ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.p_ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.c_norm_2 = nn.LayerNorm(d_model)
        self.p_norm_2 = nn.LayerNorm(d_model)

    def _update_role(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
        cross_attn: nn.MultiheadAttention,
        gate_proj: nn.Linear,
        norm_1: nn.LayerNorm,
        ffn: nn.Sequential,
        norm_2: nn.LayerNorm,
    ) -> torch.Tensor:
        attn_out, _ = cross_attn(query, context, context, key_padding_mask=~context_mask)
        gate = torch.sigmoid(gate_proj(torch.cat([query, attn_out], dim=-1)))
        h = norm_1(query + gate * attn_out)
        h = norm_2(h + ffn(h))
        return h * query_mask.unsqueeze(-1).to(h.dtype)

    def forward(
        self,
        counselor_seq: torch.Tensor,
        counselor_mask: torch.Tensor,
        patient_seq: torch.Tensor,
        patient_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counselor_out = self._update_role(
            counselor_seq,
            patient_seq,
            counselor_mask,
            patient_mask,
            self.counselor_to_patient,
            self.c_gate,
            self.c_norm_1,
            self.c_ffn,
            self.c_norm_2,
        )
        patient_out = self._update_role(
            patient_seq,
            counselor_seq,
            patient_mask,
            counselor_mask,
            self.patient_to_counselor,
            self.p_gate,
            self.p_norm_1,
            self.p_ffn,
            self.p_norm_2,
        )
        return counselor_out, patient_out


class DyadicHead(nn.Module):
    def __init__(self, cfg: TCDyFISConfig):
        super().__init__()
        d_model = cfg.d_model
        self.counselor_refine = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(d_model),
        )
        self.patient_refine = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(d_model),
        )
        self.route = nn.Sequential(
            nn.Linear(6 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, 6),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.n_labels),
        )

    def forward(
        self,
        counselor_global: torch.Tensor,
        patient_global: torch.Tensor,
        counselor_context: torch.Tensor,
        patient_context: torch.Tensor,
    ) -> torch.Tensor:
        counselor_repr = self.counselor_refine(torch.cat([counselor_global, counselor_context], dim=-1))
        patient_repr = self.patient_refine(torch.cat([patient_global, patient_context], dim=-1))
        routes = torch.stack(
            [
                counselor_repr,
                patient_repr,
                counselor_repr - patient_repr,
                counselor_repr * patient_repr,
                counselor_context - patient_context,
                counselor_global - patient_global,
            ],
            dim=1,
        )
        route_input = torch.cat(
            [
                counselor_repr,
                patient_repr,
                counselor_context,
                patient_context,
                counselor_repr - patient_repr,
                counselor_repr * patient_repr,
            ],
            dim=-1,
        )
        weights = F.softmax(self.route(route_input), dim=-1)
        fused = (weights.unsqueeze(-1) * routes).sum(dim=1)
        return self.head(fused)


class TCDyFIS(nn.Module):
    def __init__(self, cfg: TCDyFISConfig):
        super().__init__()
        self.cfg = cfg

        if not any([cfg.use_visual, cfg.use_audio, cfg.use_text]):
            raise ValueError("At least one modality must be enabled")

        self.visual_encoder = (
            RawSequenceEncoder(
                d_in=cfg.video_dim,
                d_model=cfg.d_model,
                compressed_len=cfg.compressed_len,
                n_blocks=cfg.n_temporal_blocks,
                kernel_size=cfg.conv_kernel,
                dropout=cfg.dropout,
            )
            if cfg.use_visual
            else None
        )
        self.audio_encoder = RawAudioEncoder(cfg) if cfg.use_audio else None
        self.text_encoder = (
            RawSequenceEncoder(
                d_in=cfg.text_dim,
                d_model=cfg.d_model,
                compressed_len=cfg.text_compressed_len,
                n_blocks=max(1, cfg.n_temporal_blocks - 1),
                kernel_size=cfg.conv_kernel,
                dropout=cfg.dropout,
            )
            if cfg.use_text
            else None
        )

        if cfg.task == 2 and not cfg.share_encoders:
            self.patient_visual_encoder = (
                RawSequenceEncoder(
                    d_in=cfg.video_dim,
                    d_model=cfg.d_model,
                    compressed_len=cfg.compressed_len,
                    n_blocks=cfg.n_temporal_blocks,
                    kernel_size=cfg.conv_kernel,
                    dropout=cfg.dropout,
                )
                if cfg.use_visual
                else None
            )
            self.patient_audio_encoder = RawAudioEncoder(cfg) if cfg.use_audio else None
            self.patient_text_encoder = (
                RawSequenceEncoder(
                    d_in=cfg.text_dim,
                    d_model=cfg.d_model,
                    compressed_len=cfg.text_compressed_len,
                    n_blocks=max(1, cfg.n_temporal_blocks - 1),
                    kernel_size=cfg.conv_kernel,
                    dropout=cfg.dropout,
                )
                if cfg.use_text
                else None
            )
        else:
            self.patient_visual_encoder = self.visual_encoder
            self.patient_audio_encoder = self.audio_encoder
            self.patient_text_encoder = self.text_encoder

        self.fusion = CompressedCrossModalFusion(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.pool = AttentivePooling(cfg.d_model, d_hidden=max(64, cfg.d_model // 2))
        self.route_head = DynamicRouteHead(cfg.d_model, cfg.dropout)

        if cfg.task == 1:
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_hidden, cfg.n_labels),
            )
        else:
            self.dyadic_blocks = nn.ModuleList(
                [DyadicInteractionBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(max(1, cfg.dyadic_layers))]
            )
            self.dyadic_head = DyadicHead(cfg)

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
        pooled_modals: list[tuple[torch.Tensor, torch.Tensor]] = []

        if visual_encoder is not None and role_data.get("video_openface3") is not None:
            visual_seq, visual_mask = visual_encoder(role_data["video_openface3"], seq_mask)
            modalities.append(("visual", visual_seq, visual_mask))
            pooled_modals.append((self.pool(visual_seq, visual_mask), visual_mask.any(dim=1)))

        if audio_encoder is not None and role_data.get("audio_wav2vec") is not None:
            audio_seq, audio_mask = audio_encoder(
                role_data["audio_wav2vec"],
                role_data.get("audio_librosa"),
                seq_mask,
            )
            modalities.append(("audio", audio_seq, audio_mask))
            pooled_modals.append((self.pool(audio_seq, audio_mask), audio_mask.any(dim=1)))

        if text_encoder is not None and role_data.get("text_embedding") is not None:
            text_seq, text_mask = text_encoder(role_data["text_embedding"], tok_mask)
            modalities.append(("text", text_seq, text_mask))
            pooled_modals.append((self.pool(text_seq, text_mask), text_mask.any(dim=1)))

        if not modalities:
            raise ValueError("Current role has no valid modalities")

        fused_seq, fused_mask = self.fusion(modalities)
        fused_global = self.pool(fused_seq, fused_mask)
        return fused_seq, fused_mask, self.route_head(fused_global, pooled_modals)

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
        counselor_seq, counselor_mask, counselor_repr = self._encode_role(
            batch["counselor"],
            batch["counselor_word_mask"],
            batch.get("counselor_tok_mask"),
            self.visual_encoder,
            self.audio_encoder,
            self.text_encoder,
        )
        patient_seq, patient_mask, patient_repr = self._encode_role(
            batch["patient"],
            batch["patient_word_mask"],
            batch.get("patient_tok_mask"),
            self.patient_visual_encoder,
            self.patient_audio_encoder,
            self.patient_text_encoder,
        )
        counselor_ctx, patient_ctx = counselor_seq, patient_seq
        for block in self.dyadic_blocks:
            counselor_ctx, patient_ctx = block(counselor_ctx, counselor_mask, patient_ctx, patient_mask)
        counselor_context = self.pool(counselor_ctx, counselor_mask)
        patient_context = self.pool(patient_ctx, patient_mask)
        return self.dyadic_head(counselor_repr, patient_repr, counselor_context, patient_context)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.cfg.task == 2:
            return self._forward_task2(batch)
        return self._forward_task1(batch)
