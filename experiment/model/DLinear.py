"""
DLinear + FIS 回归适配
=====================

将 Time-Series-Library 中的 DLinear 包装为当前实验统一接口：
- `forward(batch: dict) -> Tensor[B, 9]`
- 支持任务一（仅 counselor）与任务二（counselor + patient）
- 视觉/音频走 DLinear 时序编码，文本沿用现有实验中的池化方式
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from experiment.dataloader import aggregate_text_embedding
from experiment.model.ts_fis_utils import (
    build_regression_head,
    masked_mean,
    pad_or_truncate_sequence,
)

from models.DLinear import Model as _DLinearCore  # type: ignore[import]


@dataclass
class DLinearFISConfig:
    video_dim: int = 235
    audio_dim: int = 768
    text_dim: int = 1024
    hidden_dim: int = 256
    n_labels: int = 9
    dropout: float = 0.2
    task: int = 1
    use_visual: bool = True
    use_audio: bool = True
    use_text: bool = True
    text_pool: str = "mean_pool"
    seq_len: int = 128
    moving_avg: int = 25
    individual: bool = False


class _DLinearSequenceEncoder(nn.Module):
    """单模态 DLinear 序列编码器。"""

    def __init__(self, input_dim: int, cfg: DLinearFISConfig):
        super().__init__()
        self.seq_len = cfg.seq_len
        core_cfg = SimpleNamespace(
            task_name="anomaly_detection",
            seq_len=cfg.seq_len,
            pred_len=cfg.seq_len,
            enc_in=input_dim,
            moving_avg=cfg.moving_avg,
        )
        self.core = _DLinearCore(core_cfg, individual=cfg.individual)
        self.proj = nn.Linear(input_dim, cfg.hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x, mask = pad_or_truncate_sequence(x, mask, self.seq_len)
        h = self.core.anomaly_detection(x)
        return self.proj(masked_mean(h, mask))


class DLinearFIS(nn.Module):
    """DLinear 基线，适配当前 FIS 多模态回归任务。"""

    def __init__(self, cfg: DLinearFISConfig):
        super().__init__()
        self.cfg = cfg
        fusion_dim = 0

        if cfg.use_visual:
            self.visual_enc = _DLinearSequenceEncoder(cfg.video_dim, cfg)
            fusion_dim += cfg.hidden_dim
        else:
            self.visual_enc = None

        if cfg.use_audio:
            self.audio_enc = _DLinearSequenceEncoder(cfg.audio_dim, cfg)
            fusion_dim += cfg.hidden_dim
        else:
            self.audio_enc = None

        if cfg.use_text:
            self.text_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim)
            fusion_dim += cfg.hidden_dim
        else:
            self.text_proj = None

        if fusion_dim == 0:
            raise ValueError("至少启用一种模态 (use_visual / use_audio / use_text)")

        if cfg.task == 2:
            fusion_dim *= 2

        self.head = build_regression_head(fusion_dim, cfg.hidden_dim, cfg.n_labels, cfg.dropout)

    def _encode_role(
        self,
        role_data: dict[str, Any],
        word_mask: torch.Tensor,
        tok_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        if self.visual_enc is not None and role_data.get("video_openface3") is not None:
            parts.append(self.visual_enc(role_data["video_openface3"], word_mask))

        if self.audio_enc is not None and role_data.get("audio_wav2vec") is not None:
            parts.append(self.audio_enc(role_data["audio_wav2vec"], word_mask))

        if self.text_proj is not None and role_data.get("text_embedding") is not None:
            text_emb = role_data["text_embedding"]
            tok_m = tok_mask if tok_mask is not None else torch.ones(
                text_emb.size(0), text_emb.size(1), dtype=torch.bool, device=text_emb.device
            )
            text_h = aggregate_text_embedding(text_emb, tok_m, method=self.cfg.text_pool)
            parts.append(self.text_proj(text_h))

        if not parts:
            raise ValueError("当前角色无可用模态特征")
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        c_global = self._encode_role(
            batch["counselor"],
            batch["counselor_word_mask"],
            batch.get("counselor_tok_mask"),
        )
        if self.cfg.task == 1:
            return self.head(c_global)

        p_global = self._encode_role(
            batch["patient"],
            batch["patient_word_mask"],
            batch.get("patient_tok_mask"),
        )
        return self.head(torch.cat([c_global, p_global], dim=-1))


Model = DLinearFIS
