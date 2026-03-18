"""
KANAD + FIS 回归适配
===================

保留 KANAD 的逐通道异常检测式编码，再包装成当前实验统一接口：
- `forward(batch: dict) -> Tensor[B, 9]`
- 支持任务一 / 任务二
- 视觉/音频走 KANAD，文本走当前实验已有的聚合逻辑
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from experiment.dataloader import aggregate_text_embedding
from experiment.model.ts_fis_utils import (
    build_regression_head,
    masked_mean,
    pad_or_truncate_sequence,
)


class KANADModel(nn.Module):
    def __init__(self, window: int, order: int, *args, **kwargs) -> None:
        super().__init__()
        self.order = order
        self.window = window
        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine(self.window, self.order).unsqueeze(0),
        )
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Linear(window, window)

    def forward(self, x: torch.Tensor, return_last: bool = False, *args, **kwargs):
        residuals = [x.unsqueeze(1)]
        ff = torch.concat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [torch.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )
        residuals.append(ff)
        ff = self.act(self.bn1(self.init_conv(ff)))
        ff = self.act(self.bn2(self.inner_conv(ff) + residuals.pop()))
        ff = self.act(self.bn3(self.out_conv(ff) + residuals.pop()))
        ff = self.final_conv(ff)
        if return_last:
            return ff.squeeze(1), ff
        return ff.squeeze(1)

    def _create_custom_periodic_cosine(self, window: int, period: int | list[int]) -> torch.Tensor:
        d = len(period) if isinstance(period, list) else period
        periods = period if isinstance(period, list) else [i for i in range(1, period + 1)]
        result = torch.empty(d, window, dtype=torch.float32)
        for i, p in enumerate(periods):
            t = torch.arange(0, 1, 1 / window, dtype=torch.float32) / p * 2 * np.pi
            result[i, :] = torch.cos(t)
        return result


@dataclass
class KANADFISConfig:
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
    order: int = 8


class _KANADSequenceEncoder(nn.Module):
    """单模态 KANAD 序列编码器。"""

    def __init__(self, input_dim: int, cfg: KANADFISConfig):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.core = KANADModel(window=cfg.seq_len, order=cfg.order)
        self.proj = nn.Linear(input_dim, cfg.hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x, mask = pad_or_truncate_sequence(x, mask, self.seq_len)
        x_input = rearrange(x, "b l d -> (b d) l")
        h = self.core(x_input)
        h = rearrange(h, "(b d) l -> b l d", b=x.size(0), d=x.size(2))
        return self.proj(masked_mean(h, mask))


class KANADFIS(nn.Module):
    """KANAD 基线，适配当前 FIS 多模态回归任务。"""

    def __init__(self, cfg: KANADFISConfig):
        super().__init__()
        self.cfg = cfg
        fusion_dim = 0

        if cfg.use_visual:
            self.visual_enc = _KANADSequenceEncoder(cfg.video_dim, cfg)
            fusion_dim += cfg.hidden_dim
        else:
            self.visual_enc = None

        if cfg.use_audio:
            self.audio_enc = _KANADSequenceEncoder(cfg.audio_dim, cfg)
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


Model = KANADFIS
