"""
Reformer + FIS 回归适配
======================

将 Reformer 包装为当前实验统一接口：
- `forward(batch: dict) -> Tensor[B, 9]`
- 视觉/音频走 Reformer 分类式序列编码
- 文本复用现有池化逻辑
- 支持任务一 / 任务二
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
    pad_or_truncate_sequence,
)

try:
    from models.Reformer import Model as _ReformerCore  # type: ignore[import]
    _REFORMER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - 依赖缺失时在实例化阶段报错
    _ReformerCore = None
    _REFORMER_IMPORT_ERROR = exc


@dataclass
class ReformerFISConfig:
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
    d_model: int = 256
    d_ff: int = 512
    n_heads: int = 4
    e_layers: int = 2
    activation: str = "gelu"
    embed: str = "fixed"
    freq: str = "h"
    bucket_size: int = 4
    n_hashes: int = 4


class _ReformerSequenceEncoder(nn.Module):
    """单模态 Reformer 序列编码器。"""

    def __init__(self, input_dim: int, cfg: ReformerFISConfig):
        super().__init__()
        if _ReformerCore is None:
            raise ImportError(
                "ReformerFIS 依赖 `reformer_pytorch`。请先安装该依赖后再使用 `reformer` 基线。"
            ) from _REFORMER_IMPORT_ERROR
        self.seq_len = cfg.seq_len
        core_cfg = SimpleNamespace(
            task_name="classification",
            seq_len=cfg.seq_len,
            pred_len=cfg.seq_len,
            enc_in=input_dim,
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            n_heads=cfg.n_heads,
            e_layers=cfg.e_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            embed=cfg.embed,
            freq=cfg.freq,
            num_class=cfg.hidden_dim,
            c_out=input_dim,
        )
        self.core = _ReformerCore(core_cfg, bucket_size=cfg.bucket_size, n_hashes=cfg.n_hashes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x, mask = pad_or_truncate_sequence(x, mask, self.seq_len)
        return self.core.classification(x, mask.to(x.dtype))


class ReformerFIS(nn.Module):
    """Reformer 基线，适配当前 FIS 多模态回归任务。"""

    def __init__(self, cfg: ReformerFISConfig):
        super().__init__()
        self.cfg = cfg
        fusion_dim = 0

        if cfg.use_visual:
            self.visual_enc = _ReformerSequenceEncoder(cfg.video_dim, cfg)
            fusion_dim += cfg.hidden_dim
        else:
            self.visual_enc = None

        if cfg.use_audio:
            self.audio_enc = _ReformerSequenceEncoder(cfg.audio_dim, cfg)
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


Model = ReformerFIS
