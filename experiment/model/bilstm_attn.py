"""
BiLSTM + Attention 基线模型
==========================
经典时序基线，支持任务一（仅咨询师）与任务二（咨询师+来访者）。
各模态独立 BiLSTM 编码 + 注意力池化，再拼接后 MLP 回归 FIS 分数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment.dataloader import aggregate_text_embedding


# ---------------------------------------------------------------------------
#  配置
# ---------------------------------------------------------------------------


@dataclass
class BiLSTMAttnConfig:
    """BiLSTM + Attention 超参数。"""

    video_dim: int = 235
    audio_dim: int = 768
    text_dim: int = 1024
    hidden_dim: int = 256
    num_layers: int = 2
    n_labels: int = 9
    dropout: float = 0.2
    task: int = 1
    use_visual: bool = True
    use_audio: bool = True
    use_text: bool = True
    text_pool: str = "mean_pool"


# ---------------------------------------------------------------------------
#  注意力池化
# ---------------------------------------------------------------------------


class AttentivePooling(nn.Module):
    """学习式注意力池化： [B, N, D] + mask -> [B, D]。"""

    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, D], mask: [B, N] True=有效
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(scores, dim=-1)
        # 当某行 mask 全为 False 时 softmax(-inf,...,-inf)=nan，导致输出 nan；置为 0 使该样本输出零向量
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        return (alpha.unsqueeze(-1) * x).sum(dim=1)


def masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对 [B, N, D] 按 mask 做均值池化 -> [B, D]。"""
    m = mask.to(seq.dtype).unsqueeze(-1)
    s = (seq * m).sum(dim=1)
    c = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return s / c


# ---------------------------------------------------------------------------
#  单模态 BiLSTM 编码器
# ---------------------------------------------------------------------------


class ModalityBiLSTM(nn.Module):
    """单模态 BiLSTM + 注意力池化。输入 [B, N, D_in] -> 输出 [B, hidden_dim*2]。"""

    def __init__(
        self,
        d_in: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            d_in,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.pool = AttentivePooling(hidden_dim * 2, d_hidden=hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, D_in]
        out, _ = self.lstm(x)
        return self.pool(out, mask)


# ---------------------------------------------------------------------------
#  BiLSTM + Attention 主模型
# ---------------------------------------------------------------------------


class BiLSTMAttnFIS(nn.Module):
    """
    BiLSTM + Attention 基线：各模态 BiLSTM -> 注意力池化 -> 拼接 -> MLP -> n_labels。

    - 任务一：仅 counselor，forward 返回 [B, n_labels]。
    - 任务二：counselor + patient 各编码后拼接再回归。
    接口与 FIS-Net 一致：forward(batch) -> [B, n_labels]。
    """

    def __init__(self, cfg: BiLSTMAttnConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = 0

        if cfg.use_visual:
            self.visual_enc = ModalityBiLSTM(
                cfg.video_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout
            )
            in_dim += cfg.hidden_dim * 2
        else:
            self.visual_enc = None

        if cfg.use_audio:
            self.audio_enc = ModalityBiLSTM(
                cfg.audio_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout
            )
            in_dim += cfg.hidden_dim * 2
        else:
            self.audio_enc = None

        if cfg.use_text:
            self.text_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim * 2)
            in_dim += cfg.hidden_dim * 2
        else:
            self.text_proj = None

        if in_dim == 0:
            raise ValueError("至少启用一种模态 (use_visual / use_audio / use_text)")

        if cfg.task == 2:
            in_dim *= 2

        self.head = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.n_labels),
        )

    def _encode_role(
        self,
        role_data: dict[str, Any],
        word_mask: torch.Tensor,
        tok_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """单角色多模态编码 -> [B, D]."""
        parts: list[torch.Tensor] = []

        if self.visual_enc is not None and role_data.get("video_openface3") is not None:
            v = role_data["video_openface3"]
            parts.append(self.visual_enc(v, word_mask))

        if self.audio_enc is not None and role_data.get("audio_wav2vec") is not None:
            a = role_data["audio_wav2vec"]
            parts.append(self.audio_enc(a, word_mask))

        if self.text_proj is not None and role_data.get("text_embedding") is not None:
            text_emb = role_data["text_embedding"]
            tok_m = tok_mask
            h = aggregate_text_embedding(
                text_emb, tok_m, method=self.cfg.text_pool
            )
            parts.append(self.text_proj(h))

        if not parts:
            raise ValueError("当前角色无可用模态特征")
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            batch: collate_fis_batch 输出的 dict。

        Returns:
            [B, n_labels] FIS 预测分数。
        """
        if self.cfg.task == 2:
            return self._forward_task2(batch)
        return self._forward_task1(batch)

    def _forward_task1(self, batch: dict[str, Any]) -> torch.Tensor:
        c_global = self._encode_role(
            batch["counselor"],
            batch["counselor_word_mask"],
            batch.get("counselor_tok_mask"),
        )
        return self.head(c_global)

    def _forward_task2(self, batch: dict[str, Any]) -> torch.Tensor:
        c_global = self._encode_role(
            batch["counselor"],
            batch["counselor_word_mask"],
            batch.get("counselor_tok_mask"),
        )
        p_global = self._encode_role(
            batch["patient"],
            batch["patient_word_mask"],
            batch.get("patient_tok_mask"),
        )
        fused = torch.cat([c_global, p_global], dim=-1)
        return self.head(fused)
