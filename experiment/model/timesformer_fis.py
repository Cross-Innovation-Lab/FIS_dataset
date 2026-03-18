"""
TimeSformer-style 编码器 + FIS 回归
====================================
基于 ICML 2021 TimeSformer 的「分离时空注意力」思想，对词级 OpenFace 序列建模：
将 [B, N_word, 235] 视为 N 个时间步、每步 235 维特征，按 235 拆成若干「空间」patch，
在块内先做时间维注意力、再做空间维注意力，最后池化回归 FIS。
支持单模态（仅视频）或三模态（视频 + 音频 + 文本池化）融合，任务一与任务二。
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
class TimeSformerFISConfig:
    """TimeSformer-FIS 超参数。"""

    video_dim: int = 235
    audio_dim: int = 768
    text_dim: int = 1024
    d_model: int = 256
    n_patches: int = 5
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    n_labels: int = 9
    pool_hidden: int = 128
    head_hidden: int = 256
    task: int = 1
    use_visual: bool = True
    use_audio: bool = True
    use_text: bool = True
    text_pool: str = "mean_pool"


# ---------------------------------------------------------------------------
#  时空分离注意力（TimeSformer 风格）
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """将 video_dim 拆成 n_patches 个 patch 并投影到 d_model。"""

    def __init__(self, video_dim: int, n_patches: int, d_model: int):
        super().__init__()
        # 235 = 5*47，n_patches=5 时 patch_dim=47
        if video_dim % n_patches != 0:
            raise ValueError(f"video_dim={video_dim} 需被 n_patches={n_patches} 整除")
        self.patch_dim = video_dim // n_patches
        self.n_patches = n_patches
        self.proj = nn.Linear(self.patch_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, video_dim] -> [B, N, n_patches, d_model]
        B, N, _ = x.shape
        x = x.view(B, N, self.n_patches, self.patch_dim)
        return self.proj(x)


class TemporalAttention(nn.Module):
    """时间维自注意力：对 [B, n_patches, N, d] 在 N 维上做 attention。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, n_patches, N, d] -> 对 N 做 attention
        B, P, N, D = x.shape
        x_flat = x.reshape(B * P, N, D)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, P, -1).reshape(B * P, N)
        out, _ = self.attn(x_flat, x_flat, x_flat, key_padding_mask=key_padding_mask)
        out = out.reshape(B, P, N, D)
        return self.norm(x + out)


class SpatialAttention(nn.Module):
    """空间维自注意力：对 [B, N, n_patches, d] 在 n_patches 维上做 attention。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, n_patches, d]
        B, N, P, D = x.shape
        x_flat = x.reshape(B * N, P, D)
        out, _ = self.attn(x_flat, x_flat, x_flat)
        out = out.reshape(B, N, P, D)
        return self.norm(x + out)


class TimeSformerBlock(nn.Module):
    """单层：Temporal Attention -> Spatial Attention -> FFN（分离时空注意力）。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout)
        self.spatial_attn = SpatialAttention(d_model, n_heads, dropout)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, N, n_patches, d]
        B, N, P, D = x.shape
        x_t = x.permute(0, 2, 1, 3)
        x_t = self.temporal_attn(x_t, key_padding_mask)
        x = x_t.permute(0, 2, 1, 3)
        x = self.spatial_attn(x)
        x = x + self.norm(self.mlp(x))
        return x


class TimeSformerEncoder(nn.Module):
    """
    词级 OpenFace 序列的 TimeSformer 编码器。
    输入 [B, N, video_dim] -> patch embed -> 时空块 -> [B, N, n_patches, d_model]。
    """

    def __init__(self, cfg: TimeSformerFISConfig):
        super().__init__()
        self.patch_embed = PatchEmbed(
            cfg.video_dim, cfg.n_patches, cfg.d_model
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, cfg.n_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TimeSformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        video: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # video: [B, N, video_dim]
        x = self.patch_embed(video)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x, key_padding_mask=None if mask is None else ~mask)
        return self.norm(x)


class AttentivePooling(nn.Module):
    """将 [B, N, P, D] 按 mask 做注意力池化 -> [B, D]。"""

    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, P, D = x.shape
        x_flat = x.reshape(B, N * P, D)
        scores = self.attn(x_flat).squeeze(-1)
        if mask is not None:
            mask_flat = mask.unsqueeze(2).expand(-1, -1, P).reshape(B, N * P)
            scores = scores.masked_fill(~mask_flat, float("-inf"))
        alpha = F.softmax(scores, dim=-1)
        # 当某行 mask 全为 False 时 softmax(-inf,...,-inf)=nan；置为 0 使该样本输出零向量
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        return (alpha.unsqueeze(-1) * x_flat).sum(dim=1)


def masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, N, D] + mask -> [B, D] 均值池化。"""
    m = mask.to(seq.dtype).unsqueeze(-1)
    s = (seq * m).sum(dim=1)
    c = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return s / c


# ---------------------------------------------------------------------------
#  TimeSformer-FIS 主模型
# ---------------------------------------------------------------------------


class TimeSformerFIS(nn.Module):
    """
    TimeSformer 编码视频 + 可选音频/文本池化，拼接后 MLP 回归 FIS。
    任务一：仅 counselor；任务二：counselor + patient 各编码后拼接回归。
    接口：forward(batch) -> [B, n_labels]。
    """

    def __init__(self, cfg: TimeSformerFISConfig):
        super().__init__()
        self.cfg = cfg
        fusion_dim = 0

        if cfg.use_visual:
            self.visual_enc = TimeSformerEncoder(cfg)
            self.visual_pool = AttentivePooling(cfg.d_model, cfg.pool_hidden)
            fusion_dim += cfg.d_model
        else:
            self.visual_enc = None
            self.visual_pool = None

        if cfg.use_audio:
            self.audio_proj = nn.Linear(cfg.audio_dim, cfg.d_model)
            fusion_dim += cfg.d_model
        else:
            self.audio_proj = None

        if cfg.use_text:
            self.text_proj = nn.Linear(cfg.text_dim, cfg.d_model)
            fusion_dim += cfg.d_model
        else:
            self.text_proj = None

        if fusion_dim == 0:
            raise ValueError("至少启用一种模态")

        if cfg.task == 2:
            fusion_dim *= 2

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.n_labels),
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
            h = self.visual_enc(v, word_mask)
            parts.append(self.visual_pool(h, word_mask))

        if self.audio_proj is not None and role_data.get("audio_wav2vec") is not None:
            a = role_data["audio_wav2vec"]
            a_h = masked_mean(a, word_mask)
            parts.append(self.audio_proj(a_h))

        if self.text_proj is not None and role_data.get("text_embedding") is not None:
            text_emb = role_data["text_embedding"]
            tok_m = tok_mask if tok_mask is not None else torch.ones(
                text_emb.size(0), text_emb.size(1), dtype=torch.bool, device=text_emb.device
            )
            h = aggregate_text_embedding(text_emb, tok_m, method=self.cfg.text_pool)
            parts.append(self.text_proj(h))

        if not parts:
            raise ValueError("当前角色无可用模态特征")
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
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
