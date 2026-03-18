"""精简 FIS 回归模型接口（文本/多模态）。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from experiment.dataloader import aggregate_text_embedding


def masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对 [B, N, D] 序列按 mask 做均值池化。"""
    m = mask.to(seq.dtype).unsqueeze(-1)
    s = (seq * m).sum(dim=1)
    c = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return s / c


class FISTextRegressor(nn.Module):
    """仅文本回归：text_embedding -> pooling -> MLP -> 9维分数。"""

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, out_dim: int = 9, pool: str = "mean_pool") -> None:
        super().__init__()
        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        text_emb = batch["counselor"]["text_embedding"]  # [B, N_tok, D]
        tok_mask = batch["counselor_tok_mask"]  # [B, N_tok]
        h = aggregate_text_embedding(text_emb, tok_mask, method=self.pool)  # [B, D]
        return self.head(h)  # [B, out_dim]


class FISSimpleMultimodalRegressor(nn.Module):
    """精简多模态回归：文本/音频/视觉池化后拼接回归。"""

    def __init__(
        self,
        text_dim: int = 1024,
        audio_dim: int = 768,
        video_dim: int = 235,
        hidden_dim: int = 256,
        out_dim: int = 9,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
        text_pool: str = "mean_pool",
    ) -> None:
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video
        self.text_pool = text_pool

        in_dim = 0
        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            in_dim += hidden_dim
        if use_audio:
            self.audio_proj = nn.Linear(audio_dim, hidden_dim)
            in_dim += hidden_dim
        if use_video:
            self.video_proj = nn.Linear(video_dim, hidden_dim)
            in_dim += hidden_dim

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        cw_mask = batch["counselor_word_mask"]  # [B, N_word]

        if self.use_text:
            text_emb = batch["counselor"]["text_embedding"]
            tok_mask = batch["counselor_tok_mask"]
            text_h = aggregate_text_embedding(text_emb, tok_mask, method=self.text_pool)
            parts.append(self.text_proj(text_h))

        if self.use_audio and "audio_wav2vec" in batch["counselor"]:
            a = batch["counselor"]["audio_wav2vec"]  # [B, N_word, 768]
            a_h = masked_mean(a, cw_mask)
            parts.append(self.audio_proj(a_h))

        if self.use_video and "video_openface3" in batch["counselor"]:
            v = batch["counselor"]["video_openface3"]  # [B, N_word, D_vid]
            v_h = masked_mean(v, cw_mask)
            parts.append(self.video_proj(v_h))

        if not parts:
            raise ValueError("当前 batch 缺少可用模态特征。")
        x = torch.cat(parts, dim=-1)
        return self.head(x)

