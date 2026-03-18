"""Time-Series-Library 适配 FIS 任务的公共工具。"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TS_LIB_ROOT = _PROJECT_ROOT / "lib" / "Time-Series-Library"
if _TS_LIB_ROOT.exists() and str(_TS_LIB_ROOT) not in sys.path:
    sys.path.append(str(_TS_LIB_ROOT))


def build_regression_head(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def masked_mean(seq: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """对 `[B, N, D]` 序列按 mask 做均值池化。"""
    if mask is None:
        return seq.mean(dim=1)
    mask = mask.to(torch.bool)
    weight = mask.to(seq.dtype).unsqueeze(-1)
    summed = (seq * weight).sum(dim=1)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return summed / count


def pad_or_truncate_sequence(
    seq: torch.Tensor,
    mask: torch.Tensor | None,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将变长序列裁剪/补零到固定长度，并同步处理 mask。"""
    if seq.dim() != 3:
        raise ValueError(f"期望序列形状为 [B, N, D]，当前为 {tuple(seq.shape)}")

    batch_size, seq_len, feat_dim = seq.shape
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=seq.device)
    else:
        mask = mask.to(device=seq.device, dtype=torch.bool)
        if mask.size(0) != batch_size:
            raise ValueError(
                f"mask batch 维与序列不一致: seq={tuple(seq.shape)}, mask={tuple(mask.shape)}"
            )
        if mask.size(1) < seq_len:
            pad = torch.zeros(batch_size, seq_len - mask.size(1), dtype=torch.bool, device=seq.device)
            mask = torch.cat([mask, pad], dim=1)
        elif mask.size(1) > seq_len:
            mask = mask[:, :seq_len]

    if seq_len > target_len:
        seq = seq[:, :target_len, :]
        mask = mask[:, :target_len]
    elif seq_len < target_len:
        seq_pad = seq.new_zeros(batch_size, target_len - seq_len, feat_dim)
        mask_pad = torch.zeros(batch_size, target_len - seq_len, dtype=torch.bool, device=seq.device)
        seq = torch.cat([seq, seq_pad], dim=1)
        mask = torch.cat([mask, mask_pad], dim=1)

    seq = seq * mask.unsqueeze(-1).to(seq.dtype)
    return seq, mask

