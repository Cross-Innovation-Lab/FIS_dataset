"""
TimeFilter 基线 + FIS 回归
=========================

将 `lib/Time-Series-Library` 中的 TimeFilter 模型改造为 FIS 任务的基线：
- 只使用 OpenFace 序列 `video_openface3` 作为时序输入；
- 采用 TimeFilter 的 patch + TimeFilter_Backbone + classification 头得到全局表示；
- 任务一：仅咨询师；任务二：咨询师与来访者分别编码后再融合回归 9 维 FIS 分数。

接口与其他模型保持一致：`forward(batch: dict) -> Tensor[B, n_labels]`。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  引入 TimeFilter 库
# ---------------------------------------------------------------------------

from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TIMEFILTER_ROOT = _PROJECT_ROOT / "lib" / "Time-Series-Library"
if _TIMEFILTER_ROOT.exists() and str(_TIMEFILTER_ROOT) not in sys.path:
    sys.path.append(str(_TIMEFILTER_ROOT))

try:
    from models.TimeFilter import Model as _TimeFilterCore  # type: ignore[import]
except Exception as exc:  # pragma: no cover - 导入错误在实际运行时暴露
    raise ImportError(
        "导入 TimeFilter 失败，请确认 `lib/Time-Series-Library` 在工程根目录下，"
        "且其中包含 `models/TimeFilter.py` 与 `layers` 目录。"
    ) from exc


# ---------------------------------------------------------------------------
#  配置
# ---------------------------------------------------------------------------


@dataclass
class TimeFilterFISConfig:
    """TimeFilter-FIS 超参数。

    仅使用视觉 OpenFace 特征：`video_openface3` [B, N_word, video_dim]。
    `seq_len` 需与 dataloader 的 `max_len_word` 对齐（或为其上界），
    模型内部会对序列做截断 / padding 到 `seq_len`。
    """

    video_dim: int = 235
    seq_len: int = 128
    d_model: int = 256
    d_ff: int = 512
    patch_len: int = 16
    n_heads: int = 4
    e_layers: int = 2
    dropout: float = 0.1
    alpha: float = 0.1
    top_p: float = 0.5
    n_labels: int = 9
    task: int = 1  # 1=仅咨询师, 2=咨询师+来访者


# ---------------------------------------------------------------------------
#  TimeFilter-FIS 主模型
# ---------------------------------------------------------------------------


class TimeFilterFIS(nn.Module):
    """TimeFilter 基线模型，适配 FIS 任务一 / 任务二。

    - 任务一：仅使用咨询师 `counselor.video_openface3` → TimeFilter → 9 维回归。
    - 任务二：咨询师与来访者分别通过同一 TimeFilter 编码，
      得到两个 9 维向量后拼接，经 MLP 融合为最终 9 维输出。
    """

    def __init__(self, cfg: TimeFilterFISConfig):
        super().__init__()
        self.cfg = cfg

        # 构造 TimeFilter 的配置对象（对应 TimeFilter 原始代码中的 configs）
        tf_cfg = SimpleNamespace(
            task_name="classification",
            seq_len=cfg.seq_len,
            pred_len=cfg.seq_len,  # 分类任务中未实际使用
            c_out=cfg.video_dim,
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            patch_len=cfg.patch_len,
            n_heads=cfg.n_heads,
            e_layers=cfg.e_layers,
            dropout=cfg.dropout,
            alpha=cfg.alpha,
            top_p=cfg.top_p,
            pos=True,
            enc_in=cfg.video_dim,
            num_class=cfg.n_labels,
        )
        # 单个 TimeFilter 模型，用于编码任一角色
        self.core = _TimeFilterCore(tf_cfg)

        # 任务二：角色级预测拼接后的融合头
        if cfg.task == 2:
            self.task2_head = nn.Sequential(
                nn.Linear(cfg.n_labels * 2, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.n_labels),
            )
        else:
            self.task2_head = None

    # ------------------------------------------------------------------
    #  单角色编码
    # ------------------------------------------------------------------

    def _encode_role(self, role_data: dict[str, Any]) -> torch.Tensor:
        """使用 TimeFilter 对单个角色的 OpenFace 序列做编码并输出 9 维预测。

        Args:
            role_data: `batch["counselor"]` 或 `batch["patient"]` 字典。

        Returns:
            Tensor[B, n_labels]
        """
        video = role_data.get("video_openface3")
        if video is None:
            raise ValueError("TimeFilterFIS 需要 `video_openface3` 特征。")

        if video.dim() != 3:
            raise ValueError(f"video_openface3 期望形状 [B, N_word, D]，当前为 {tuple(video.shape)}")

        B, N, D = video.shape
        if D != self.cfg.video_dim:
            raise ValueError(
                f"video_openface3 最后维度应为 {self.cfg.video_dim}，当前为 {D}。"
            )

        # 截断或 padding 到固定 seq_len
        if N > self.cfg.seq_len:
            x = video[:, : self.cfg.seq_len, :]
        elif N < self.cfg.seq_len:
            pad = video.new_zeros(B, self.cfg.seq_len - N, D)
            x = torch.cat([video, pad], dim=1)
        else:
            x = video

        # TimeFilter 分类分支：x_enc 为 [B, T, C]，x_mark_enc 未使用，传 None 即可
        logits = self.core.classification(x, None)  # [B, n_labels]
        return logits

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """适配 `experiment.dataloader.collate_fis_batch` 的 batch 格式。"""
        if self.cfg.task == 2:
            return self._forward_task2(batch)
        return self._forward_task1(batch)

    def _forward_task1(self, batch: dict[str, Any]) -> torch.Tensor:
        if "counselor" not in batch:
            raise KeyError("batch 缺少 `counselor` 键。")
        return self._encode_role(batch["counselor"])

    def _forward_task2(self, batch: dict[str, Any]) -> torch.Tensor:
        if "counselor" not in batch or "patient" not in batch:
            raise KeyError("任务二需要 batch 同时包含 `counselor` 与 `patient`。")

        c_logits = self._encode_role(batch["counselor"])  # [B, n_labels]
        p_logits = self._encode_role(batch["patient"])    # [B, n_labels]

        if self.task2_head is None:
            # 退化方案：直接平均两个角色预测
            return 0.5 * (c_logits + p_logits)

        fused = torch.cat([c_logits, p_logits], dim=-1)  # [B, 2 * n_labels]
        return self.task2_head(fused)


