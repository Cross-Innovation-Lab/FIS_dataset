"""回归任务指标：MSE / MAE / 与真人相似度（0–1），支持多维度及指定维度平均。"""

from __future__ import annotations

from typing import Sequence

import torch


def concordance_correlation_coefficient(pred: torch.Tensor, target: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """计算 Concordance Correlation Coefficient (CCC)，沿 dim 维做样本聚合。

    CCC 衡量预测与真值的一致性（含平移与缩放），常用于情感/特质评分与真人标注的一致性。
    公式: CCC = (2 * ρ * σ_pred * σ_target) / (σ_pred^2 + σ_target^2 + (μ_pred - μ_target)^2)
    范围 [-1, 1]，1 表示完全一致。

    Args:
        pred: 预测值，形状 [N, ...] 或 [N, C]（多维度时每列为一特质）。
        target: 真值，形状与 pred 相同。
        dim: 样本所在维度，默认 0。

    Returns:
        标量或形状 [C] 的 CCC，与 pred 去掉 dim 后的形状一致。方差为 0 或样本不足时为 NaN。
    """
    pred_flat = pred.moveaxis(dim, 0)  # [N, ...]
    target_flat = target.moveaxis(dim, 0)
    n = pred_flat.shape[0]
    if n < 2:
        return torch.full((), float("nan"), device=pred.device, dtype=pred.dtype)

    mu_p = pred_flat.mean(dim=0)
    mu_t = target_flat.mean(dim=0)
    var_p = pred_flat.var(dim=0, unbiased=True)
    var_t = target_flat.var(dim=0, unbiased=True)
    cov = ((pred_flat - mu_p) * (target_flat - mu_t)).sum(dim=0) / max(n - 1, 1)
    # CCC = 2*cov / (var_p + var_t + (mu_p - mu_t)^2)
    denom = var_p + var_t + (mu_p - mu_t) ** 2
    ccc = torch.where(denom > 1e-12, 2.0 * cov / denom, torch.full_like(denom, float("nan")))
    return ccc.squeeze()


def human_similarity_from_ccc(ccc: torch.Tensor) -> torch.Tensor:
    """将 CCC ∈ [-1, 1] 映射为与真人相似度 ∈ [0, 1]，1 表示越接近真人评分。"""
    return ((ccc.clamp(-1.0, 1.0) + 1.0) / 2.0).to(ccc.dtype)


def regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    label_names: Sequence[str] | None = None,
    dim_indices: Sequence[int] | None = None,
    return_per_dim: bool = False,
) -> dict[str, float | list[float] | dict[str, float]]:
    """计算回归指标：MSE、MAE、与真人相似度（0–1）；支持多维度及指定维度平均。

    与真人相似度基于 Concordance Correlation Coefficient (CCC)：(ccc + 1) / 2，
    范围 [0, 1]，1 表示预测与真人评分越一致。

    Args:
        pred: 预测值 [B, C]。
        target: 真值 [B, C]。
        label_names: 可选，长度为 C 的标签名列表，用于 per_dim 的键名。
        dim_indices: 可选，要参与「指定维度平均」的列下标；为 None 时与「整体」一致（全部维度）。
        return_per_dim: 是否返回每维的指标列表/字典。

    Returns:
        至少包含:
        - mse, mae, human_sim: 整体（对全部 C 维求平均）。
        - 若 dim_indices 非空: mse_specified, mae_specified, human_sim_specified（仅对指定维度平均）。
        - 若 return_per_dim 为 True: mse_per_dim, mae_per_dim, human_sim_per_dim（列表或按 label_names 的字典）。
    """
    pred = pred.float()
    target = target.float()
    B, C = pred.shape
    if B == 0:
        return {"mse": float("nan"), "mae": float("nan"), "human_sim": float("nan")}

    # 整体：所有维度一起算标量
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    ccc = concordance_correlation_coefficient(pred, target, dim=0)
    if ccc.dim() == 0:
        ccc_scalar = ccc
    else:
        ccc_scalar = torch.nanmean(ccc)
    human_sim = float(human_similarity_from_ccc(ccc_scalar).item())

    out: dict[str, float | list[float] | dict[str, float]] = {
        "mse": float(mse),
        "mae": float(mae),
        "human_sim": human_sim,
    }

    # 每维指标（用于指定维度平均与 per_dim 输出）
    mse_per = torch.mean((pred - target) ** 2, dim=0)  # [C]
    mae_per = torch.mean(torch.abs(pred - target), dim=0)  # [C]
    ccc_per = concordance_correlation_coefficient(pred, target, dim=0)
    if ccc_per.dim() == 0:
        ccc_per = ccc_per.unsqueeze(0).expand(C)
    human_sim_per = human_similarity_from_ccc(ccc_per).cpu().tolist()

    idx = dim_indices if dim_indices is not None else list(range(C))
    if idx:
        out["mse_specified"] = float(torch.mean(mse_per[idx]).item())
        out["mae_specified"] = float(torch.mean(mae_per[idx]).item())
        valid_idx = [i for i in idx if i < len(human_sim_per)]
        if valid_idx:
            sim_tensor = human_similarity_from_ccc(ccc_per[valid_idx])
            finite_mask = torch.isfinite(sim_tensor)
            out["human_sim_specified"] = (
                float(torch.mean(sim_tensor[finite_mask]).item()) if finite_mask.any() else float("nan")
            )
        else:
            out["human_sim_specified"] = float("nan")

    if return_per_dim:
        out["mse_per_dim"] = mse_per.cpu().tolist()
        out["mae_per_dim"] = mae_per.cpu().tolist()
        out["human_sim_per_dim"] = human_sim_per
        if label_names is not None and len(label_names) >= C:
            out["mse_per_dim_named"] = {label_names[i]: mse_per[i].item() for i in range(C)}
            out["mae_per_dim_named"] = {label_names[i]: mae_per[i].item() for i in range(C)}
            out["human_sim_per_dim_named"] = {label_names[i]: human_sim_per[i] for i in range(C)}

    return out
