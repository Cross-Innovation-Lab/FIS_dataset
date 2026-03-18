"""
单 batch 前向检查：用 task1 的 config 和 FIS-Net 跑一个 batch，打印 y_hat / loss 是否含 NaN。
用于在「数据正常、无 batch 被 word_mask 跳过」时排查 NaN 是否来自模型前向或 loss。

用法:
  cd /CIL_PROJECTS/CODES/MM_FIS && python -m experiment.check_nan_forward
  或指定 config: python -m experiment.check_nan_forward --config outs/fisnet_task1/config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment.config import load_config
from experiment.dataloader import FISDataset, collate_fis_batch
from experiment.model import build_model
from experiment.train import to_device
from torch.utils.data import DataLoader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="outs/fisnet_task1/config.json", help="实验 config.json 路径")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据：与 train 一致
    ds = FISDataset(
        csv_path=cfg.data.csv_path,
        feature_root=cfg.data.feature_root,
        task=cfg.data.task,
        counselor_role=cfg.data.counselor_role,
        patient_role=cfg.data.patient_role,
        valid_id_csv=getattr(cfg.data, "avalid_csv", None),
        patient_basename_map=Path(cfg.data.patient_basename_map_csv) if getattr(cfg.data, "patient_basename_map_csv", None) else None,
    )
    collate = lambda b: collate_fis_batch(b, max_len_word=cfg.data.max_len_word, max_len_tok=cfg.data.max_len_tok)
    loader = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=False, collate_fn=collate)

    # 取第一个有效 batch（word_mask 非全 0）
    batch = None
    for b in loader:
        if b["counselor_word_mask"].sum() > 0:
            batch = b
            break
    if batch is None:
        print("未找到 counselor_word_mask.sum()>0 的 batch，请先确认数据。")
        return

    batch = to_device(batch, device)

    # 模型
    model = build_model(cfg.model.name, cfg.model.kwargs).to(device)
    model.eval()

    # 前向
    with torch.no_grad():
        y_hat = model(batch)

    labels = batch["labels"]
    loss = F.mse_loss(y_hat, labels)

    # 诊断
    y_hat_nan = torch.isnan(y_hat).any().item() or torch.isinf(y_hat).any().item()
    labels_nan = torch.isnan(labels).any().item() or torch.isinf(labels).any().item()
    loss_finite = torch.isfinite(loss).item()

    print("--- 单 batch 前向检查 ---")
    print(f"  y_hat: shape={tuple(y_hat.shape)}, 含 NaN/Inf={y_hat_nan}, min={y_hat.min().item():.6f}, max={y_hat.max().item():.6f}")
    print(f"  labels: 含 NaN/Inf={labels_nan}, min={labels.min().item():.6f}, max={labels.max().item():.6f}")
    print(f"  loss: 有限={loss_finite}, value={loss.item()}")
    if y_hat_nan or not loss_finite:
        print("  -> 问题来自模型输出或 loss 非有限，需检查模型初始化/数值稳定性。")
    else:
        print("  -> 本 batch 前向与 loss 正常；若训练仍 NaN，可能是某些 batch 或 train 模式下的 dropout 等导致。")


if __name__ == "__main__":
    main()
