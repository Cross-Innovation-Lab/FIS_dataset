"""评估入口：加载 checkpoint 并输出 MSE/MAE/与真人相似度；可选指定维度平均与每维指标、保存逐样本预测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiment.config import load_config
from experiment.dataloader import LABEL_COLUMNS
from experiment.model import build_model
from experiment.train import build_loaders, run_eval, run_eval_collect_predictions, save_predictions_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径（best.pt）")
    p.add_argument("--config", type=str, default=None, help="可选：覆盖配置 JSON")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--dim-indices",
        type=str,
        default=None,
        help="指定参与平均的维度下标，逗号分隔，如 0,4,7；不传则使用全部维度",
    )
    p.add_argument("--per-dim", action="store_true", help="是否输出并保存每维指标（mse/mae/human_sim per label）")
    p.add_argument("--predictions-out", type=str, default=None, help="可选：保存逐样本预测的 CSV 路径，便于检查过拟合")
    p.add_argument("--device", type=str, default=None, help="覆盖配置中的 device，如 cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 优先从 ckpt 中恢复配置，命令行 --config 可覆盖
    if args.config:
        cfg = load_config(args.config)
    elif "config" in ckpt and isinstance(ckpt["config"], str):
        import dataclasses
        cfg = load_config(None)
        cfg_dict = json.loads(ckpt["config"])
        for section_name, section_val in cfg_dict.items():
            section = getattr(cfg, section_name, None)
            if section and dataclasses.is_dataclass(section) and isinstance(section_val, dict):
                for k, v in section_val.items():
                    if hasattr(section, k):
                        setattr(section, k, v)
    else:
        cfg = load_config(None)

    model = build_model(cfg.model.name, cfg.model.kwargs)
    model.load_state_dict(ckpt["model"], strict=False)
    device_str = args.device if args.device is not None else cfg.train.device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, test_loader = build_loaders(cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    dim_indices = None
    if args.dim_indices:
        dim_indices = [int(x.strip()) for x in args.dim_indices.split(",")]
    metrics = run_eval(
        model,
        loader,
        device,
        label_names=LABEL_COLUMNS,
        dim_indices=dim_indices,
        return_per_dim=args.per_dim,
    )
    print(f"[{args.split}] {metrics}")

    out = ckpt_path.parent / f"{args.split}_metrics_eval.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out}")

    if args.predictions_out:
        pred_path = Path(args.predictions_out)
        ids, y_true, y_pred = run_eval_collect_predictions(model, loader, device)
        if len(ids) > 0:
            save_predictions_csv(pred_path, ids, y_true, y_pred)
            print(f"predictions saved: {pred_path} (n={len(ids)})")
        else:
            print("warning: no valid predictions, skipped writing CSV")


if __name__ == "__main__":
    main()
