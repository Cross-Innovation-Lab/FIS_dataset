"""FIS 实验训练入口：MSE 回归，带日志文件写入与 checkpoint 管理。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiment.config import Config, load_config
from experiment.dataloader import FISDataset, LABEL_COLUMNS, collate_fis_batch, normalize_sample_id
from experiment.metrics import concordance_correlation_coefficient, regression_metrics

# 以下指标越大越好，用于 save_best_by 时取最大值；其余（如 mse/mae）越小越好
HIGHER_IS_BETTER_KEYS = {"human_sim", "human_sim_specified"}

from experiment.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_split_spec(split_spec: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in str(split_spec).split(":")]
    if len(parts) != 3:
        raise ValueError(f"split_spec 格式错误: {split_spec}，期望形如 16:4:5")
    train_u, val_u, test_u = (int(p) for p in parts)
    if min(train_u, val_u, test_u) < 0 or (train_u + val_u + test_u) <= 0:
        raise ValueError(f"split_spec 非法: {split_spec}")
    return train_u, val_u, test_u


def _allocate_split_counts(n: int, split_spec: str) -> dict[str, int]:
    train_u, val_u, test_u = _parse_split_spec(split_spec)
    units = {"train": train_u, "val": val_u, "test": test_u}
    total_units = sum(units.values())
    raw = {name: n * unit / total_units for name, unit in units.items()}
    counts = {name: int(np.floor(val)) for name, val in raw.items()}
    remain = n - sum(counts.values())
    order = sorted(
        units.keys(),
        key=lambda name: (raw[name] - counts[name], units[name], name),
        reverse=True,
    )
    for i in range(remain):
        counts[order[i % len(order)]] += 1
    return counts


def make_reproducible_split_ids(
    sample_ids: list[str], split_spec: str, seed: int, group_by: str = "none",
) -> dict[str, list[str]]:
    clean_ids = [str(sample_id).strip() for sample_id in sample_ids]
    normalized_ids = [normalize_sample_id(sample_id) for sample_id in clean_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("检测到重复 sample_id（归一化后），无法生成可复现划分")

    mode = str(group_by or "none").strip().lower()
    rng = random.Random(seed)
    counts = _allocate_split_counts(len(clean_ids), split_spec)

    if mode in ("", "none"):
        shuffled_ids = sorted(clean_ids)
        rng.shuffle(shuffled_ids)
        n_train = counts["train"]
        n_val = counts["val"]
        return {
            "train": shuffled_ids[:n_train],
            "val": shuffled_ids[n_train:n_train + n_val],
            "test": shuffled_ids[n_train + n_val:],
        }

    def infer_group_id(sample_id: str) -> str:
        sid = normalize_sample_id(sample_id)
        parts = sid.split("_")
        if mode in {"session_prefix", "prefix", "counselor_prefix"}:
            return parts[0] if parts else sid
        if mode in {"session_time", "prefix_time"}:
            return "_".join(parts[:3]) if len(parts) >= 3 else sid
        raise ValueError(f"不支持的 split_group_by: {group_by}")

    grouped_ids: dict[str, list[str]] = {}
    for sample_id in clean_ids:
        grouped_ids.setdefault(infer_group_id(sample_id), []).append(sample_id)

    group_items = [
        (group_id, sorted(ids), rng.random())
        for group_id, ids in grouped_ids.items()
    ]
    group_items.sort(key=lambda item: (-len(item[1]), item[2], item[0]))

    split_ids = {"train": [], "val": [], "test": []}
    assigned_counts = {name: 0 for name in split_ids}
    split_order = ("train", "val", "test")

    for _, ids, _ in group_items:
        size = len(ids)
        best_split = min(
            split_order,
            key=lambda name: (
                max(0, assigned_counts[name] + size - counts[name]),
                abs(counts[name] - (assigned_counts[name] + size)),
                assigned_counts[name] / max(counts[name], 1),
                name,
            ),
        )
        split_ids[best_split].extend(ids)
        assigned_counts[best_split] += size

    return split_ids


def save_split_manifest(
    path: Path,
    split_ids: dict[str, list[str]],
    *,
    split_spec: str,
    seed: int,
    csv_path: str,
    group_by: str = "none",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_spec": split_spec,
        "seed": seed,
        "csv_path": str(csv_path),
        "group_by": group_by,
        "counts": {name: len(ids) for name, ids in split_ids.items()},
        "splits": split_ids,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_split_manifest(path: Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = payload.get("splits", {})
    if not all(name in splits for name in ("train", "val", "test")):
        raise ValueError(f"划分文件缺少 train/val/test: {path}")
    return {name: [str(x).strip() for x in splits[name]] for name in ("train", "val", "test")}


def split_ids_to_indices(dataset: FISDataset, split_ids: dict[str, list[str]]) -> tuple[list[int], list[int], list[int]]:
    dataset_ids = dataset.df["ID"].astype(str).str.strip().tolist()
    id_to_idx = {normalize_sample_id(sample_id): idx for idx, sample_id in enumerate(dataset_ids)}
    if len(id_to_idx) != len(dataset_ids):
        raise ValueError("数据集中的 ID 归一化后不唯一，无法按划分文件映射索引")
    missing = [
        sample_id
        for ids in split_ids.values()
        for sample_id in ids
        if normalize_sample_id(sample_id) not in id_to_idx
    ]
    if missing:
        raise ValueError(f"划分文件中存在当前数据集没有的样本 ID，例如: {missing[:5]}")
    return (
        [id_to_idx[normalize_sample_id(sample_id)] for sample_id in split_ids["train"]],
        [id_to_idx[normalize_sample_id(sample_id)] for sample_id in split_ids["val"]],
        [id_to_idx[normalize_sample_id(sample_id)] for sample_id in split_ids["test"]],
    )


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = to_device(v, device)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    label_names: list[str] | None = None,
    dim_indices: list[int] | None = None,
    return_per_dim: bool = False,
) -> dict[str, float | list[float] | dict[str, float]]:
    """评估模型，返回 MSE / MAE / human_sim；可选指定维度平均与每维指标。"""
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        batch = to_device(batch, device)
        if batch["counselor_word_mask"].sum() == 0:
            continue
        try:
            y_hat = model(batch)
        except (ValueError, ZeroDivisionError, RuntimeError):
            continue
        preds.append(y_hat.detach().cpu())
        tgts.append(batch["labels"].detach().cpu())
    if not preds:
        base = {"mse": float("nan"), "mae": float("nan"), "human_sim": float("nan")}
        if dim_indices:
            base["mse_specified"] = base["mae_specified"] = base["human_sim_specified"] = float("nan")
        return base
    return regression_metrics(
        torch.cat(preds, 0),
        torch.cat(tgts, 0),
        label_names=label_names,
        dim_indices=dim_indices,
        return_per_dim=return_per_dim,
    )


@torch.no_grad()
def run_eval_collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """跑一遍 loader，收集每条样本的 sample_id、真实标签、预测值。用于保存预测结果以检查过拟合等。

    Returns:
        sample_ids: 长度为 N 的 list[str]
        y_true: [N, n_labels] float
        y_pred: [N, n_labels] float
    """
    model.eval()
    ids_list: list[str] = []
    preds_list: list[torch.Tensor] = []
    tgts_list: list[torch.Tensor] = []
    for batch in loader:
        batch = to_device(batch, device)
        if batch["counselor_word_mask"].sum() == 0:
            continue
        try:
            y_hat = model(batch)
        except (ValueError, ZeroDivisionError, RuntimeError):
            continue
        sid = batch["sample_id"]
        ids_list.extend(sid)
        preds_list.append(y_hat.detach().cpu())
        tgts_list.append(batch["labels"].detach().cpu())
    if not ids_list:
        return [], np.array([]), np.array([])
    y_true = torch.cat(tgts_list, 0).numpy()
    y_pred = torch.cat(preds_list, 0).numpy()
    return ids_list, y_true, y_pred


def save_predictions_csv(
    path: Path,
    sample_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_columns: list[str] | None = None,
) -> None:
    """将 sample_id、真实标签、预测值写入 CSV，便于后续分析过拟合与误差分布。"""
    cols = label_columns or LABEL_COLUMNS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_labels = y_true.shape[1]
    header = ["sample_id"] + [f"true_{c}" for c in cols[:n_labels]] + [f"pred_{c}" for c in cols[:n_labels]]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, sid in enumerate(sample_ids):
            row = [sid] + y_true[i].tolist() + y_pred[i].tolist()
            w.writerow(row)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
    ccc_loss_weight: float = 0.0,
    *,
    log_skip_diagnostic: bool = False,
) -> float:
    model.train()
    losses: list[float] = []
    n_batches = 0
    n_skipped_mask = 0
    n_skipped_error = 0
    for batch in loader:
        batch = to_device(batch, device)
        n_batches += 1
        # 跳过全无效序列的 batch（否则 Mamba/AttentivePooling 会报错或产生 NaN）
        if batch["counselor_word_mask"].sum() == 0:
            n_skipped_mask += 1
            continue
        try:
            y_hat = model(batch)
        except (ValueError, ZeroDivisionError, RuntimeError):
            n_skipped_error += 1
            continue
        mse_loss = F.mse_loss(y_hat, batch["labels"])
        loss = mse_loss
        if ccc_loss_weight > 0:
            ccc = concordance_correlation_coefficient(y_hat, batch["labels"], dim=0)
            if ccc.dim() == 0:
                ccc = ccc.unsqueeze(0)
            finite_mask = torch.isfinite(ccc)
            if finite_mask.any():
                loss = loss + ccc_loss_weight * (1.0 - ccc[finite_mask].mean())
        if not torch.isfinite(loss):
            n_skipped_error += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(mse_loss.item())
    if log_skip_diagnostic and n_batches > 0:
        import logging
        logger = logging.getLogger("fis_train")
        logger.warning(
            "train_one_epoch 诊断: 总 batch=%d, 因 word_mask 全 0 跳过=%d, 因异常/非有限 loss 跳过=%d, 有效 step=%d",
            n_batches, n_skipped_mask, n_skipped_error, len(losses),
        )
    return float(np.mean(losses)) if losses else float("nan")


def build_loaders(cfg: Config, split_output_path: Path | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = FISDataset(
        csv_path=cfg.data.csv_path,
        feature_root=cfg.data.feature_root,
        task=cfg.data.task,
        counselor_role=cfg.data.counselor_role,
        patient_role=cfg.data.patient_role,
        feature_source=getattr(cfg.data, "feature_source", "raw"),
        feature_categories=getattr(cfg.data, "feature_categories", None),
        valid_id_csv=getattr(cfg.data, "avalid_csv", None),
        patient_basename_map=Path(cfg.data.patient_basename_map_csv) if getattr(cfg.data, "patient_basename_map_csv", None) else None,
    )
    split_file = getattr(cfg.train, "split_file", None)
    split_path = Path(split_file).expanduser() if split_file else (
        Path(split_output_path).expanduser() if split_output_path is not None else None
    )
    if split_path is not None and split_path.exists():
        split_ids = load_split_manifest(split_path)
        cfg.train.split_file = str(split_path.resolve())
    else:
        split_ids = make_reproducible_split_ids(
            dataset.df["ID"].astype(str).str.strip().tolist(),
            getattr(cfg.train, "split_spec", "16:4:5"),
            cfg.train.seed,
            getattr(cfg.train, "split_group_by", "none"),
        )
        if split_path is not None:
            save_split_manifest(
                split_path,
                split_ids,
                split_spec=getattr(cfg.train, "split_spec", "16:4:5"),
                seed=cfg.train.seed,
                csv_path=cfg.data.csv_path,
                group_by=getattr(cfg.train, "split_group_by", "none"),
            )
            cfg.train.split_file = str(split_path.resolve())
    train_idx, val_idx, test_idx = split_ids_to_indices(dataset, split_ids)
    collate = lambda b: collate_fis_batch(
        b, max_len_word=cfg.data.max_len_word, max_len_tok=cfg.data.max_len_tok,
    )
    kw = dict(num_workers=cfg.data.num_workers, collate_fn=collate)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=cfg.train.batch_size, shuffle=True, **kw,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=cfg.train.batch_size, shuffle=False, **kw,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=cfg.train.batch_size, shuffle=False, **kw,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
#  日志与路径
# ---------------------------------------------------------------------------

def _resolve_run_name(cfg: Config) -> str:
    if cfg.experiment.run_name:
        return cfg.experiment.run_name
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{cfg.model.name}_{ts}"


def setup_logging(log_path: Path) -> logging.Logger:
    """配置同时输出到文件和终端的 logger。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fis_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
#  训练主流程
# ---------------------------------------------------------------------------

def run_training(cfg: Config) -> Path:
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    run_name = _resolve_run_name(cfg)

    # ---- 目录 ----
    out_dir = Path(cfg.experiment.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.experiment.ckpt_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- 日志 ----
    logger = setup_logging(out_dir / "train.log")
    logger.info(f"run_name   : {run_name}")
    logger.info(f"out_dir    : {out_dir}")
    logger.info(f"ckpt_dir   : {ckpt_dir}")
    logger.info(f"device     : {device}")
    logger.info(f"model      : {cfg.model.name}")

    # ---- 构建模型与数据 ----
    model = build_model(cfg.model.name, cfg.model.kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量 : {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    split_output_path = (
        Path(getattr(cfg.train, "split_file", "")).expanduser()
        if getattr(cfg.train, "split_file", None)
        else (out_dir / "data_split.json")
    )
    train_loader, val_loader, test_loader = build_loaders(cfg, split_output_path=split_output_path)
    # 划分文件生成后再保存配置快照，便于评估时严格复用同一 train/val/test 样本集合。
    cfg_json = json.dumps(cfg, default=lambda o: o.__dict__, indent=2, ensure_ascii=False)
    (out_dir / "config.json").write_text(cfg_json, encoding="utf-8")
    logger.info(f"config saved to {out_dir / 'config.json'}")
    logger.info(f"数据集大小 : train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    logger.info(f"数据划分   : split_spec={getattr(cfg.train, 'split_spec', '16:4:5')}, seed={cfg.train.seed}")
    logger.info(f"划分分组   : split_group_by={getattr(cfg.train, 'split_group_by', 'none')}")
    if getattr(cfg.train, "split_file", None):
        logger.info(f"划分文件   : {cfg.train.split_file}")
    logger.info(f"损失权重   : ccc_loss_weight={getattr(cfg.train, 'ccc_loss_weight', 0.0)}")

    # ---- 训练循环 ----
    save_best_key = cfg.experiment.save_best_by
    metric_key = save_best_key.replace("val_", "", 1) if save_best_key.startswith("val_") else save_best_key
    higher_better = metric_key in HIGHER_IS_BETTER_KEYS
    best_metric = float("-inf") if higher_better else float("inf")
    best_epoch = 0
    best_ckpt = ckpt_dir / "best.pt"
    history: list[dict] = []

    logger.info(f"开始训练: epochs={cfg.train.epochs}, batch_size={cfg.train.batch_size}, lr={cfg.train.lr}")
    logger.info("-" * 80)

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, opt, device, grad_clip=cfg.train.grad_clip,
            ccc_loss_weight=getattr(cfg.train, "ccc_loss_weight", 0.0),
            log_skip_diagnostic=(epoch == 1),
        )
        val_metrics = run_eval(model, val_loader, device)

        # 写入所有标量指标（含 mse_specified / human_sim_specified 等）；跳过 list/dict 以保持 history 精简
        row = {"epoch": epoch, "train_mse": train_loss}
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                row[f"val_{k}"] = float(v)
        history.append(row)

        is_best = ""
        metric_val = val_metrics.get(metric_key, val_metrics["mae"])
        is_improved = (higher_better and metric_val > best_metric) or (not higher_better and metric_val < best_metric)
        if np.isfinite(metric_val) and is_improved:
            best_metric = metric_val
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg_json}, best_ckpt)
            is_best = " ★ best"

        logger.info(
            f"[Epoch {epoch:03d}/{cfg.train.epochs:03d}] "
            f"train_mse={train_loss:.6f}  "
            f"val_mse={val_metrics['mse']:.6f}  val_mae={val_metrics['mae']:.6f}  "
            f"val_human_sim={val_metrics.get('human_sim', float('nan')):.4f}"
            f"{is_best}"
        )
        if any(k.endswith("_specified") for k in val_metrics):
            logger.info(
                "  val_specified: " + "  ".join(
                    f"{k}={val_metrics[k]:.4f}" for k in ["mse_specified", "mae_specified", "human_sim_specified"]
                    if k in val_metrics
                )
            )

    logger.info("-" * 80)

    # ---- 保存最后一个 epoch 的 checkpoint ----
    torch.save({"epoch": cfg.train.epochs, "model": model.state_dict(), "config": cfg_json}, ckpt_dir / "last.pt")

    # ---- 保存训练历史 ----
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 测试集评估 ----
    if best_ckpt.exists():
        ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model"])
        logger.info(f"加载最优 checkpoint (epoch {best_epoch}) 进行测试集评估")
    test_metrics = run_eval(model, test_loader, device)
    logger.info(
        f"test_mse={test_metrics['mse']:.6f}  test_mae={test_metrics['mae']:.6f}  "
        f"test_human_sim={test_metrics.get('human_sim', float('nan')):.4f}"
    )
    if any(k.endswith("_specified") for k in test_metrics):
        logger.info(
            "  test_specified: " + "  ".join(
                f"{k}={test_metrics[k]:.4f}" for k in ["mse_specified", "mae_specified", "human_sim_specified"]
                if k in test_metrics
            )
        )

    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ---- 保存 train/val/test 的逐样本预测，便于检查过拟合与误差分布 ----
    for split_name, split_loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        ids, yt, yp = run_eval_collect_predictions(model, split_loader, device)
        if len(ids) > 0:
            save_predictions_csv(out_dir / f"predictions_{split_name}.csv", ids, yt, yp)
            logger.info(f"已保存 {split_name} 预测: {out_dir / f'predictions_{split_name}.csv'} (n={len(ids)})")
        else:
            logger.warning(f"{split_name} 无有效预测，未写入 CSV")

    logger.info(f"训练完成。best_epoch={best_epoch}, best_{metric_key}={best_metric:.6f}")
    logger.info(f"best_ckpt : {best_ckpt}")
    logger.info(f"last_ckpt : {ckpt_dir / 'last.pt'}")
    logger.info(f"日志文件  : {out_dir / 'train.log'}")
    return out_dir


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    p.add_argument("--task", type=int, default=None, choices=[1, 2], help="任务 1 或 2（覆盖 config 中的 data.task）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.task is not None:
        cfg.data.task = args.task
    run_training(cfg)


if __name__ == "__main__":
    main()
