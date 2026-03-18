#!/usr/bin/env python3
"""Compare experiment results across multiple runs.

Usage:
    python scripts/compare_results.py <run_dir1> <run_dir2> ...
    python scripts/compare_results.py outs-baselines/tcdyfis_v2_task2_*

Reads test_metrics.json, config.json, and history.json from each run directory,
then prints a comparison table sorted by test human_sim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_run_info(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "test_metrics.json"
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"

    if not metrics_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics.get("mse"), (int, float)):
        return None

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    history = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    model_kwargs = config.get("model", {}).get("kwargs", {})
    train_cfg = config.get("train", {})

    best_val_hsim = 0.0
    best_epoch = 0
    for row in history:
        hsim = row.get("val_human_sim_specified", row.get("val_human_sim", 0.0))
        if isinstance(hsim, (int, float)) and hsim > best_val_hsim:
            best_val_hsim = hsim
            best_epoch = row.get("epoch", 0)

    return {
        "run_name": run_dir.name,
        "model": config.get("model", {}).get("name", "?"),
        "task": config.get("data", {}).get("task", "?"),
        "d_model": model_kwargs.get("d_model", "?"),
        "compressed_len": model_kwargs.get("compressed_len", "?"),
        "dropout": model_kwargs.get("dropout", "?"),
        "dyadic_layers": model_kwargs.get("dyadic_layers", "?"),
        "input_noise": model_kwargs.get("input_noise_std", 0),
        "lr": train_cfg.get("lr", "?"),
        "weight_decay": train_cfg.get("weight_decay", "?"),
        "ccc_weight": train_cfg.get("ccc_loss_weight", "?"),
        "batch_size": train_cfg.get("batch_size", "?"),
        "epochs": train_cfg.get("epochs", "?"),
        "n_params": _count_params(config_path),
        "best_epoch": best_epoch,
        "best_val_hsim": best_val_hsim,
        "test_mse": metrics.get("mse", float("nan")),
        "test_mae": metrics.get("mae", float("nan")),
        "test_hsim": metrics.get("human_sim", float("nan")),
        "test_mse_spec": metrics.get("mse_specified", float("nan")),
        "test_mae_spec": metrics.get("mae_specified", float("nan")),
        "test_hsim_spec": metrics.get("human_sim_specified", float("nan")),
    }


def _count_params(config_path: Path) -> str:
    log_path = config_path.parent / "train.log"
    if not log_path.exists():
        return "?"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "模型参数量" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                return parts[-1].strip()
    return "?"


def print_comparison_table(runs: list[dict]) -> None:
    runs = sorted(runs, key=lambda r: r.get("test_hsim", 0), reverse=True)

    print("\n" + "=" * 120)
    print("  EXPERIMENT RESULTS COMPARISON")
    print("=" * 120)

    header = (
        f"{'Run Name':<30s} {'Model':<12s} {'T':>1s} "
        f"{'d_m':>4s} {'CL':>3s} {'drop':>5s} {'lr':>8s} {'wd':>6s} "
        f"{'ccc_w':>5s} {'noise':>5s} {'#Param':>10s} "
        f"{'BstEp':>5s} {'ValHS':>6s} "
        f"{'MSE':>7s} {'MAE':>7s} {'H_Sim':>7s}"
    )
    print(header)
    print("-" * 120)

    for r in runs:
        lr_str = f"{r['lr']:.1e}" if isinstance(r['lr'], float) else str(r['lr'])
        wd_str = f"{r['weight_decay']}" if isinstance(r['weight_decay'], (int, float)) else str(r['weight_decay'])
        noise_str = f"{r['input_noise']}" if isinstance(r['input_noise'], (int, float)) else "0"

        print(
            f"{r['run_name']:<30s} {str(r['model']):<12s} {str(r['task']):>1s} "
            f"{str(r['d_model']):>4s} {str(r['compressed_len']):>3s} "
            f"{str(r['dropout']):>5s} {lr_str:>8s} {wd_str:>6s} "
            f"{str(r['ccc_weight']):>5s} {noise_str:>5s} {str(r['n_params']):>10s} "
            f"{r['best_epoch']:>5d} {r['best_val_hsim']:>6.4f} "
            f"{r['test_mse']:>7.4f} {r['test_mae']:>7.4f} {r['test_hsim']:>7.4f}"
        )

    print("-" * 120)

    if any(r.get("test_hsim_spec") and r["test_hsim_spec"] != r["test_hsim"] for r in runs):
        print("\n  Specified-dimension metrics:")
        print(f"  {'Run Name':<30s} {'MSE_spec':>9s} {'MAE_spec':>9s} {'HSim_spec':>9s}")
        print(f"  {'-'*30} {'-'*9} {'-'*9} {'-'*9}")
        for r in runs:
            print(
                f"  {r['run_name']:<30s} "
                f"{r['test_mse_spec']:>9.4f} {r['test_mae_spec']:>9.4f} {r['test_hsim_spec']:>9.4f}"
            )

    print("\n" + "=" * 120)

    best = runs[0] if runs else None
    if best:
        print(f"\n  Best run: {best['run_name']}")
        print(f"    test_human_sim = {best['test_hsim']:.4f}")
        print(f"    test_mse       = {best['test_mse']:.4f}")
        print(f"    best_epoch     = {best['best_epoch']}")
        print(f"    params         = {best['n_params']}")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_results.py <run_dir1> [run_dir2] ...")
        print("       python scripts/compare_results.py outs-baselines/tcdyfis_v2_*")
        sys.exit(1)

    run_dirs = [Path(p) for p in sys.argv[1:]]
    runs = []
    for d in run_dirs:
        if not d.is_dir():
            continue
        info = load_run_info(d)
        if info is not None:
            runs.append(info)

    if not runs:
        print("No valid runs found.")
        sys.exit(0)

    print_comparison_table(runs)


if __name__ == "__main__":
    main()
