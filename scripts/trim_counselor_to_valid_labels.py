#!/usr/bin/env python3
"""
基于 dataset/all_labels_Valid.csv 中的当前记录，删除 dataset/FIS_dataset/Counselor 下
无对应标签的特征文件，使 Counselor 目录与 CSV 记录一致。

CSV ID 格式: XXX_FIS_TimeN_Name 或 XXX_FIs_TimeN_Name（拼写变体）
Counselor 文件命名: XXX_TN_Name.<ext>（如 .npz）

用法:
  python scripts/trim_counselor_to_valid_labels.py [--dry-run]
  python scripts/trim_counselor_to_valid_labels.py --execute
  --dry-run: 仅列出将要删除的文件，不实际删除（默认）。
  --execute: 实际执行删除。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "dataset" / "all_labels_Valid.csv"
COUNSELOR_DIR = PROJECT_ROOT / "dataset" / "FIS_dataset" / "Counselor"


def id_to_stem(row_id: str) -> str:
    """将 CSV 的 ID 转为 Counselor 文件名 stem。支持 FIS_Time 与 FIs_Time。"""
    s = row_id.strip()
    for prefix in ("FIS_Time", "FIs_Time"):
        if prefix in s:
            s = s.replace(prefix, "T", 1)
            break
    return s


def collect_valid_stems(csv_path: Path) -> set[str]:
    """从 CSV 收集所有有效 stem（当前标签表中的样本名）。"""
    stems: set[str] = set()
    if not csv_path.exists():
        return stems
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_val = (row.get("ID") or "").strip()
            if id_val:
                stems.add(id_to_stem(id_val))
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 all_labels_Valid.csv 当前记录删除 Counselor 中多余的特征文件。"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际执行删除；默认仅列出将要删除的文件（dry-run）。",
    )
    args = parser.parse_args()
    do_delete = args.execute

    valid_stems = collect_valid_stems(CSV_PATH)
    print(f"CSV 中有效 ID 数（stem 数）: {len(valid_stems)}")

    if not COUNSELOR_DIR.exists():
        print(f"Counselor 目录不存在: {COUNSELOR_DIR}")
        return

    to_remove: list[Path] = []
    for path in COUNSELOR_DIR.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem
        if stem not in valid_stems:
            to_remove.append(path)

    only_in_csv = sorted(valid_stems - {p.stem for p in COUNSELOR_DIR.rglob("*") if p.is_file()})
    if only_in_csv:
        print(f"仅出现在 CSV、未出现在 Counselor 中的 stem 数: {len(only_in_csv)}")

    to_remove.sort(key=lambda p: (p.relative_to(COUNSELOR_DIR).as_posix(),))
    print(f"无对应 CSV 记录、待删除文件数: {len(to_remove)}")

    if not to_remove:
        print("没有需要删除的文件。")
        return

    if do_delete:
        for p in to_remove:
            try:
                p.unlink()
                print(f"已删除: {p.relative_to(COUNSELOR_DIR)}")
            except OSError as e:
                print(f"删除失败 {p}: {e}")
        print(f"共删除 {len(to_remove)} 个文件。")
    else:
        print("【dry-run】以下文件将被删除（使用 --execute 执行）:")
        for p in to_remove[:50]:
            print(f"  {p.relative_to(COUNSELOR_DIR)}")
        if len(to_remove) > 50:
            print(f"  ... 及另外 {len(to_remove) - 50} 个文件")


if __name__ == "__main__":
    main()
