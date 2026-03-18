#!/usr/bin/env python3
"""将 63 个「无 AU」且出现在 all_labels_Valid.csv 中的 Nov2021 视频复制到 dataset/testset。

用法:
  python scripts/copy_valid_no_au_videos_to_testset.py [--dry-run]

依赖:
  - preprocess/FIS_FEA/Counselor_video_openface3_nov2021_dim200_stems.txt（83 个无 AU stem）
  - dataset/all_labels_Valid.csv（有效标签 ID）
  - dataset/Nov2021/*.mp4（源视频）
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def stem_to_label_id(stem: str) -> str:
    """视频 stem (CODE_T1_Name) -> 标签 CSV ID (CODE_FIS_Time1_Name)。"""
    s = stem.replace("_T1_", "_FIS_Time1_").replace("_T2_", "_FIS_Time2_").replace("_T3_", "_FIS_Time3_")
    if "_T1" in s or "_T2" in s or "_T3" in s:
        s = s.replace("_T1", "_FIS_Time1").replace("_T2", "_FIS_Time2").replace("_T3", "_FIS_Time3")
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="复制 63 个 Valid 中且无 AU 的视频到 dataset/testset")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要复制的文件，不实际复制")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dim200_file = root / "preprocess" / "FIS_FEA" / "Counselor_video_openface3_nov2021_dim200_stems.txt"
    valid_csv = root / "dataset" / "all_labels_Valid.csv"
    nov2021_dir = root / "dataset" / "Nov2021"
    testset_dir = root / "dataset" / "testset"

    if not dim200_file.exists():
        print(f"错误: 未找到 {dim200_file}", file=sys.stderr)
        sys.exit(1)
    if not valid_csv.exists():
        print(f"错误: 未找到 {valid_csv}", file=sys.stderr)
        sys.exit(1)
    if not nov2021_dir.is_dir():
        print(f"错误: 未找到目录 {nov2021_dir}", file=sys.stderr)
        sys.exit(1)

    # 1) 83 个无 AU stem
    stems_200 = [line.strip() for line in dim200_file.read_text(encoding="utf-8").strip().splitlines() if line.strip()]

    # 2) Valid 中的 ID
    valid_ids = set()
    with open(valid_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            valid_ids.add(row["ID"].strip())
            if "FIs_" in row["ID"]:
                valid_ids.add(row["ID"].strip().replace("FIs_", "FIS_"))

    # 3) 在 Valid 中的无 AU stem（63 个）
    stems_in_valid = [s for s in stems_200 if stem_to_label_id(s) in valid_ids]
    if len(stems_in_valid) != 63:
        print(f"提示: 预期 63 个，实际 {len(stems_in_valid)} 个在 Valid 中。", file=sys.stderr)

    # 4) 在 Nov2021 下查找对应视频（优先精确 stem.mp4，否则同名前缀）
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    to_copy: list[tuple[Path, Path]] = []
    missing: list[str] = []

    for stem in stems_in_valid:
        src = nov2021_dir / f"{stem}.mp4"
        if not src.exists():
            candidates = list(nov2021_dir.glob(f"{stem}*"))
            src = next((p for p in candidates if p.suffix.lower() in video_exts), None)
        if src is None or not src.exists():
            missing.append(stem)
            continue
        to_copy.append((src, testset_dir / src.name))

    if missing:
        print(f"警告: 以下 {len(missing)} 个 stem 在 Nov2021 下未找到视频:", file=sys.stderr)
        for s in missing:
            print(f"  {s}", file=sys.stderr)

    if not to_copy:
        print("没有可复制的文件。", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"（dry-run）将复制 {len(to_copy)} 个视频到 {testset_dir}:")
        for src, dst in to_copy:
            print(f"  {src} -> {dst}")
        return

    testset_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in to_copy:
        shutil.copy2(src, dst)
        print(f"已复制: {src.name} -> {dst}")
    print(f"\n共复制 {len(to_copy)} 个文件到 {testset_dir}")


if __name__ == "__main__":
    main()
