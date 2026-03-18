#!/usr/bin/env python3
"""检查 Nov2021/Oct2020 下：1) 全黑视频 2) OpenFace AU CSV 异常（缺失、空/少行、success 率低、AU 全零/无效）。"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# 与 openface3.py 一致
def _is_au_column(name: str) -> bool:
    return bool(
        re.match(r"^AU\d{2}_[rc]$", name)
        or re.match(r"^AU\d+_[rc]$", name)
        or re.match(r"^AU_0\d+$", name)
    )


def resolve_au_csv(stem: str, au_roots: list[Path]) -> Path | None:
    """与 openface3._resolve_au_csv 一致。"""
    for root in au_roots:
        for p in (root / stem / f"{stem}.csv", root / f"{stem}.csv"):
            if p.exists():
                return p
    return None


def is_corrupt_ffprobe(path: Path) -> bool:
    """ffprobe 无法打开或无视频流则视为损坏。"""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return r.returncode != 0 or "video" not in (r.stdout or "")


def mean_brightness(path: Path, frame_indices: list[int]) -> float | None:
    """采样帧平均亮度，失败返回 None。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    brightness_sum = 0.0
    count = 0
    for idx in frame_indices:
        if idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_sum += gray.mean()
        count += 1
    cap.release()
    return brightness_sum / count if count else None


def check_openface_csv(csv_path: Path) -> tuple[str | None, dict]:
    """
    检查 OpenFace AU CSV 是否异常。
    返回 (error_type, info_dict)。
    error_type: None=正常, 'missing', 'empty', 'few_rows', 'low_success', 'au_all_zero', 'au_no_variance'
    """
    info: dict = {"rows": 0, "success_rate": None, "au_cols": 0, "au_mean_nonzero": False, "au_has_variance": False}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return "empty", info
        au_cols = [c for c in reader.fieldnames if _is_au_column(c)]
        info["au_cols"] = len(au_cols)

        rows = list(reader)
        info["rows"] = len(rows)
        if len(rows) == 0:
            return "empty", info
        if len(rows) < 10:
            return "few_rows", info

        # success 列（若有）
        if "success" in reader.fieldnames:
            try:
                success_vals = [int(float(row.get("success", 0))) for row in rows]
                info["success_rate"] = sum(success_vals) / len(success_vals) if success_vals else 0.0
            except (ValueError, TypeError):
                info["success_rate"] = None
            if info["success_rate"] is not None and info["success_rate"] < 0.2:
                return "low_success", info
        else:
            info["success_rate"] = None

        # AU 列：全零或几乎无方差
        if not au_cols:
            return None, info
        arr = np.array([[float(row.get(c, 0) or 0) for c in au_cols] for row in rows], dtype=np.float64)
        nans = np.isnan(arr)
        arr[nans] = 0
        col_means = np.abs(np.nanmean(arr, axis=0))
        col_stds = np.nanstd(arr, axis=0)
        info["au_mean_nonzero"] = bool(np.any(col_means > 1e-6))
        info["au_has_variance"] = bool(np.any(np.isfinite(col_stds) & (col_stds > 1e-6)))
        if not info["au_mean_nonzero"] and not info["au_has_variance"]:
            return "au_all_zero", info
        if not info["au_has_variance"] and info["au_mean_nonzero"]:
            # 有非零均值但无方差，可能是常数（异常）
            return "au_no_variance", info
    return None, info


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    video_dirs = [root / "dataset" / "Nov2021", root / "dataset" / "Oct2020"]
    au_roots = [
        root / "preprocess" / "AU_results" / "FIS_experients",
        root / "preprocess" / "AU_results" / "Stimulis_clips",
    ]
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    videos: list[Path] = []
    for d in video_dirs:
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    videos.append(f)

    print(f"共 {len(videos)} 个视频，AU CSV 根目录: {au_roots}\n")

    # 1) 损坏
    print("1) 检测损坏（ffprobe）...")
    corrupt = [p for p in videos if is_corrupt_ffprobe(p)]
    print(f"   损坏: {len(corrupt)} 个")
    for p in sorted(corrupt)[:20]:
        print(f"     {p}")
    if len(corrupt) > 20:
        print(f"     ... 共 {len(corrupt)} 个")

    ok_videos = [p for p in videos if p not in corrupt]

    # 2) 全黑（首/中/尾 3 帧）
    print("\n2) 检测全黑（首/中/尾帧亮度<15）...")
    black_list: list[tuple[Path, float]] = []
    for i, p in enumerate(ok_videos):
        if (i + 1) % 200 == 0:
            print(f"   已处理 {i + 1}/{len(ok_videos)}")
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        if n < 1:
            continue
        mid = n // 2
        b = mean_brightness(p, [0, mid, n - 1])
        if b is not None and b < 15.0:
            black_list.append((p, round(b, 2)))
    print(f"   全黑: {len(black_list)} 个")
    for p, b in sorted(black_list, key=lambda x: (str(x[0]), x[1]))[:20]:
        print(f"     {p} 亮度={b}")
    if len(black_list) > 20:
        print(f"     ... 共 {len(black_list)} 个")

    # 3) OpenFace CSV 异常（按视频 stem 解析 CSV）
    print("\n3) 检测 OpenFace AU CSV 异常...")
    missing_csv: list[Path] = []
    csv_empty: list[Path] = []
    csv_few_rows: list[tuple[Path, int]] = []
    csv_low_success: list[tuple[Path, float]] = []
    csv_au_all_zero: list[Path] = []
    csv_au_no_variance: list[Path] = []
    csv_ok = 0

    for i, p in enumerate(videos):
        if (i + 1) % 200 == 0:
            print(f"   已处理 {i + 1}/{len(videos)}")
        stem = p.stem
        csv_path = resolve_au_csv(stem, au_roots)
        if csv_path is None:
            missing_csv.append(p)
            continue
        err, info = check_openface_csv(csv_path)
        if err == "empty":
            csv_empty.append(p)
        elif err == "few_rows":
            csv_few_rows.append((p, info["rows"]))
        elif err == "low_success":
            csv_low_success.append((p, info["success_rate"] or 0.0))
        elif err == "au_all_zero":
            csv_au_all_zero.append(p)
        elif err == "au_no_variance":
            csv_au_no_variance.append(p)
        else:
            csv_ok += 1

    print(f"   缺少 CSV:     {len(missing_csv)} 个")
    print(f"   CSV 空:      {len(csv_empty)} 个")
    print(f"   CSV 行数过少: {len(csv_few_rows)} 个")
    print(f"   success 率低: {len(csv_low_success)} 个")
    print(f"   AU 全零:     {len(csv_au_all_zero)} 个")
    print(f"   AU 无方差:   {len(csv_au_no_variance)} 个")
    print(f"   CSV 正常:    {csv_ok} 个")

    # 输出明细
    if missing_csv:
        print("\n--- 缺少 OpenFace CSV 的视频（前 30）---")
        for p in sorted(missing_csv)[:30]:
            print(f"  {p}")
        if len(missing_csv) > 30:
            print(f"  ... 共 {len(missing_csv)} 个")
    if csv_empty:
        print("\n--- CSV 为空 ---")
        for p in sorted(csv_empty)[:20]:
            print(f"  {p}")
    if csv_few_rows:
        print("\n--- CSV 行数过少(<10) ---")
        for p, r in sorted(csv_few_rows, key=lambda x: (str(x[0]), x[1]))[:20]:
            print(f"  {p}  rows={r}")
    if csv_low_success:
        print("\n--- success 率<0.2 ---")
        for p, rate in sorted(csv_low_success, key=lambda x: (str(x[0]), x[1]))[:20]:
            print(f"  {p}  success_rate={rate:.2f}")
    if csv_au_all_zero:
        print("\n--- AU 全零/无有效值 ---")
        for p in sorted(csv_au_all_zero)[:20]:
            print(f"  {p}")
    if csv_au_no_variance:
        print("\n--- AU 无方差（常数）---")
        for p in sorted(csv_au_no_variance)[:20]:
            print(f"  {p}")

    # 汇总：所有问题视频（去重）
    problem_stems = set()
    for p in corrupt + [x[0] for x in black_list] + missing_csv + csv_empty + [x[0] for x in csv_few_rows] + [x[0] for x in csv_low_success] + csv_au_all_zero + csv_au_no_variance:
        problem_stems.add((p.relative_to(root), p.stem))
    print("\n===== 汇总 =====")
    print(f"损坏: {len(corrupt)} | 全黑: {len(black_list)} | 缺CSV: {len(missing_csv)} | CSV空: {len(csv_empty)} | 少行: {len(csv_few_rows)} | 低success: {len(csv_low_success)} | AU全零: {len(csv_au_all_zero)} | AU无方差: {len(csv_au_no_variance)}")
    print(f"至少一类问题的视频数（去重）: {len({s[1] for s in problem_stems})}")


if __name__ == "__main__":
    main()
