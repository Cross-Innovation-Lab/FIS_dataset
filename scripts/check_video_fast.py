#!/usr/bin/env python3
"""快速检测：ffprobe 判损坏，仅采样 3 帧判全黑。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2


def is_corrupt_ffprobe(path: Path) -> bool:
    """用 ffprobe 检测是否无法打开或无有效视频流。"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return r.returncode != 0 or "video" not in (r.stdout or "")


def mean_brightness(path: Path, frame_indices: list[int]) -> float | None:
    """读取指定帧并返回平均亮度，失败返回 None。"""
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dirs = [root / "dataset" / "Nov2021", root / "dataset" / "Oct2020"]
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = []
    for d in dirs:
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    videos.append(f)

    print(f"共 {len(videos)} 个视频\n1) ffprobe 检测损坏...")
    corrupt = []
    for i, p in enumerate(videos):
        if (i + 1) % 200 == 0:
            print(f"   {i + 1}/{len(videos)}")
        if is_corrupt_ffprobe(p):
            corrupt.append(p)
    print(f"   损坏/无法打开: {len(corrupt)} 个")
    if corrupt:
        for p in sorted(corrupt)[:30]:
            print(f"     {p}")
        if len(corrupt) > 30:
            print(f"     ... 共 {len(corrupt)} 个")

    ok = [p for p in videos if p not in corrupt]
    print("\n2) 采样 3 帧检测全黑（首/中/尾）...")
    black = []
    for i, p in enumerate(ok):
        if (i + 1) % 200 == 0:
            print(f"   {i + 1}/{len(ok)}")
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        if n < 1:
            continue
        mid = n // 2
        indices = [0, mid, n - 1]
        b = mean_brightness(p, indices)
        if b is not None and b < 15.0:
            black.append((p, round(b, 2)))
    print(f"   全黑(亮度<15): {len(black)} 个")
    if black:
        for p, b in sorted(black, key=lambda x: (str(x[0]), x[1]))[:30]:
            print(f"     {p} 亮度={b}")
        if len(black) > 30:
            print(f"     ... 共 {len(black)} 个")

    print("\n===== 汇总 =====")
    print(f"总视频数:     {len(videos)}")
    print(f"损坏:         {len(corrupt)}")
    print(f"全黑:         {len(black)}")
    print(f"正常:         {len(videos) - len(corrupt) - len(black)}")


if __name__ == "__main__":
    main()
