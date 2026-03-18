#!/usr/bin/env python3
"""检测 Nov2021 / Oct2020 下视频是否全黑或损坏。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2


def check_video(path: Path, sample_frac: float = 0.1, black_threshold: float = 15.0) -> tuple[str | None, float | None]:
    """
    检查单个视频：能否打开、是否全黑。
    返回 (error_type, mean_brightness)。
    error_type: None=正常, 'corrupt'=无法打开/无帧, 'black'=全黑
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return "corrupt", None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return "corrupt", None

    # 采样若干帧：首、中、尾，以及中间均匀几点
    indices = {0}
    if total_frames > 1:
        indices.add(total_frames - 1)
    n_sample = max(3, int(total_frames * sample_frac))
    step = max(1, total_frames // n_sample)
    for i in range(0, total_frames, step):
        indices.add(min(i, total_frames - 1))
    indices = sorted(indices)

    brightness_sum = 0.0
    count = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return "corrupt", None
        # 灰度均值作为亮度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_sum += gray.mean()
        count += 1

    cap.release()
    mean_brightness = brightness_sum / count if count else 0.0

    if mean_brightness < black_threshold:
        return "black", round(mean_brightness, 2)
    return None, round(mean_brightness, 2)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dirs = [
        root / "dataset" / "Nov2021",
        root / "dataset" / "Oct2020",
    ]
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    all_videos: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            print(f"跳过不存在的目录: {d}", file=sys.stderr)
            continue
        for f in d.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                all_videos.append(f)

    print(f"共扫描 {len(all_videos)} 个视频文件\n")

    corrupt: list[Path] = []
    black: list[tuple[Path, float]] = []
    ok_count = 0

    for i, p in enumerate(all_videos):
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(all_videos)} ...")
        err, brightness = check_video(p)
        if err == "corrupt":
            corrupt.append(p)
        elif err == "black":
            black.append((p, brightness))
        else:
            ok_count += 1

    print("\n===== 统计 =====")
    print(f"正常:     {ok_count}")
    print(f"损坏:     {len(corrupt)}")
    print(f"全黑:     {len(black)}")

    if corrupt:
        print("\n损坏/无法打开:")
        for p in sorted(corrupt):
            print(f"  {p}")
    if black:
        print("\n全黑 (平均亮度 < 15):")
        for p, b in sorted(black, key=lambda x: (str(x[0]), x[1])):
            print(f"  {p}  (亮度={b})")


if __name__ == "__main__":
    main()
