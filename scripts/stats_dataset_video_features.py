#!/usr/bin/env python3
"""
基于 all_labels_Valid.csv 的 ID 索引，统计视频文件与转化后特征的规模与维度，
并生成文档存入 docs/。

视频：Counselor 视频在 Oct2020/Nov2021，Patient（刺激）视频在 FIS_stimulis_clips。
特征：FIS_dataset 下 Counselor 与 Patient 的 npz 特征（帧数、维度等）。
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# 路径配置（相对项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
CSV_PATH = DATASET_ROOT / "all_labels_Valid.csv"
VIDEO_DIRS = {
    "20-Oct": DATASET_ROOT / "Oct2020",
    "21-Nov": DATASET_ROOT / "Nov2021",
}
STIMULUS_VIDEO_DIR = DATASET_ROOT / "FIS_stimulis_clips"
FIS_DATASET = DATASET_ROOT / "FIS_dataset"
COUNSELOR_FEATURE_DIR = FIS_DATASET / "Counselor"
PATIENT_FEATURE_DIR = FIS_DATASET / "Patient"
OUTPUT_DOC = PROJECT_ROOT / "docs" / "dataset_video_feature_statistics.md"


def id_to_video_basename(label_id: str) -> str | None:
    """将 CSV 中的 ID 转为视频文件名（无扩展名）。如 AG0914_FIS_Time1_Jackson -> AG0914_T1_Jackson。"""
    # 兼容 FIs_Time 拼写
    s = re.sub(r"FIS_Time(\d)|FIs_Time(\d)", lambda m: f"T{m.group(1) or m.group(2)}", label_id, flags=re.I)
    return s if s != label_id else None


def get_video_duration_seconds(video_path: Path) -> float | None:
    """使用 ffprobe 获取视频时长（秒）。"""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def load_label_ids_and_folders() -> list[tuple[str, str]]:
    """读取 CSV，返回 (ID, folder) 列表，folder 为 20-Oct 或 21-Nov。"""
    rows: list[tuple[str, str]] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            vid = (row.get("ID") or "").strip()
            folder = (row.get("folder") or "").strip()
            if vid and folder:
                rows.append((vid, folder))
    return rows


def collect_counselor_video_stats(
    id_folders: list[tuple[str, str]],
) -> dict[str, Any]:
    """统计 Counselor 视频：个数、时长（平均/最长/最短）、对应特征帧数；并收集时长短于 30s 的列表。"""
    durations: list[float] = []
    missing: list[str] = []
    under_30s: list[tuple[str, float]] = []  # (filename.mp4, duration_sec)
    id_to_basename: dict[str, str] = {}
    for label_id, folder in id_folders:
        base = id_to_video_basename(label_id)
        if not base:
            missing.append(label_id)
            continue
        id_to_basename[label_id] = base
        video_dir = VIDEO_DIRS.get(folder)
        if not video_dir or not video_dir.exists():
            missing.append(label_id)
            continue
        video_path = video_dir / f"{base}.mp4"
        if not video_path.exists():
            missing.append(label_id)
            continue
        d = get_video_duration_seconds(video_path)
        if d is not None:
            durations.append(d)
            if d < 30:
                under_30s.append((f"{base}.mp4", d))
    # 特征帧数：仅针对 valid CSV 对应的 basename，从 Counselor 的 npz 中读取 video_openface3 的 shape[0]
    valid_basenames = set(id_to_basename.values())
    frame_counts: list[int] = []
    feature_dims: dict[str, list[int]] = {}
    if COUNSELOR_FEATURE_DIR.exists():
        for npz_path in sorted(COUNSELOR_FEATURE_DIR.glob("*.npz")):
            if npz_path.stem not in valid_basenames:
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
                if "video_openface3" in data:
                    arr = data["video_openface3"]
                    frame_counts.append(int(arr.shape[0]))
                    for key in ("video_openface3", "video_occ_pad", "audio_wav2vec", "audio_prosody", "audio_librosa", "text_embedding"):
                        if key in data and hasattr(data[key], "shape"):
                            s = data[key].shape
                            if key not in feature_dims:
                                feature_dims[key] = []
                            if len(s) >= 2:
                                feature_dims[key].append(int(s[1]))
            except Exception:
                continue
    # 维度取众数或中位数（一般一致）
    dims_summary: dict[str, int] = {}
    for k, v in feature_dims.items():
        dims_summary[k] = int(np.median(v)) if v else 0
    # 按时长升序排列，便于查看
    under_30s.sort(key=lambda x: x[1])
    return {
        "count_with_duration": len(durations),
        "count_missing": len(missing),
        "duration_avg_sec": float(np.mean(durations)) if durations else None,
        "duration_max_sec": float(np.max(durations)) if durations else None,
        "duration_min_sec": float(np.min(durations)) if durations else None,
        "under_30s": under_30s,
        "frame_counts": frame_counts,
        "feature_dims": dims_summary,
        "total_label_count": len(id_folders),
    }


def collect_patient_video_stats() -> dict[str, Any]:
    """统计 Patient（刺激）视频：FIS_stimulis_clips 下的 mp4 时长及 Patient 特征。"""
    durations: list[float] = []
    names: list[str] = []
    if not STIMULUS_VIDEO_DIR.exists():
        return {
            "count": 0,
            "duration_avg_sec": None,
            "duration_max_sec": None,
            "duration_min_sec": None,
            "frame_counts": [],
            "feature_dims": {},
        }
    for mp4 in sorted(STIMULUS_VIDEO_DIR.glob("*.mp4")):
        names.append(mp4.stem)
        d = get_video_duration_seconds(mp4)
        if d is not None:
            durations.append(d)
    frame_counts: list[int] = []
    feature_dims: dict[str, list[int]] = {}
    if PATIENT_FEATURE_DIR.exists():
        for npz_path in sorted(PATIENT_FEATURE_DIR.glob("*.npz")):
            if "FIS_stimulis_clips__" not in npz_path.name:
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
                if "video_openface3" in data:
                    arr = data["video_openface3"]
                    frame_counts.append(int(arr.shape[0]))
                    for key in ("video_openface3", "video_occ_pad", "audio_wav2vec", "audio_prosody", "audio_librosa", "text_embedding"):
                        if key in data and hasattr(data[key], "shape"):
                            s = data[key].shape
                            if key not in feature_dims:
                                feature_dims[key] = []
                            if len(s) >= 2:
                                feature_dims[key].append(int(s[1]))
            except Exception:
                continue
    dims_summary = {k: int(np.median(v)) if v else 0 for k, v in feature_dims.items()}
    return {
        "count": len(durations),
        "video_names": names,
        "duration_avg_sec": float(np.mean(durations)) if durations else None,
        "duration_max_sec": float(np.max(durations)) if durations else None,
        "duration_min_sec": float(np.min(durations)) if durations else None,
        "frame_counts": frame_counts,
        "feature_dims": dims_summary,
    }


def format_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    return f"{sec:.2f} s"


def write_markdown_doc(counselor_stats: dict[str, Any], patient_stats: dict[str, Any]) -> None:
    """将统计结果写入 docs/dataset_video_feature_statistics.md。"""
    OUTPUT_DOC.parent.mkdir(parents=True, exist_ok=True)

    c = counselor_stats
    p = patient_stats

    c_frames = c.get("frame_counts") or []
    c_n_frames = len(c_frames)
    c_avg_frames = float(np.mean(c_frames)) if c_frames else None
    c_min_frames = int(min(c_frames)) if c_frames else None
    c_max_frames = int(max(c_frames)) if c_frames else None

    p_frames = p.get("frame_counts") or []
    p_avg_frames = float(np.mean(p_frames)) if p_frames else None
    p_min_frames = int(min(p_frames)) if p_frames else None
    p_max_frames = int(max(p_frames)) if p_frames else None

    lines = [
        "# FIS 数据集：视频与特征统计",
        "",
        "本文档基于 `dataset/all_labels_Valid.csv` 中的 ID 索引，对 Counselor 与 Patient 视频及其转化后的特征进行统计。",
        "",
        "## 1. 数据路径说明",
        "",
        "### 1.1 视频文件存储路径",
        "",
        "| 角色 | 路径 | 说明 |",
        "|------|------|------|",
        "| Counselor | `dataset/Oct2020/` | 2020 年采集批次（folder=20-Oct） |",
        "| Counselor | `dataset/Nov2021/` | 2021 年采集批次（folder=21-Nov） |",
        "| Patient（刺激） | `dataset/FIS_stimulis_clips/` | 刺激视频片段，按角色名命名（如 Bethany.mp4） |",
        "",
        "### 1.2 特征文件存储路径",
        "",
        "| 类型 | 路径 |",
        "|------|------|",
        "| 标签索引 | `dataset/all_labels_Valid.csv` |",
        "| Counselor 特征 | `dataset/FIS_dataset/Counselor/` |",
        "| Patient 特征 | `dataset/FIS_dataset/Patient/` |",
        "",
        "## 2. Counselor 视频与特征",
        "",
        "Counselor 视频与 CSV 中 ID 的对应规则：`{ID}` 中 `FIS_Time1`/`FIS_Time2`/`FIS_Time3` 分别映射为 `T1`/`T2`/`T3`，得到视频文件名（如 `AG0914_T1_Jackson.mp4`），并根据 `folder` 列在 Oct2020 或 Nov2021 中查找。",
        "",
        "### 2.1 视频统计",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 标签条数（CSV 行数） | {c.get('total_label_count', 0)} |",
        f"| 成功获取时长的视频个数 | {c.get('count_with_duration', 0)} |",
        f"| 缺失或无法读时长的条数 | {c.get('count_missing', 0)} |",
        f"| 平均时长 | {format_duration(c.get('duration_avg_sec'))} |",
        f"| 最长时长 | {format_duration(c.get('duration_max_sec'))} |",
        f"| 最短时长 | {format_duration(c.get('duration_min_sec'))} |",
        "",
        "#### 时长短于 30 秒的视频列表",
        "",
    ]
    under_30s = c.get("under_30s") or []
    if under_30s:
        lines.append(f"以下共 **{len(under_30s)}** 个 Counselor 视频时长小于 30 秒。")
        lines.append("")
        lines.append("| 文件名 | 时长 (s) |")
        lines.append("|--------|----------|")
        for fname, dur in under_30s:
            lines.append(f"| {fname} | {dur:.2f} |")
    else:
        lines.append("无时长小于 30 秒的 Counselor 视频。")
    lines.extend([
        "",
        "### 2.2 转化后用于分析的帧数与特征维度",
        "",
        "特征来源于 `FIS_dataset/Counselor/*.npz`，以 `video_openface3` 的帧数为“用于分析的帧数”。",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 特征样本数（npz 文件数） | {c_n_frames} |",
        f"| 平均帧数（每样本） | {f'{c_avg_frames:.1f}' if c_avg_frames is not None else '—'} |",
        f"| 最少帧数 | {c_min_frames} |",
        f"| 最多帧数 | {c_max_frames} |",
        "",
        "**特征维度（按模态）**",
        "",
        "| 特征键 | 维度 |",
        "|--------|------|",
    ])
    for k, dim in (c.get("feature_dims") or {}).items():
        lines.append(f"| {k} | {dim} |")
    lines.extend([
        "",
        "## 3. Patient（刺激）视频与特征",
        "",
        "Patient 视频为刺激片段，存放在 `FIS_stimulis_clips/`，按角色名命名（如 `Bethany.mp4`）。特征来自 `FIS_dataset/Patient/FIS_stimulis_clips__*.npz`。",
        "",
        "### 3.1 视频统计",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 视频个数 | {p.get('count', 0)} |",
        f"| 平均时长 | {format_duration(p.get('duration_avg_sec'))} |",
        f"| 最长时长 | {format_duration(p.get('duration_max_sec'))} |",
        f"| 最短时长 | {format_duration(p.get('duration_min_sec'))} |",
        "",
        "### 3.2 转化后用于分析的帧数与特征维度",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 特征样本数 | {len(p_frames)} |",
        f"| 平均帧数（每样本） | {f'{p_avg_frames:.1f}' if p_avg_frames is not None else '—'} |",
        f"| 最少帧数 | {p_min_frames} |",
        f"| 最多帧数 | {p_max_frames} |",
        "",
        "**特征维度（按模态）**",
        "",
        "| 特征键 | 维度 |",
        "|--------|------|",
    ])
    for k, dim in (p.get("feature_dims") or {}).items():
        lines.append(f"| {k} | {dim} |")
    lines.extend([
        "",
        "## 4. 如何复现本文档",
        "",
        "在项目根目录下执行：",
        "",
        "```bash",
        "python scripts/stats_dataset_video_features.py",
        "```",
        "",
        "依赖：需安装 `numpy`，且系统已安装 `ffprobe`（用于读取视频时长）。",
        "",
        "---",
        "",
        "*文档由 `scripts/stats_dataset_video_features.py` 自动生成。*",
        "",
    ])

    OUTPUT_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入: {OUTPUT_DOC}")


def main() -> int:
    print("读取标签索引...")
    id_folders = load_label_ids_and_folders()
    print(f"  共 {len(id_folders)} 条 ID。")

    print("统计 Counselor 视频与特征...")
    counselor_stats = collect_counselor_video_stats(id_folders)

    print("统计 Patient 视频与特征...")
    patient_stats = collect_patient_video_stats()

    print("生成文档...")
    write_markdown_doc(counselor_stats, patient_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
