"""
扫描 FIS Dataset 中每条样本各模态的特征维，列出特征维为 200 的样本（用于排查 collate 235 vs 200 错误）。
用法（与 timesformer_task2 同数据）:
  cd /CIL_PROJECTS/CODES/MM_FIS && python -m experiment.check_feature_dims
或指定参数:
  python -m experiment.check_feature_dims --csv_path ... --feature_root ... --task 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment.dataloader import FISDataset

KEYS_WORD = ["video_openface3", "audio_wav2vec", "audio_librosa", "text_word_timestamps"]
KEYS_TOK = ["text_embedding", "text_token", "text_token_timestamps"]
ALL_KEYS = KEYS_WORD + KEYS_TOK


def _feat_dim(role_dict: dict | None, key: str) -> int | None:
    if not role_dict:
        return None
    t = role_dict.get(key)
    if t is None or not hasattr(t, "shape") or t.ndim < 2:
        return None
    return t.shape[1]


def main() -> None:
    p = argparse.ArgumentParser(description="List samples with 200-dim feature for any modality.")
    p.add_argument("--csv_path", default="/CIL_PROJECTS/CODES/MM_FIS/dataset/fis_modeling_data_all_labels_Combined.csv")
    p.add_argument("--feature_root", default="/CIL_PROJECTS/CODES/MM_FIS/preprocess/FIS_FEA")
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--target_dim", type=int, default=200, help="要查找的特征维（默认 200）")
    args = p.parse_args()

    ds = FISDataset(
        csv_path=args.csv_path,
        feature_root=args.feature_root,
        task=args.task,
        counselor_role="Counselor",
        patient_role="Patient",
        valid_id_csv=None,
        patient_basename_map=None,
    )

    # 收集: sample_id -> { (role, key): feat_dim }
    samples_with_target_dim: list[tuple[str, list[tuple[str, str, int]]]] = []  # (sample_id, [(role, key, dim), ...])
    dim_counts: dict[int, set[tuple[str, str]]] = {}  # dim -> set of (role, key) 出现过的组合

    for idx in range(len(ds)):
        sample = ds[idx]
        sid = sample["sample_id"]
        entries: list[tuple[str, str, int]] = []

        for role, role_key in [("counselor", "counselor"), ("patient", "patient")]:
            role_dict = sample.get(role_key)
            for key in ALL_KEYS:
                d = _feat_dim(role_dict, key)
                if d is None:
                    continue
                dim_counts.setdefault(d, set()).add((role, key))
                if d == args.target_dim:
                    entries.append((role, key, d))

        if entries:
            samples_with_target_dim.append((sid, entries))

    # 输出
    print(f"数据集样本数: {len(ds)}, task={args.task}")
    print(f"特征维为 {args.target_dim} 的样本数: {len(samples_with_target_dim)}")
    print()
    print("--- 各特征维出现过的 (role, key) ---")
    for dim in sorted(dim_counts.keys()):
        pairs = sorted(dim_counts[dim])
        print(f"  dim={dim}: {pairs}")
    print()
    print(f"--- 特征维为 {args.target_dim} 的样本列表 ---")
    for sid, entries in samples_with_target_dim:
        parts = [f"{role}.{key}={d}" for role, key, d in entries]
        print(f"  {sid}  ->  {', '.join(parts)}")
    print()
    if samples_with_target_dim:
        print("样本 ID 列表（可复制）:")
        print([sid for sid, _ in samples_with_target_dim])


if __name__ == "__main__":
    main()
