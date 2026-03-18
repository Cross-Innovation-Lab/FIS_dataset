"""
列出 FIS 数据集中 audio_wav2vec 为空或缺失的样本 ID。
用法:
  cd /CIL_PROJECTS/CODES/MM_FIS && python -m experiment.list_empty_wav2vec
  python -m experiment.list_empty_wav2vec --csv_path ... --feature_root ... --task 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment.dataloader import FISDataset


def main() -> None:
    p = argparse.ArgumentParser(description="列出 audio_wav2vec 为空或缺失的样本 ID")
    p.add_argument("--csv_path", default="/CIL_PROJECTS/CODES/MM_FIS/dataset/fis_modeling_data_all_labels_Combined.csv")
    p.add_argument("--feature_root", default="/CIL_PROJECTS/CODES/MM_FIS/preprocess/FIS_FEA")
    p.add_argument("--task", type=int, default=1)
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

    empty_counselor: list[str] = []
    empty_patient: list[str] = []

    for i in range(len(ds)):
        s = ds[i]
        sid = s["sample_id"]
        c = s["counselor"]
        if c.get("audio_wav2vec") is None or (hasattr(c["audio_wav2vec"], "numel") and c["audio_wav2vec"].numel() == 0):
            empty_counselor.append(sid)
        if args.task == 2 and s.get("patient"):
            p = s["patient"]
            if p.get("audio_wav2vec") is None or (hasattr(p["audio_wav2vec"], "numel") and p["audio_wav2vec"].numel() == 0):
                empty_patient.append(sid)

    print(f"总样本数: {len(ds)}, task={args.task}")
    print(f"Counselor 空/缺 audio_wav2vec: {len(empty_counselor)} 条")
    for sid in empty_counselor:
        print(f"  {sid}")
    if args.task == 2:
        print(f"Patient 空/缺 audio_wav2vec: {len(empty_patient)} 条")
        for sid in empty_patient:
            print(f"  {sid}")
    if empty_counselor:
        print("\nID 列表（可复制）:")
        print(empty_counselor)


if __name__ == "__main__":
    main()
