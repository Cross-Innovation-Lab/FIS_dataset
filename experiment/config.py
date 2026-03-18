"""实验配置：JSON + dataclass。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    csv_path: str = "/CIL_PROJECTS/CODES/MM_FIS/dataset/all_labels_Valid.csv"
    feature_root: str = "/CIL_PROJECTS/CODES/MM_FIS/dataset/FIS_dataset"
    feature_source: str = "raw"
    feature_categories: list[str] = field(default_factory=lambda: ["audio", "video", "text"])
    # 已剔除 avalid 机制：样本仅由标签 CSV 决定；不再使用 valid_id_csv 过滤
    avalid_csv: str | None = None
    # task=2 时可选：咨询师 ID -> Patient basename 映射表（CSV 或 None 表示自动扫描解析）
    patient_basename_map_csv: str | None = None
    # 1=仅咨询师, 2=咨询师+来访者（与 dataloader.FISDataset 保持一致）
    task: int = 1
    counselor_role: str = "Counselor"
    patient_role: str = "Patient"
    max_len_word: int | None = None
    max_len_tok: int | None = 128
    num_workers: int = 0


@dataclass
class ModelConfig:
    name: str = "simple_multimodal"
    kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "text_dim": 1024,
            "audio_dim": 768,
            "video_dim": 235,
            "hidden_dim": 256,
            "out_dim": 9,
            "use_text": True,
            "use_audio": True,
            "use_video": True,
            "text_pool": "mean_pool",
        }
    )


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 0.0  # 梯度裁剪，0 表示不裁剪
    ccc_loss_weight: float = 0.0
    # 兼容旧配置保留 train_ratio / val_ratio，默认与 16:4:5 一致。
    train_ratio: float = 16 / 25
    val_ratio: float = 4 / 25
    split_spec: str = "16:4:5"
    split_group_by: str = "none"
    split_file: str | None = None
    seed: int = 42
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    output_dir: str = "/CIL_PROJECTS/CODES/MM_FIS/outs"
    ckpt_dir: str = "/CIL_PROJECTS/CODES/MM_FIS/checkpoints"
    run_name: str = ""  # 留空则自动按 model_name + 时间戳生成
    save_best_by: str = "val_mae"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def _update_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for k, v in values.items():
        if hasattr(obj, k):
            cur = getattr(obj, k)
            if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
                _update_dataclass(cur, v)
            else:
                setattr(obj, k, v)
    return obj


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _update_dataclass(cfg, data)

