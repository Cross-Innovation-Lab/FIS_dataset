"""实验配置：JSON + dataclass。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 默认 train:val:test 比例（8:1:1 → 80% / 10% / 10%）
DEFAULT_SPLIT_SPEC = "8:1:1"
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_GROUP_BY = "none"


def project_root() -> Path:
    """FIS_dataset 仓库根目录（experiment/ 的上一级）。"""
    return Path(__file__).resolve().parent.parent


def default_split_path(task: int, experiment_root: Path | None = None) -> Path:
    """任务一/任务二默认划分 manifest 路径：experiment/splits/task{1|2}_split.json。"""
    if task not in (1, 2):
        raise ValueError(f"task 须为 1 或 2，收到: {task}")
    root = experiment_root or Path(__file__).resolve().parent
    return root / "splits" / f"task{task}_split.json"


def resolve_split_path(split_file: str | Path, repo_root: Path | None = None) -> Path:
    """将配置中的 split_file 解析为绝对路径（相对路径相对于 FIS_dataset 根目录）。"""
    path = Path(split_file).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root or project_root()) / path


@dataclass
class DataConfig:
    csv_path: str = "dataset/all_labels_Valid.csv"
    feature_root: str = "dataset/FIS_dataset"
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
    # 兼容旧配置保留 train_ratio / val_ratio，默认与 8:1:1 一致。
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    split_spec: str = DEFAULT_SPLIT_SPEC
    split_group_by: str = DEFAULT_SPLIT_GROUP_BY
    split_file: str | None = None  # 留空则按 data.task 使用 default_split_path(task)
    seed: int = 42
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    output_dir: str = "pth/outs"
    ckpt_dir: str = "pth/checkpoints"
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

