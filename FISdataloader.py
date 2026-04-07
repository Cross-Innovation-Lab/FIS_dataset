"""Simple feature reader for loading FIS features and matching labels."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# FEATURE_ROOT = "/absolute/path/to/your/feature/folder"
# LABEL_CSV_PATH = "/absolute/path/to/your/label/file.csv"
FEATURE_ROOT = "/CIL_PROJECTS/CODES/MM_FIS/dataset/FIS_dataset/Counselor"
LABEL_CSV_PATH = "/CIL_PROJECTS/CODES/MM_FIS/dataset/all_labels_Valid.csv"
FEATURE_SOURCE = "raw"
FEATURE_CATEGORIES = ["audio", "video", "text"]
PREVIEW_SAMPLE_COUNT = 3

LABEL_COLUMNS = [
    "abc",
    "all_ratings",
    "arrr",
    "emotional_expression",
    "empathy",
    "hope_and_pe",
    "persuasiveness",
    "verbal_fluency",
    "wau",
]

RAW_CATEGORY_TO_REQUIRED_SUBDIRS = {
    "audio": ["audio/wav2vec", "audio/librosa"],
    "video": ["video/openface3"],
    "text": ["text/embedding", "text/token", "text/words"],
}

ALIGNED_CATEGORY_TO_REQUIRED_SUBDIRS = {
    "audio": ["aligned/audio_wav2vec", "aligned/audio_librosa"],
    "video": ["aligned/video_openface3"],
    "text": ["text/embedding", "text/token", "text/words"],
}


def normalize_sample_id(sample_id: str) -> str:
    """Normalize sample IDs from the label CSV."""
    value = str(sample_id).strip()
    value = re.sub(r"fis_time1", "FIS_Time1", value, flags=re.IGNORECASE)
    value = re.sub(r"fis_time2", "FIS_Time2", value, flags=re.IGNORECASE)
    value = re.sub(r"fis_time3", "FIS_Time3", value, flags=re.IGNORECASE)
    return value


def csv_id_to_basename(csv_id: str) -> str:
    """Convert a label CSV ID to the feature basename."""
    value = normalize_sample_id(csv_id)
    value = re.sub(r"FIS_Time1(?=_|$)", "T1", value, flags=re.IGNORECASE)
    value = re.sub(r"FIS_Time2(?=_|$)", "T2", value, flags=re.IGNORECASE)
    value = re.sub(r"FIS_Time3(?=_|$)", "T3", value, flags=re.IGNORECASE)
    return value


def basename_to_csv_id(basename: str) -> str:
    """Convert a feature basename back to the label CSV ID format."""
    value = str(basename).strip()
    value = re.sub(r"_T1_", "_FIS_Time1_", value, flags=re.IGNORECASE)
    value = re.sub(r"_T1$", "_FIS_Time1", value, flags=re.IGNORECASE)
    value = re.sub(r"_T2_", "_FIS_Time2_", value, flags=re.IGNORECASE)
    value = re.sub(r"_T2$", "_FIS_Time2", value, flags=re.IGNORECASE)
    value = re.sub(r"_T3_", "_FIS_Time3_", value, flags=re.IGNORECASE)
    value = re.sub(r"_T3$", "_FIS_Time3", value, flags=re.IGNORECASE)
    return normalize_sample_id(value)


def _normalize_feature_categories(feature_categories: list[str] | tuple[str, ...] | str | None) -> list[str]:
    """Normalize the feature category configuration."""
    if feature_categories is None:
        categories = list(FEATURE_CATEGORIES)
    elif isinstance(feature_categories, str):
        categories = [item.strip().lower() for item in feature_categories.split(",") if item.strip()]
    else:
        categories = [str(item).strip().lower() for item in feature_categories if str(item).strip()]
    valid = {"audio", "video", "text"}
    invalid = [item for item in categories if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported feature categories: {invalid}. Valid options: {sorted(valid)}")
    return categories or list(FEATURE_CATEGORIES)


def _resolve_required_subdirs(
    feature_source: str = FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> list[str]:
    """Return the required subdirectories for the selected source and categories."""
    categories = _normalize_feature_categories(feature_categories)
    source = str(feature_source).strip().lower()
    if source == "raw":
        mapping = RAW_CATEGORY_TO_REQUIRED_SUBDIRS
    elif source == "aligned":
        mapping = ALIGNED_CATEGORY_TO_REQUIRED_SUBDIRS
    elif source == "packed":
        return []
    else:
        raise ValueError(f"Unsupported feature_source: {feature_source}")
    subdirs: list[str] = []
    for category in categories:
        subdirs.extend(mapping[category])
    return subdirs


def _resolve_reference_subdir(
    feature_source: str = FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> str | None:
    """Pick one directory to enumerate available basenames."""
    categories = _normalize_feature_categories(feature_categories)
    source = str(feature_source).strip().lower()
    if source == "packed":
        return None
    candidates = {
        "raw": {
            "audio": "audio/wav2vec",
            "video": "video/openface3",
            "text": "text/embedding",
        },
        "aligned": {
            "audio": "aligned/audio_wav2vec",
            "video": "aligned/video_openface3",
            "text": "text/embedding",
        },
    }
    if source not in candidates:
        raise ValueError(f"Unsupported feature_source: {feature_source}")
    for category in categories:
        return candidates[source][category]
    return candidates[source]["audio"]


def collect_available_basenames(
    feature_root: str | Path,
    feature_source: str = FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> list[str]:
    """Collect basenames that have the required feature files."""
    root = Path(feature_root)
    if not root.exists():
        raise FileNotFoundError(f"Feature root does not exist: {root}")

    subdirs = _resolve_required_subdirs(feature_source, feature_categories)
    ref_subdir = _resolve_reference_subdir(feature_source, feature_categories)
    ref_dir = root if ref_subdir is None else root / ref_subdir
    if not ref_dir.exists():
        return []

    def is_complete(basename: str) -> bool:
        if not subdirs:
            return (root / f"{basename}.npz").exists()
        return all((root / subdir / f"{basename}.npz").exists() for subdir in subdirs)

    return sorted(file.stem for file in ref_dir.glob("*.npz") if is_complete(file.stem))


def load_feature_bundle(
    feature_root: str | Path,
    basename: str,
    feature_source: str = FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> dict[str, Any]:
    """Load one feature bundle from the feature directory."""
    root = Path(feature_root)
    categories = _normalize_feature_categories(feature_categories)
    source = str(feature_source).strip().lower()
    output: dict[str, Any] = {
        "audio_wav2vec": None,
        "audio_librosa": None,
        "audio_prosody": None,
        "video_openface3": None,
        "text_embedding": None,
        "text_token": None,
        "text_word_timestamps": None,
        "text_token_timestamps": None,
        "text_words": [],
    }

    if source == "packed":
        packed_path = root / f"{basename}.npz"
        if not packed_path.exists():
            return output
        data = np.load(packed_path, allow_pickle=True)
        for key in [
            "audio_wav2vec",
            "audio_librosa",
            "audio_prosody",
            "video_openface3",
            "text_embedding",
            "text_token",
            "text_word_timestamps",
            "text_token_timestamps",
        ]:
            if key in data.files:
                output[key] = data[key]
        if "text_words" in data.files:
            text_words = data["text_words"]
            output["text_words"] = text_words.tolist() if hasattr(text_words, "tolist") else text_words
        return output

    if "audio" in categories:
        audio_mapping = [
            ("audio_wav2vec", "audio/wav2vec" if source == "raw" else "aligned/audio_wav2vec"),
            ("audio_librosa", "audio/librosa" if source == "raw" else "aligned/audio_librosa"),
        ]
        for key, subdir in audio_mapping:
            file_path = root / subdir / f"{basename}.npz"
            if file_path.exists():
                data = np.load(file_path, allow_pickle=True)
                output[key] = data["features"]
        if source == "raw":
            prosody_path = root / "audio/prosody" / f"{basename}.npz"
            if prosody_path.exists():
                data = np.load(prosody_path, allow_pickle=True)
                output["audio_prosody"] = data["features"]

    if "video" in categories:
        video_mapping = [
            ("video_openface3", "video/openface3" if source == "raw" else "aligned/video_openface3"),
        ]
        for key, subdir in video_mapping:
            file_path = root / subdir / f"{basename}.npz"
            if file_path.exists():
                data = np.load(file_path, allow_pickle=True)
                output[key] = data["features"]

    if "text" in categories:
        for key, subdir in [
            ("text_embedding", "text/embedding"),
            ("text_token", "text/token"),
        ]:
            file_path = root / subdir / f"{basename}.npz"
            if file_path.exists():
                data = np.load(file_path, allow_pickle=True)
                output[key] = data["features"]
                if "timestamps" in data.files and key == "text_token":
                    output["text_token_timestamps"] = data["timestamps"]

        words_path = root / "text/words" / f"{basename}.npz"
        if words_path.exists():
            data = np.load(words_path, allow_pickle=True)
            if "timestamps" in data.files:
                output["text_word_timestamps"] = data["timestamps"]
            if "text" in data.files:
                text_words = data["text"]
                output["text_words"] = text_words.tolist() if hasattr(text_words, "tolist") else text_words

    return output


def _to_tensor(array: np.ndarray | None, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
    if array is None:
        return None
    tensor = torch.from_numpy(np.asarray(array))
    if tensor.dtype in (torch.float64, torch.float32) and dtype == torch.float32:
        tensor = tensor.float()
    elif dtype == torch.long and tensor.dtype != torch.long:
        tensor = tensor.long()
    return tensor


def _numpy_to_tensors(features: dict[str, Any]) -> dict[str, Any]:
    """Convert NumPy arrays to tensors while keeping text metadata readable."""
    result: dict[str, Any] = {}
    for key, value in features.items():
        if key == "text_words":
            result[key] = value
            continue
        if isinstance(value, np.ndarray):
            if value.dtype == object or value.dtype.kind == "O":
                result[key] = value.tolist() if hasattr(value, "tolist") else value
            else:
                result[key] = _to_tensor(value, dtype=torch.long if key == "text_token" else torch.float32)
        else:
            result[key] = value
    return result


class FeatureReaderDataset(Dataset):
    """Task-agnostic dataset for previewing feature files and matching labels."""

    def __init__(
        self,
        label_csv_path: str | Path,
        feature_root: str | Path,
        feature_source: str = FEATURE_SOURCE,
        feature_categories: list[str] | tuple[str, ...] | str | None = None,
        label_columns: list[str] | None = None,
    ) -> None:
        self.label_csv_path = Path(label_csv_path)
        self.feature_root = Path(feature_root)
        self.feature_source = str(feature_source).strip().lower()
        self.feature_categories = _normalize_feature_categories(feature_categories)
        self.label_columns = label_columns or LABEL_COLUMNS

        if not self.label_csv_path.exists():
            raise FileNotFoundError(f"Label CSV does not exist: {self.label_csv_path}")
        if not self.feature_root.exists():
            raise FileNotFoundError(f"Feature root does not exist: {self.feature_root}")

        dataframe = pd.read_csv(self.label_csv_path)
        if "ID" not in dataframe.columns:
            raise ValueError("Label CSV must contain an 'ID' column.")

        dataframe = dataframe.copy()
        dataframe["sample_id"] = dataframe["ID"].astype(str).map(str.strip)
        dataframe["feature_basename"] = dataframe["sample_id"].map(csv_id_to_basename)

        available_basenames = set(
            collect_available_basenames(
                feature_root=self.feature_root,
                feature_source=self.feature_source,
                feature_categories=self.feature_categories,
            )
        )
        self.df = dataframe[dataframe["feature_basename"].isin(available_basenames)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sample_id = str(row["sample_id"]).strip()
        feature_basename = str(row["feature_basename"]).strip()
        label_dict = {
            column: float(0.0 if pd.isna(row[column]) else row[column])
            for column in self.label_columns
            if column in row
        }
        return {
            "sample_id": sample_id,
            "feature_basename": feature_basename,
            "labels": torch.tensor([label_dict.get(column, 0.0) for column in self.label_columns], dtype=torch.float32),
            "label_dict": label_dict,
            "features": _numpy_to_tensors(
                load_feature_bundle(
                    feature_root=self.feature_root,
                    basename=feature_basename,
                    feature_source=self.feature_source,
                    feature_categories=self.feature_categories,
                )
            ),
        }


def _summarize_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"tensor{tuple(value.shape)}"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if value is None:
        return "None"
    return str(type(value).__name__)


def describe_sample(sample: dict[str, Any]) -> None:
    """Print a readable summary for one sample."""
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Feature basename: {sample['feature_basename']}")
    print(f"Labels: {sample['label_dict']}")
    print("Features:")
    for key, value in sample["features"].items():
        print(f"  - {key}: {_summarize_value(value)}")


def main() -> None:
    """Preview a few samples using the top-level configuration."""
    print("Feature reader configuration:")
    print(f"  FEATURE_ROOT: {FEATURE_ROOT}")
    print(f"  LABEL_CSV_PATH: {LABEL_CSV_PATH}")
    print(f"  FEATURE_SOURCE: {FEATURE_SOURCE}")
    print(f"  FEATURE_CATEGORIES: {FEATURE_CATEGORIES}")

    try:
        dataset = FeatureReaderDataset(
            label_csv_path=LABEL_CSV_PATH,
            feature_root=FEATURE_ROOT,
            feature_source=FEATURE_SOURCE,
            feature_categories=FEATURE_CATEGORIES,
            label_columns=LABEL_COLUMNS,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Configuration error: {error}")
        return

    print(f"Matched samples with available features: {len(dataset)}")
    if len(dataset) == 0:
        print("No samples matched the current feature directory and label CSV.")
        return

    preview_count = min(PREVIEW_SAMPLE_COUNT, len(dataset))
    for index in range(preview_count):
        print(f"\\n--- Preview {index + 1}/{preview_count} ---")
        describe_sample(dataset[index])


if __name__ == "__main__":
    main()
