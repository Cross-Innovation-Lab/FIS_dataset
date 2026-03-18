"""
FIS 多模态数据加载：从 FIS_FEA 分模态特征目录 + CSV 标签构建 Dataset/DataLoader。
支持任务一（单咨询师 FIS 评估）与任务二（咨询师-来访者交互 FIS 评估）。

文本 token 维度的使用：
  - batch 中 counselor.text_embedding / text_token 形状为 [B, N_tok, D]（N_tok 为 padding 后长度）。
  - counselor_tok_mask 形状 [B, N_tok]，True 表示有效 token，False 表示 padding。
  - 常见处理方式：
    1) 注意力中屏蔽 padding：在 self-attention / cross-attention 里对 mask 为 False 的位置
       置为 -inf 或乘以 0，避免 padding 参与计算。
    2) 序列级表示：用 [CLS]（首 token）、或对有效 token 做 mask 加权 mean-pool，
       得到 [B, D] 再接分类/回归头。见 aggregate_text_embedding()。
    3) 与词级对齐：用 text_token_timestamps 与 text_word_timestamps 做 token→word
       mean-pool，得到 [B, N_word, D]，与 audio_wav2vec / video_openface3 共享 N_word 轴做融合。

训练时 text 模态使用建议（DeBERTa 等变长 N_tok）：
  - DataLoader 已对 N_tok 做 batch 内 padding，并给出 counselor_tok_mask。
  - 若仅用文本做 FIS 回归：先 aggregate_text_embedding(emb, tok_mask, "mean_pool") 或 "cls"
    得到 [B, D]，再接 MLP 预测分数；无需在模型内再做序列建模。
  - 若在模型内对 token 做 self-attention：用 attention_mask_from_tok_mask(tok_mask)
    得到 [B,1,1,N_tok]，加到 attention logits 上，避免 padding 参与。
  - 若与音/视做词级融合：用 text_token_timestamps 与 text_word_timestamps 做
    token→word 的 mask 加权 mean-pool，得到 [B, N_word, D] 再与 audio_wav2vec 等融合。
  - 可选：build_dataloader(..., max_len_tok=128) 限制最大 token 数以控制显存与速度。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# CSV 中的 FIS 标签列（连续分数）
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

DEFAULT_FEATURE_SOURCE = "raw"
DEFAULT_FEATURE_CATEGORIES = ("audio", "video", "text")

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
    """归一化样本 ID，兼容标签 CSV 中 `FIs`/`FIS` 等大小写不一致问题。"""
    s = str(sample_id).strip()
    s = re.sub(r"fis_time1", "FIS_Time1", s, flags=re.IGNORECASE)
    s = re.sub(r"fis_time2", "FIS_Time2", s, flags=re.IGNORECASE)
    s = re.sub(r"fis_time3", "FIS_Time3", s, flags=re.IGNORECASE)
    return s


def csv_id_to_basename(csv_id: str) -> str:
    """将 CSV 中的 ID 转为特征文件名 basename（无扩展名）。
    例如: AG0914_FIS_Time1_Jackson -> AG0914_T1_Jackson, AS1031_FIS_Time3_Lauren -> AS1031_T3_Lauren
    """
    s = normalize_sample_id(csv_id)
    s = re.sub(r"FIS_Time1(?=_|$)", "T1", s, flags=re.IGNORECASE)
    s = re.sub(r"FIS_Time2(?=_|$)", "T2", s, flags=re.IGNORECASE)
    s = re.sub(r"FIS_Time3(?=_|$)", "T3", s, flags=re.IGNORECASE)
    return s


def basename_to_csv_id(basename: str) -> str:
    """将特征文件名 basename 转为标签 CSV 中的 ID。
    例如: AG0914_T1_Jackson -> AG0914_FIS_Time1_Jackson
    """
    s = str(basename).strip()
    s = re.sub(r"_T1_", "_FIS_Time1_", s, flags=re.IGNORECASE)
    s = re.sub(r"_T1$", "_FIS_Time1", s, flags=re.IGNORECASE)
    s = re.sub(r"_T2_", "_FIS_Time2_", s, flags=re.IGNORECASE)
    s = re.sub(r"_T2$", "_FIS_Time2", s, flags=re.IGNORECASE)
    s = re.sub(r"_T3_", "_FIS_Time3_", s, flags=re.IGNORECASE)
    s = re.sub(r"_T3$", "_FIS_Time3", s, flags=re.IGNORECASE)
    return normalize_sample_id(s)


def _normalize_feature_categories(feature_categories: list[str] | tuple[str, ...] | str | None) -> list[str]:
    """规范化数据类别配置，支持 None / 逗号分隔字符串 / list。"""
    if feature_categories is None:
        categories = list(DEFAULT_FEATURE_CATEGORIES)
    elif isinstance(feature_categories, str):
        categories = [x.strip().lower() for x in feature_categories.split(",") if x.strip()]
    else:
        categories = [str(x).strip().lower() for x in feature_categories if str(x).strip()]
    valid = {"audio", "video", "text"}
    invalid = [x for x in categories if x not in valid]
    if invalid:
        raise ValueError(f"不支持的数据类别: {invalid}，仅支持 {sorted(valid)}")
    return categories or list(DEFAULT_FEATURE_CATEGORIES)


def _resolve_required_subdirs(
    feature_source: str = DEFAULT_FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> list[str]:
    """根据读取来源与数据类别返回样本完整性检查所需子目录。"""
    categories = _normalize_feature_categories(feature_categories)
    source = str(feature_source).strip().lower()
    if source == "raw":
        mapping = RAW_CATEGORY_TO_REQUIRED_SUBDIRS
    elif source == "aligned":
        mapping = ALIGNED_CATEGORY_TO_REQUIRED_SUBDIRS
    elif source == "packed":
        return []
    else:
        raise ValueError(f"不支持的 feature_source: {feature_source}")
    subdirs: list[str] = []
    for category in categories:
        subdirs.extend(mapping[category])
    return subdirs


def _resolve_reference_subdir(
    feature_source: str = DEFAULT_FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> str | None:
    """选择扫描 basename 时使用的参考目录。"""
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
        raise ValueError(f"不支持的 feature_source: {feature_source}")
    for category in categories:
        return candidates[source][category]
    return candidates[source]["audio"]


def _counselor_name_from_basename(basename: str) -> str:
    """从 Counselor 特征 basename 提取咨询师名（最后一段）。如 AB0511_T1_Jackson -> Jackson。"""
    parts = str(basename).strip().split("_")
    return parts[-1] if parts else ""


def _collect_valid_patient_basenames(
    root: Path,
    patient_role: str,
    subdirs: list[str],
    ref_subdir: str | None,
) -> set[str]:
    """扫描 Patient 目录，返回「六模态特征均存在」的 basename 集合。"""
    patient_root = root / patient_role if patient_role else root
    ref_dir = patient_root if ref_subdir is None else patient_root / ref_subdir
    if not ref_dir.exists():
        return set()

    def is_complete(basename: str) -> bool:
        for sub in subdirs:
            if not (patient_root / sub / f"{basename}.npz").exists():
                return False
        return True

    return {f.stem for f in ref_dir.glob("*.npz") if is_complete(f.stem)}


def _resolve_patient_basename_for_counselor(
    counselor_name: str,
    valid_patient_basenames: set[str],
) -> str | None:
    """根据咨询师名在「有效 Patient basename 集合」中解析出对应的 Patient basename。

    Patient 侧命名已与 dataset/FIS_stimulis_clips 规范对齐（如 Bethany、Jackson，无 " 1.0" 等后缀）。
    匹配顺序：精确 "Name 1.0"（兼容旧数据）-> 精确 "Name" -> 任意以 Name 开头或包含的项（不区分大小写）。
    """
    name = (counselor_name or "").strip()
    if not name or not valid_patient_basenames:
        return None
    # 优先: "Name 1.0"
    candidate = f"{name} 1.0"
    if candidate in valid_patient_basenames:
        return candidate
    # 精确匹配
    if name in valid_patient_basenames:
        return name
    name_lower = name.lower()
    for p in valid_patient_basenames:
        if p.lower() == name_lower or p.lower().startswith(name_lower + " ") or name_lower in p.lower():
            return p
    return None


def collect_valid_ids_from_fea(
    feature_root: str | Path,
    counselor_role: str = "Counselor",
    patient_role: str = "Patient",
    task: int = 1,
    feature_source: str = DEFAULT_FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> list[str]:
    """扫描 FIS_FEA 目录，收集特征完整样本，返回标签 CSV 可用的 ID 列表。

    - task=1：只检查 counselor 特征完整，返回对应 CSV ID。
    - task=2：检查 counselor 与 patient 均完整，返回两者交集对应的 CSV ID（以 counselor ID 为准）。
    - feature_source: "raw" 读取 audio/video/text 原始目录；"aligned" 读取 aligned 目录；"packed" 读取根级单文件 npz。
    """
    root = Path(feature_root)
    counselor_root = root / counselor_role if counselor_role else root
    required_counselor_subdirs = _resolve_required_subdirs(feature_source, feature_categories)
    required_patient_subdirs = list(required_counselor_subdirs)
    ref_subdir = _resolve_reference_subdir(feature_source, feature_categories)
    ref_dir = counselor_root if ref_subdir is None else counselor_root / ref_subdir
    if not ref_dir.exists():
        return []
    basenames = [f.stem for f in ref_dir.glob("*.npz")]

    def is_role_complete(role_path: Path, subdirs: list[str], basename: str) -> bool:
        if not subdirs:
            return (role_path / f"{basename}.npz").exists()
        for sub in subdirs:
            if not (role_path / sub / f"{basename}.npz").exists():
                return False
        return True

    valid_basenames: list[str] = []
    if task == 2:
        patient_root = root / patient_role if patient_role else root
        valid_patient_set = _collect_valid_patient_basenames(
            root, patient_role, required_patient_subdirs, ref_subdir,
        )
    for bn in basenames:
        if not is_role_complete(counselor_root, required_counselor_subdirs, bn):
            continue
        if task == 2:
            counselor_name = _counselor_name_from_basename(bn)
            patient_bn = _resolve_patient_basename_for_counselor(counselor_name, valid_patient_set)
            if not patient_bn or not is_role_complete(
                patient_root, required_patient_subdirs, patient_bn
            ):
                continue
        valid_basenames.append(bn)

    return [basename_to_csv_id(bn) for bn in valid_basenames]


def get_counselor_to_patient_basename_map(
    feature_root: str | Path,
    counselor_role: str = "Counselor",
    patient_role: str = "Patient",
    feature_source: str = DEFAULT_FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> dict[str, str]:
    """扫描 FIS_FEA，返回「Counselor 与 Patient 特征均完整」的 session 的 counselor_id -> patient_basename 映射。

    用于 task=2 时未提供 patient_basename_map 的自动解析（Patient 侧按角色名命名，如 Bethany 1.0）。
    """
    root = Path(feature_root)
    counselor_root = root / counselor_role if counselor_role else root
    required_counselor_subdirs = _resolve_required_subdirs(feature_source, feature_categories)
    required_patient_subdirs = list(required_counselor_subdirs)
    ref_subdir = _resolve_reference_subdir(feature_source, feature_categories)
    ref_dir = counselor_root if ref_subdir is None else counselor_root / ref_subdir
    if not ref_dir.exists():
        return {}

    def is_role_complete(role_path: Path, subdirs: list[str], basename: str) -> bool:
        if not subdirs:
            return (role_path / f"{basename}.npz").exists()
        for sub in subdirs:
            if not (role_path / sub / f"{basename}.npz").exists():
                return False
        return True

    patient_root = root / patient_role if patient_role else root
    valid_patient_set = _collect_valid_patient_basenames(
        root, patient_role, required_patient_subdirs, ref_subdir,
    )
    result: dict[str, str] = {}
    for f in ref_dir.glob("*.npz"):
        bn = f.stem
        if not is_role_complete(counselor_root, required_counselor_subdirs, bn):
            continue
        counselor_name = _counselor_name_from_basename(bn)
        patient_bn = _resolve_patient_basename_for_counselor(counselor_name, valid_patient_set)
        if patient_bn and is_role_complete(patient_root, required_patient_subdirs, patient_bn):
            result[basename_to_csv_id(bn)] = patient_bn
    return result


def write_avalid_csv(
    out_path: str | Path,
    feature_root: str | Path,
    counselor_role: str = "Counselor",
    patient_role: str = "Patient",
    task: int = 1,
    feature_source: str = DEFAULT_FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> int:
    """扫描 FIS_FEA 并将「特征完整」的样本 ID 写入 CSV。返回写入的 ID 数量。"""
    valid_ids = collect_valid_ids_from_fea(
        feature_root,
        counselor_role=counselor_role,
        patient_role=patient_role,
        task=task,
        feature_source=feature_source,
        feature_categories=feature_categories,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ID": valid_ids})
    df.to_csv(out_path, index=False, encoding="utf-8")
    return len(valid_ids)


def load_role_features(
    feature_root: Path,
    role: str,
    basename: str,
    feature_source: str = DEFAULT_FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
) -> dict[str, Any]:
    """从 FIS_FEA 分模态目录加载单角色（Counselor/Patient）特征。

    目录约定:
    - raw: feature_root/{role}/audio|video|text/.../{basename}.npz
    - aligned: feature_root/{role}/aligned|text/.../{basename}.npz
    - packed: feature_root/{role}/{basename}.npz
    若 role 为空字符串，则从 feature_root 下直接读取（单角色根目录）。
    返回字典与 plan.md / FEATURE_FORMAT 对齐：
    - video_openface3 [N_word, D_vid]
    - audio_wav2vec [N_word, 768]
    - audio_librosa [N_word, 93]
    - text_embedding [N_tok, D_text]
    - text_token [N_tok, 3]
    - text_word_timestamps [N_word, 2]
    - text_token_timestamps [N_tok, 2]
    - text_words: list[str]（可选）
    """
    root = Path(feature_root) / role if role else Path(feature_root)
    categories = _normalize_feature_categories(feature_categories)
    source = str(feature_source).strip().lower()
    out: dict[str, Any] = {
        "audio_wav2vec": None,
        "audio_librosa": None,
        "video_openface3": None,
        "text_embedding": None,
        "text_token": None,
        "text_word_timestamps": None,
        "text_token_timestamps": None,
        "text_words": [],
    }

    if source == "packed":
        packed_path = root / f"{basename}.npz"
        if packed_path.exists():
            data = np.load(packed_path, allow_pickle=True)
            for key in [
                "audio_wav2vec",
                "audio_librosa",
                "audio_prosody",
                "video_openface3",
                "video_occ_pad",
                "text_embedding",
                "text_token",
                "text_word_timestamps",
                "text_token_timestamps",
            ]:
                if key in data.files:
                    out[key] = data[key]
            if "text_words" in data.files:
                text_val = data["text_words"]
                out["text_words"] = text_val.tolist() if hasattr(text_val, "tolist") else text_val
        return out

    if "audio" in categories:
        audio_mapping = [
            ("audio_wav2vec", "audio/wav2vec" if source == "raw" else "aligned/audio_wav2vec"),
            ("audio_librosa", "audio/librosa" if source == "raw" else "aligned/audio_librosa"),
        ]
        for key, subdir in audio_mapping:
            p = root / subdir / f"{basename}.npz"
            if p.exists():
                data = np.load(p, allow_pickle=True)
                out[key] = data["features"]
        # 原始目录下额外保留 prosody，供后续模型扩展；旧接口不会受影响。
        if source == "raw":
            prosody_p = root / "audio/prosody" / f"{basename}.npz"
            if prosody_p.exists():
                data = np.load(prosody_p, allow_pickle=True)
                out["audio_prosody"] = data["features"]

    if "video" in categories:
        video_mapping = [
            ("video_openface3", "video/openface3" if source == "raw" else "aligned/video_openface3"),
        ]
        for key, subdir in video_mapping:
            p = root / subdir / f"{basename}.npz"
            if p.exists():
                data = np.load(p, allow_pickle=True)
                out[key] = data["features"]
        if source == "raw":
            occ_p = root / "video/occ_pad" / f"{basename}.npz"
            if occ_p.exists():
                data = np.load(occ_p, allow_pickle=True)
                out["video_occ_pad"] = data["features"]

    if "text" in categories:
        for key, subdir, npz_key in [
            ("text_embedding", "text/embedding", "features"),
            ("text_token", "text/token", "features"),
        ]:
            p = root / subdir / f"{basename}.npz"
            if p.exists():
                data = np.load(p, allow_pickle=True)
                out[key] = data[npz_key]
                if "timestamps" in data.files:
                    out["text_token_timestamps" if key == "text_token" else "_tok_ts"] = data["timestamps"]

        words_p = root / "text/words" / f"{basename}.npz"
        if words_p.exists():
            data = np.load(words_p, allow_pickle=True)
            out["text_word_timestamps"] = data["timestamps"]
            text_val = data["text"]
            out["text_words"] = text_val.tolist() if hasattr(text_val, "tolist") and np.issubdtype(text_val.dtype, object) else []

    if "_tok_ts" in out:
        out["text_token_timestamps"] = out.pop("_tok_ts")
    return out


def _to_tensor(x: np.ndarray | None, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
    if x is None:
        return None
    t = torch.from_numpy(np.asarray(x))
    if t.dtype in (torch.float64, torch.float32) and dtype == torch.float32:
        t = t.float()
    elif dtype == torch.long and t.dtype != torch.long:
        t = t.long()
    return t


def _numpy_to_tensors(feats: dict[str, Any]) -> dict[str, Any]:
    """将特征字典中的 ndarray 转为 Tensor，保留 list 等。"""
    result: dict[str, Any] = {}
    for k, v in feats.items():
        if k == "text_words":
            result[k] = v
            continue
        if isinstance(v, np.ndarray):
            if v.dtype == object or v.dtype.kind == "O":
                result[k] = v
            else:
                result[k] = _to_tensor(v, dtype=torch.long if k == "text_token" else torch.float32)
        else:
            result[k] = v
    return result


class FISDataset(Dataset):
    """FIS 多模态数据集：从特征根目录 + CSV 标签加载，支持 task=1（仅咨询师）与 task=2（咨询师+来访者）。"""

    def __init__(
        self,
        csv_path: str | Path,
        feature_root: str | Path,
        task: int = 1,
        label_columns: list[str] | None = None,
        counselor_role: str = "Counselor",
        patient_role: str = "Patient",
        feature_source: str = DEFAULT_FEATURE_SOURCE,
        feature_categories: list[str] | tuple[str, ...] | str | None = None,
        patient_basename_map: dict[str, str] | pd.DataFrame | Path | None = None,
        valid_id_csv: str | Path | None = None,
        transform: Optional[Any] = None,
    ):
        """
        Args:
            csv_path: 标签 CSV 路径（含 ID 与 FIS 分数列）。
            feature_root: 特征根目录（如 preprocess/FIS_FEA），下含 Counselor/、Patient/ 等。
            task: 1=仅咨询师特征，2=咨询师+来访者特征。
            label_columns: 使用的标签列，默认 LABEL_COLUMNS。
            counselor_role: 咨询师特征子目录名（如 "Counselor"）。若特征根下直接为 aligned/text 无角色子目录，传 ""。
            patient_role: 来访者特征子目录名（如 "Patient"）。
            feature_source: 特征读取来源。"raw" 从 audio/video/text 目录读取；"aligned" 从 aligned 目录读取；
                "packed" 从角色根目录下的单文件 npz 读取。
            feature_categories: 指定读取的数据类别，可为 ["text"]、["video"]、["audio"] 或它们的组合。
            patient_basename_map: 仅 task=2 时使用。咨询师 sample_id -> 来访者 basename 的映射；
                可为 dict、或含 counselor_id/patient_basename 的 DataFrame、或该表的 CSV 路径。
                若为 None，则尝试用与咨询师相同的 basename 在 Patient/ 下查找。
            valid_id_csv: 可选。仅包含「特征完整」样本 ID 的 CSV 路径（须有列 ID）。
                若提供，则只保留标签 CSV 中 ID 出现在该列表里的行，用于训练/评估。
            transform: 可选的样本级变换（如归一化）。
        """
        self.csv_path = Path(csv_path)
        self.feature_root = Path(feature_root)
        self.task = int(task)
        self.label_columns = label_columns or LABEL_COLUMNS
        self.counselor_role = counselor_role
        self.patient_role = patient_role
        self.feature_source = str(feature_source).strip().lower()
        self.feature_categories = _normalize_feature_categories(feature_categories)
        self.transform = transform

        self.df = pd.read_csv(self.csv_path)
        if "ID" not in self.df.columns:
            raise ValueError("CSV 需包含列 'ID'")

        if valid_id_csv is not None:
            vpath = Path(valid_id_csv)
            if vpath.exists():
                valid_df = pd.read_csv(vpath)
                if "ID" not in valid_df.columns:
                    raise ValueError("valid_id_csv 需包含列 'ID'")
                valid_ids = {normalize_sample_id(x) for x in valid_df["ID"].astype(str)}
                self.df = self.df[
                    self.df["ID"].astype(str).map(normalize_sample_id).isin(valid_ids)
                ].reset_index(drop=True)

        self._patient_map = self._resolve_patient_basename_map(patient_basename_map)
        # task=2 时只保留在 _patient_map 中的 ID，避免出现「找不到 patient」的样本
        if self.task == 2 and self._patient_map:
            map_ids = set(self._patient_map)
            self.df = self.df[
                self.df["ID"].astype(str).map(normalize_sample_id).isin(map_ids)
            ].reset_index(drop=True)

    def _resolve_patient_basename_map(
        self,
        patient_basename_map: dict[str, str] | pd.DataFrame | Path | None,
    ) -> dict[str, str]:
        if self.task != 2:
            return {}
        # 未提供映射时：根据 Counselor 名与 Patient 侧命名规则自动扫描得到 counselor_id -> patient_basename
        if patient_basename_map is None:
            return get_counselor_to_patient_basename_map(
                self.feature_root,
                counselor_role=self.counselor_role,
                patient_role=self.patient_role,
                feature_source=self.feature_source,
                feature_categories=self.feature_categories,
            )
        if isinstance(patient_basename_map, dict):
            return {normalize_sample_id(k): str(v).strip() for k, v in patient_basename_map.items()}
        if isinstance(patient_basename_map, Path):
            patient_basename_map = pd.read_csv(patient_basename_map)
        if isinstance(patient_basename_map, pd.DataFrame):
            df = patient_basename_map
            if "counselor_id" in df.columns and "patient_basename" in df.columns:
                return {
                    normalize_sample_id(k): str(v).strip()
                    for k, v in zip(df["counselor_id"].astype(str), df["patient_basename"].astype(str))
                }
            if "ID" in df.columns and "patient_basename" in df.columns:
                return {
                    normalize_sample_id(k): str(v).strip()
                    for k, v in zip(df["ID"].astype(str), df["patient_basename"].astype(str))
                }
        return {}

    def __len__(self) -> int:
        return len(self.df)

    def _get_counselor_basename(self, csv_id: str) -> str:
        return csv_id_to_basename(csv_id)

    def _get_patient_basename(self, csv_id: str, counselor_basename: str) -> str | None:
        """仅当映射中存在该 ID 时返回 patient basename；否则返回 None，避免用 counselor_basename 去 Patient 下误找。"""
        norm_id = normalize_sample_id(csv_id)
        if norm_id in self._patient_map:
            return self._patient_map[norm_id]
        return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        csv_id = str(row["ID"]).strip()
        counselor_basename = self._get_counselor_basename(csv_id)

        # 标签：连续分数
        labels = {}
        for col in self.label_columns:
            if col in row:
                val = row[col]
                if pd.isna(val):
                    val = 0.0
                labels[col] = float(val)
        labels_tensor = torch.tensor([labels.get(c, 0.0) for c in self.label_columns], dtype=torch.float32)

        # 咨询师特征
        counselor_feats = load_role_features(
            self.feature_root,
            self.counselor_role,
            counselor_basename,
            feature_source=self.feature_source,
            feature_categories=self.feature_categories,
        )
        counselor_feats = _numpy_to_tensors(counselor_feats)

        sample: dict[str, Any] = {
            "sample_id": csv_id,
            "counselor_basename": counselor_basename,
            "labels": labels_tensor,
            "label_dict": labels,
            "counselor": counselor_feats,
        }

        if self.task == 2:
            patient_basename = self._get_patient_basename(csv_id, counselor_basename)
            if patient_basename:
                patient_feats = load_role_features(
                    self.feature_root,
                    self.patient_role,
                    patient_basename,
                    feature_source=self.feature_source,
                    feature_categories=self.feature_categories,
                )
                sample["patient"] = _numpy_to_tensors(patient_feats)
                sample["patient_basename"] = patient_basename
            else:
                sample["patient"] = None
                sample["patient_basename"] = None

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def _pad_and_stack(
    batch_tensors: list[torch.Tensor | None],
    seq_dim: int,
    max_len: int,
    pad_value: float = 0.0,
) -> torch.Tensor | None:
    """将 batch 内变长 tensor 对齐到 max_len 并 stack；None 用零张量填充。
    若同一 key 下不同样本的特征维不一致，会按 batch 内最大特征维对齐并补 pad_value。"""
    valid = [t for t in batch_tensors if t is not None and t.numel() > 0]
    if not valid:
        return None
    ref = valid[0]
    dtype = ref.dtype
    B = len(batch_tensors)
    if ref.ndim == 2:
        # 使用 batch 内最大特征维，避免“Target size 235 vs 200”类错误（同一 key 不同样本 feat 维不一致）
        feat_dim = max(t.size(1) for t in valid)
        out = torch.full((B, max_len, feat_dim), pad_value, dtype=dtype)
    else:
        out = torch.full((B, max_len), pad_value, dtype=dtype)
    for i, t in enumerate(batch_tensors):
        if t is not None and t.numel() > 0:
            L = min(t.size(seq_dim), max_len)
            if t.ndim == 2:
                # 只拷贝 min(L, t.size(0)) 与 min(feat_dim, t.size(1))，避免越界或形状不匹配
                d = min(t.size(1), feat_dim)
                out[i, :L, :d] = t[:L, :d]
            else:
                out[i, :L] = t[:L]
    return out


def pad_mask(lengths: list[int], max_len: int) -> torch.Tensor:
    """生成 [B, max_len] 的 True/False mask，True 表示有效位置。"""
    B = len(lengths)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, L in enumerate(lengths):
        if L > 0:
            mask[i, : min(L, max_len)] = True
    return mask


def aggregate_text_embedding(
    text_emb: torch.Tensor,
    tok_mask: torch.Tensor,
    method: str = "mean_pool",
) -> torch.Tensor:
    """将变长 token 序列聚合成定长向量，供训练时接预测头。

    Args:
        text_emb: [B, N_tok, D] 文本嵌入（含 padding）。
        tok_mask: [B, N_tok] True=有效 token，False=padding。
        method: "mean_pool" 对有效 token 做加权平均；"cls" 取首 token（视为 [CLS]）。

    Returns:
        [B, D] 的序列级文本表示。
    """
    if method == "cls":
        return text_emb[:, 0, :]
    # mean_pool: 仅对有效位置求平均
    mask_float = tok_mask.to(text_emb.dtype).unsqueeze(-1)
    sum_emb = (text_emb * mask_float).sum(dim=1)
    count = tok_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return sum_emb / count


def attention_mask_from_tok_mask(tok_mask: torch.Tensor) -> torch.Tensor:
    """将 counselor_tok_mask 转为注意力用的 mask（padding 位置为 -inf）。

    Args:
        tok_mask: [B, N_tok]，True=有效，False=padding。

    Returns:
        [B, 1, 1, N_tok] 或 [B, 1, N_tok]，无效位置为 -inf，可直接加到 attention logits 上。
    """
    # True -> 0, False -> -inf
    return torch.where(tok_mask.unsqueeze(1).unsqueeze(2), 0.0, float("-inf")).to(tok_mask.device)


def _sequence_length(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, torch.Tensor):
        return int(x.size(0)) if x.ndim > 0 else 0
    if isinstance(x, np.ndarray):
        return int(x.shape[0]) if x.ndim > 0 else 0
    if isinstance(x, (list, tuple)):
        return len(x)
    return 0


def _infer_role_word_length(role_dict: dict[str, Any] | None) -> int:
    if not role_dict:
        return 0
    candidate_keys = [
        "audio_wav2vec",
        "video_openface3",
        "audio_librosa",
        "text_word_timestamps",
    ]
    return max(_sequence_length(role_dict.get(k)) for k in candidate_keys)


def _infer_role_tok_length(role_dict: dict[str, Any] | None) -> int:
    if not role_dict:
        return 0
    candidate_keys = ["text_embedding", "text_token", "text_token_timestamps"]
    return max(_sequence_length(role_dict.get(k)) for k in candidate_keys)


def collate_fis_batch(
    batch: list[dict],
    pad_value: float = 0.0,
    max_len_word: int | None = None,
    max_len_tok: int | None = None,
) -> dict[str, Any]:
    """将 list of samples 转为 batched dict；对变长序列做 padding 到 batch 内最大长度或 max_len_*。"""
    if not batch:
        return {}

    first = batch[0]
    has_patient = first.get("patient") is not None

    keys_word = ["video_openface3", "audio_wav2vec", "audio_librosa", "text_word_timestamps"]
    keys_tok = ["text_embedding", "text_token", "text_token_timestamps"]

    counselor_word_lens = []
    counselor_tok_lens = []
    for s in batch:
        c = s["counselor"]
        counselor_word_lens.append(_infer_role_word_length(c))
        counselor_tok_lens.append(_infer_role_tok_length(c))

    n_word = max(counselor_word_lens, default=0)
    if max_len_word is not None and max_len_word > 0:
        n_word = min(n_word, max_len_word)
    n_word = max(n_word, 1)  # Mamba 等模块要求序列长度至少为 1，避免 ZeroDivisionError
    n_tok = max(counselor_tok_lens, default=0)
    if max_len_tok is not None and max_len_tok > 0:
        n_tok = min(n_tok, max_len_tok)
    n_tok = max(n_tok, 1)

    out: dict[str, Any] = {
        "sample_id": [s["sample_id"] for s in batch],
        "labels": torch.stack([s["labels"] for s in batch], dim=0),
        "label_dict": [s["label_dict"] for s in batch],
        "counselor": {},
        "counselor_word_mask": pad_mask(counselor_word_lens, n_word),
        "counselor_tok_mask": pad_mask(counselor_tok_lens, n_tok),
    }

    for k in keys_word:
        tensors = [s["counselor"].get(k) for s in batch]
        stacked = _pad_and_stack(tensors, 0, n_word, pad_value)
        if stacked is not None:
            out["counselor"][k] = stacked
    for k in keys_tok:
        tensors = [s["counselor"].get(k) for s in batch]
        stacked = _pad_and_stack(tensors, 0, n_tok, pad_value)
        if stacked is not None:
            out["counselor"][k] = stacked
    out["counselor"]["text_words"] = [s["counselor"].get("text_words", []) for s in batch]

    if has_patient:
        patient_word_lens = []
        patient_tok_lens = []
        for s in batch:
            p = s.get("patient")
            patient_word_lens.append(_infer_role_word_length(p))
            patient_tok_lens.append(_infer_role_tok_length(p))
        n_word_p = max(patient_word_lens, default=0)
        if max_len_word is not None and max_len_word > 0:
            n_word_p = min(n_word_p, max_len_word)
        n_word_p = max(n_word_p, 1)
        n_tok_p = max(patient_tok_lens, default=0)
        if max_len_tok is not None and max_len_tok > 0:
            n_tok_p = min(n_tok_p, max_len_tok)
        n_tok_p = max(n_tok_p, 1)
        out["patient"] = {}
        out["patient_word_mask"] = pad_mask(patient_word_lens, n_word_p)
        out["patient_tok_mask"] = pad_mask(patient_tok_lens, n_tok_p)
        for k in keys_word:
            tensors = [s.get("patient", {}).get(k) if s.get("patient") else None for s in batch]
            stacked = _pad_and_stack(tensors, 0, n_word_p, pad_value)
            if stacked is not None:
                out["patient"][k] = stacked
        for k in keys_tok:
            tensors = [s.get("patient", {}).get(k) if s.get("patient") else None for s in batch]
            stacked = _pad_and_stack(tensors, 0, n_tok_p, pad_value)
            if stacked is not None:
                out["patient"][k] = stacked
        out["patient"]["text_words"] = [s.get("patient", {}).get("text_words", []) for s in batch]
    else:
        out["patient"] = None

    return out


def build_dataloader(
    csv_path: str | Path,
    feature_root: str | Path,
    task: int = 1,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    label_columns: list[str] | None = None,
    feature_source: str = DEFAULT_FEATURE_SOURCE,
    feature_categories: list[str] | tuple[str, ...] | str | None = None,
    patient_basename_map: dict[str, str] | pd.DataFrame | Path | None = None,
    valid_id_csv: str | Path | None = None,
    max_len_word: int | None = None,
    max_len_tok: int | None = None,
):
    """构建 FIS DataLoader。"""
    dataset = FISDataset(
        csv_path=csv_path,
        feature_root=feature_root,
        task=task,
        label_columns=label_columns,
        feature_source=feature_source,
        feature_categories=feature_categories,
        patient_basename_map=patient_basename_map,
        valid_id_csv=valid_id_csv,
    )
    collate = lambda b: collate_fis_batch(b, max_len_word=max_len_word, max_len_tok=max_len_tok)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,
    )


def main() -> None:
    """测试 dataloader，或扫描 FIS_FEA 生成 avalid.csv（--write-avalid）。"""
    import argparse
    import sys

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "experiment" else script_dir
    default_csv = project_root / "dataset" / "all_labels_Valid.csv"
    default_fea = project_root / "preprocess" / "FIS_FEA"
    default_avalid = project_root / "dataset" / "avalid.csv"

    parser = argparse.ArgumentParser(description="FIS dataloader 测试或生成有效样本 ID 表")
    parser.add_argument("--write-avalid", action="store_true", help="扫描 FIS_FEA 并将特征完整的样本 ID 写入 CSV")
    parser.add_argument("--avalid-out", type=str, default=str(default_avalid), help="avalid CSV 输出路径（默认 dataset/avalid.csv）")
    parser.add_argument("--feature-root", type=str, default=str(default_fea), help="特征根目录")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2], help="task=2 时同时检查 Patient 特征")
    parser.add_argument("--feature-source", type=str, default=DEFAULT_FEATURE_SOURCE, choices=["raw", "aligned", "packed"], help="特征来源")
    parser.add_argument("--feature-categories", type=str, default="audio,video,text", help="读取的数据类别，逗号分隔，如 text 或 audio,video")
    args = parser.parse_args()
    feature_categories = _normalize_feature_categories(args.feature_categories)

    if args.write_avalid:
        n = write_avalid_csv(
            args.avalid_out,
            feature_root=args.feature_root,
            counselor_role="Counselor",
            patient_role="Patient",
            task=args.task,
            feature_source=args.feature_source,
            feature_categories=feature_categories,
        )
        print(f"已写入 {n} 个有效 ID 到 {args.avalid_out}")
        return

    csv_path = default_csv
    fea_root = default_fea
    if not csv_path.exists():
        print(f"未找到 CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print("=== 1. ID 映射 ===")
    for sid in ["AG0914_FIS_Time1_Jackson", "AS1031_FIS_Time3_Lauren", "EB0819_FIS_Time2_Jackson"]:
        print(f"  {sid} -> {csv_id_to_basename(sid)}")

    print("\n=== 2. 任务一 Dataset（FIS_FEA/Counselor）===")
    ds1 = FISDataset(csv_path, fea_root, task=1, feature_source=args.feature_source, feature_categories=feature_categories)
    print(f"  样本数: {len(ds1)}")
    n_with_feat = sum(1 for i in range(len(ds1)) if ds1[i]["counselor"].get("audio_wav2vec") is not None)
    print(f"  含 audio_wav2vec 的样本数: {n_with_feat}")
    if n_with_feat > 0:
        idx = next(i for i in range(len(ds1)) if ds1[i]["counselor"].get("audio_wav2vec") is not None)
        s = ds1[idx]
        c = s["counselor"]
        n_word = c["audio_wav2vec"].size(0) if c.get("audio_wav2vec") is not None else 0
        n_tok = c["text_embedding"].size(0) if c.get("text_embedding") is not None else 0
        print(f"  示例 sample_id: {s['sample_id']}, labels.shape: {s['labels'].shape}")
        print(f"    原文本时间步: N_word={n_word}, N_tok={n_tok}")
        for k in ["video_openface3", "audio_wav2vec", "audio_librosa", "text_embedding"]:
            v = c.get(k)
            print(f"    counselor.{k}: {v.shape if v is not None else None}")

    print("\n=== 3. 任务一 Dataset（FIS_FEA/sample，无角色子目录）===")
    sample_root = fea_root / "sample"
    if sample_root.exists():
        ds_sample = FISDataset(
            csv_path, sample_root, task=1, counselor_role="", feature_source=args.feature_source, feature_categories=feature_categories,
        )
        for i in range(len(ds_sample)):
            s = ds_sample[i]
            if s["counselor"].get("audio_wav2vec") is not None:
                c = s["counselor"]
                n_w = c["audio_wav2vec"].size(0)
                n_t = c["text_embedding"].size(0) if c.get("text_embedding") is not None else 0
                print(f"  示例: {s['sample_id']} -> {s['counselor_basename']}, audio_wav2vec {s['counselor']['audio_wav2vec'].shape}")
                print(f"    原文本时间步: N_word={n_w}, N_tok={n_t}")
                break
        else:
            print("  sample 下无匹配特征")
    else:
        print(f"  跳过（目录不存在: {sample_root}）")

    print("\n=== 4. Collate 小 batch ===")
    batch_list = []
    for i in range(len(ds1)):
        if ds1[i]["counselor"].get("audio_wav2vec") is not None:
            batch_list.append(ds1[i])
            if len(batch_list) >= 3:
                break
    if batch_list:
        print(f"  batch 大小: {len(batch_list)}")
        for i, s in enumerate(batch_list):
            c = s["counselor"]
            n_word = c["audio_wav2vec"].size(0) if c.get("audio_wav2vec") is not None else 0
            n_tok = c["text_embedding"].size(0) if c.get("text_embedding") is not None else 0
            print(f"    样本 {i} ({s['sample_id']}): 原文本时间步 N_word={n_word}, N_tok={n_tok}")
        out = collate_fis_batch(batch_list)
        print(f"  keys: {list(out.keys())}")
        for k, v in out.get("counselor", {}).items():
            if hasattr(v, "shape"):
                print(f"  counselor.{k}: {v.shape}")
    else:
        print("  无可用样本，跳过 collate")

    print("\n=== 5. DataLoader 迭代 1 个 batch ===")
    dl = build_dataloader(
        csv_path,
        fea_root,
        task=1,
        batch_size=4,
        shuffle=False,
        feature_source=args.feature_source,
        feature_categories=feature_categories,
    )
    for batch in dl:
        print(f"  labels: {batch['labels'].shape}")
        c = batch.get("counselor", {})
        for k in ["audio_wav2vec", "video_openface3", "text_embedding"]:
            if k in c and c[k] is not None:
                print(f"  counselor.{k}: {c[k].shape}")
                break
        break
    else:
        print("  无 batch（数据集为空或无特征）")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()
