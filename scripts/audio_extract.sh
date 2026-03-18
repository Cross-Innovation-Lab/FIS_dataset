#!/bin/bash
# 使用 ffmpeg 批量从指定目录下所有 mp4 提取音频，同名保存为 .wav（16kHz 单声道）
# 用法: bash scripts/audio_extract.sh

# 输出采样率（与 pipeline 一致）
SR=16000
# 声道数
CHANNELS=1

ROOTS=(
  /CIL_PROJECTS/CODES/MM_FIS/dataset/testset
  # "/CIL_PROJECTS/CODES/MM_FIS/dataset/Nov2021"
  # "/CIL_PROJECTS/CODES/MM_FIS/dataset/Oct2020"
  # "/CIL_PROJECTS/CODES/MM_FIS/dataset/FIS_stimulis_clips"
)

for ROOT in "${ROOTS[@]}"; do
  if [[ ! -d "$ROOT" ]]; then
    echo "[SKIP] 目录不存在: $ROOT"
    continue
  fi
  echo "[INFO] 处理目录: $ROOT"
  count=0
  for f in "$ROOT"/*.mp4; do
    [[ -f "$f" ]] || continue
    base="${f%.*}"
    out="${base}.wav"
    if [[ -f "$out" ]]; then
      echo "  已存在, 跳过: $(basename "$out")"
    else
      if ffmpeg -y -i "$f" -vn -acodec pcm_s16le -ar "$SR" -ac "$CHANNELS" "$out" -loglevel error; then
        echo "  已提取: $(basename "$out")"
        ((count++)) || true
      else
        echo "  失败: $f" >&2
      fi
    fi
  done
  echo "[INFO] $ROOT 本批次新提取: $count 个"
done
echo "[DONE] 全部完成"
