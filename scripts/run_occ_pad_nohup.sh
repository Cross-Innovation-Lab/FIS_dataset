#!/usr/bin/env bash
# =============================================================================
# OCC-PAD 全量视频后台处理脚本
#
# 递归处理 dataset 目录下所有视频（.mp4/.mkv/.webm），结果输出到 out/。
# 使用 nohup 后台运行，日志写入 outs/occ_pad_nohup.log。
#
# 用法:
#   bash scripts/run_occ_pad_nohup.sh              # 后台启动
#   bash scripts/run_occ_pad_nohup.sh --foreground  # 前台运行（便于调试）
#
# 可选环境变量:
#   OCC_PAD_DATASET    - 视频根目录，默认 PROJECT_ROOT/dataset
#   OCC_PAD_OUT        - 输出目录，默认 PROJECT_ROOT/out
#   OCC_PAD_FRAME_FREQ - 帧采样频率，默认 4
#   OCC_PAD_DEVICE     - CUDA 设备，如 cuda:0、cuda:1 或 cpu；不设则自动选择
#   OCC_PAD_JOBS       - 并行处理视频数上限，默认 8
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/CIL_PROJECTS/CODES/MM_FIS"
OCC_PAD_SCRIPT="${PROJECT_ROOT}/lib/OCC-PAD/occ_pad.py"
OUTS_DIR="${PROJECT_ROOT}/results"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/occ_pad_nohup.log"

# 可覆盖的路径与参数
DATASET_DIR="${OCC_PAD_DATASET:-${PROJECT_ROOT}/dataset}"
OUT_DIR="${OCC_PAD_OUT:-${PROJECT_ROOT}/results}"
FRAME_FREQ="${OCC_PAD_FRAME_FREQ:-4}"
# 可选：指定 GPU/CPU，如 OCC_PAD_DEVICE=cuda:1
DEVICE_ARG=""
if [[ -n "${OCC_PAD_DEVICE:-}" ]]; then
  DEVICE_ARG="-d ${OCC_PAD_DEVICE}"
fi
# 并行任务数上限（默认 8）
MAX_PARALLEL="${OCC_PAD_JOBS:-8}"

# 视频扩展名（与 occ_pad.py 中 VIDEO_EXTENSIONS 一致）
VIDEO_EXTS=".mp4 .mkv .webm"

mkdir -p "${OUT_DIR}"
mkdir -p "${OUTS_DIR}"

# 收集 dataset 下全部视频（递归）
collect_videos() {
  find "${DATASET_DIR}" -type f \
    \( -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" \) \
    | sort
}

# 前台/后台
FOREGROUND=false
for arg in "$@"; do
  if [[ "${arg}" == "--foreground" ]]; then
    FOREGROUND=true
    break
  fi
done

run_occ_pad() {
  local video="$1"
  python "${OCC_PAD_SCRIPT}" \
    "${video}" \
    -o "${OUT_DIR}" \
    -c \
    -f "${FRAME_FREQ}" \
    ${DEVICE_ARG}
}

main() {
  echo "========================================"
  echo "OCC-PAD 全量视频处理"
  echo "  视频根目录: ${DATASET_DIR}"
  echo "  输出目录:   ${OUT_DIR}"
  echo "  帧采样:     每 ${FRAME_FREQ} 帧"
  echo "  并行数:     最多 ${MAX_PARALLEL} 个视频同时处理"
  [[ -n "${OCC_PAD_DEVICE:-}" ]] && echo "  设备:       ${OCC_PAD_DEVICE}"
  echo "========================================"

  if [[ ! -f "${OCC_PAD_SCRIPT}" ]]; then
    echo "错误: 未找到 ${OCC_PAD_SCRIPT}" >&2
    exit 1
  fi

  if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "错误: 目录不存在 ${DATASET_DIR}" >&2
    exit 1
  fi

  mapfile -t VIDEOS < <(collect_videos)
  TOTAL=${#VIDEOS[@]}

  if [[ ${TOTAL} -eq 0 ]]; then
    echo "未在 ${DATASET_DIR} 下发现任何视频文件（.mp4/.mkv/.webm）。"
    exit 0
  fi

  echo "共发现 ${TOTAL} 个视频，开始处理（最多 ${MAX_PARALLEL} 路并行）..."
  echo ""

  # 用于记录每个任务的退出码（并行时无法在循环内直接计数）
  TMP_EXIT_DIR=$(mktemp -d)
  trap 'rm -rf "${TMP_EXIT_DIR}"' EXIT

  running=0
  for i in "${!VIDEOS[@]}"; do
    vid="${VIDEOS[$i]}"
    # 达到上限时等待任意一个任务结束（wait -n 可能返回非 0，避免 set -e 导致退出）
    while [[ ${running} -ge ${MAX_PARALLEL} ]]; do
      wait -n || true
      ((running--)) || true
    done
    (
      if run_occ_pad "${vid}"; then
        echo 0
      else
        echo "[$((i + 1))/${TOTAL}] 失败: ${vid}" >&2
        echo 1
      fi
    ) > "${TMP_EXIT_DIR}/${i}.exit" &
    ((running++)) || true
  done
  # 等待剩余任务（任一失败时 wait 可能非 0，不触发 set -e）
  wait || true

  SUCCESS=0
  FAIL=0
  for i in "${!VIDEOS[@]}"; do
    if [[ -f "${TMP_EXIT_DIR}/${i}.exit" ]]; then
      code=$(cat "${TMP_EXIT_DIR}/${i}.exit")
      if [[ "${code}" == "0" ]]; then
        ((SUCCESS++)) || true
      else
        ((FAIL++)) || true
      fi
    fi
  done

  echo ""
  echo "========================================"
  echo "完成: 成功 ${SUCCESS}, 失败 ${FAIL}, 共 ${TOTAL}"
  echo "结果目录: ${OUT_DIR}"
  echo "========================================"
}

cd "${PROJECT_ROOT}"

if [[ "${FOREGROUND}" == "true" ]]; then
  main "$@"
else
  nohup bash "${BASH_SOURCE[0]}" --foreground >> "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "已在后台启动，PID: ${PID}"
  echo "日志: ${LOG_FILE}"
  echo "查看: tail -f ${LOG_FILE}"
  echo "${PID}" > "${OUTS_DIR}/occ_pad_nohup.pid"
  exit 0
fi
