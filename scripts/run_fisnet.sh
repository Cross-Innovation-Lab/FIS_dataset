#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  FIS-Net 训练 / 评估 启动脚本
#
#  预设: task1, task2（对应 fisnet_task1 / fisnet_task2）
#
#  最大并行数: train_all 同时运行的任务数上限（默认 2）
#  CUDA 设备（可覆盖）:
#    FIS_DEVICE=0        # 单模型 train/eval 使用的 GPU（默认 0）
#    FIS_DEVICES=0,1     # train_all 时各进程使用的 GPU，逗号分隔（至少 MAX_PARALLEL 个）
#
#  用法:
#    单模型训练:   bash scripts/run_fisnet.sh train task1
#    单模型后台:   bash scripts/run_fisnet.sh train task2 --bg
#    单模型评估:   bash scripts/run_fisnet.sh eval  task1 [ckpt_path] [split]   # 不写 ckpt 则用 checkpoints/<run_name>/best.pt
#
#    并行训练全部: bash scripts/run_fisnet.sh train_all [--wait]
#    仅测试全部:   bash scripts/run_fisnet.sh test_all
#
#  直接传配置路径: bash scripts/run_fisnet.sh train /path/to/custom.json
#
#  终端输出日志:   outs/<run_name>_console.log
#  最优 checkpoint: checkpoints/<run_name>/best.pt
# ============================================================================

PROJECT_ROOT="/CIL_PROJECTS/CODES/MM_FIS"
CONFIG_DIR="${PROJECT_ROOT}/experiment/configs"
OUTS_DIR="${PROJECT_ROOT}/outs-baselines"
CKPT_DIR="${PROJECT_ROOT}/checkpoints"

# 单模型 train/eval 使用的 GPU 编号（可设环境变量 FIS_DEVICE 覆盖，默认 0）
FIS_DEVICE="${FIS_DEVICE:-1}"
# train_all 最大并行任务数（可设 MAX_PARALLEL 覆盖）
MAX_PARALLEL="${MAX_PARALLEL:-2}"
# train_all 时各进程使用的 GPU 列表，逗号分隔（可设 FIS_DEVICES 覆盖，建议不少于 MAX_PARALLEL 个）
FIS_DEVICES="${FIS_DEVICES:-0,1}"
IFS=',' read -ra DEVICE_LIST <<< "$FIS_DEVICES"

# 预设列表：task -> config, run_name
ALL_PRESETS=(task1 task2)
declare -A PRESET_CONFIG
declare -A PRESET_RUN_NAME
PRESET_CONFIG[task1]="${CONFIG_DIR}/fisnet_task1.json"
PRESET_RUN_NAME[task1]="fisnet_task1"
PRESET_CONFIG[task2]="${CONFIG_DIR}/fisnet_task2.json"
PRESET_RUN_NAME[task2]="fisnet_task2"

# 根据第二个参数解析：预设 task 或配置文件路径
resolve_config_and_run_name() {
  local preset="$1"
  if [[ -n "${PRESET_CONFIG[$preset]:-}" ]]; then
    CONFIG="${PRESET_CONFIG[$preset]}"
    RUN_NAME="${PRESET_RUN_NAME[$preset]}"
    return 0
  fi
  if [[ -f "$preset" ]]; then
    CONFIG="$preset"
    RUN_NAME=""
    return 0
  fi
  return 1
}

cd "${PROJECT_ROOT}"

CMD="${1:-}"
ARG2="${2:-}"
ARG3="${3:-}"

usage() {
  echo "用法:"
  echo "  单模型训练   train <preset> [--bg]"
  echo "  单模型评估   eval <preset> [ckpt_path] [split]   # 不写 ckpt 则用 checkpoints/<run_name>/best.pt"
  echo "  并行训练全部 train_all [--wait]                 # --wait 表示等训练结束后自动 test_all"
  echo "  仅测试全部   test_all"
  echo "预设: ${ALL_PRESETS[*]}"
  echo "或传入配置文件路径代替 preset。"
  echo "CUDA: 单模型用 FIS_DEVICE（默认 0）；train_all 用 FIS_DEVICES（默认 0,1）。样本由标签文件决定，不再使用 avalid.csv。"
}

if [[ -z "${CMD}" || "${CMD}" == "--help" || "${CMD}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${CMD}" == "train" && -z "${ARG2}" ]]; then
  ARG2="task1"
fi

# ---------------------------------------------------------------------------
#  train_all [--wait]：并行启动 task1 / task2 训练，可选等待结束后跑 test_all
# ---------------------------------------------------------------------------
if [[ "${CMD}" == "train_all" ]]; then
  DO_WAIT=false
  [[ "${ARG2:-}" == "--wait" ]] && DO_WAIT=true

  PIDS=()
  idx=0
  for preset in "${ALL_PRESETS[@]}"; do
    # 达到最大并行数时先等待当前批完成再启动新任务
    if [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]]; then
      echo "[train_all] 已达最大并行数 ${MAX_PARALLEL}，等待当前批完成..."
      for p in "${PIDS[@]}"; do wait "$p" || true; done
      PIDS=()
    fi
    resolve_config_and_run_name "$preset" || true
    run_name="${PRESET_RUN_NAME[$preset]}"
    log="${OUTS_DIR}/${run_name}_console.log"
    mkdir -p "${OUTS_DIR}"
    dev="${DEVICE_LIST[$((idx % ${#DEVICE_LIST[@]}))]}"
    idx=$((idx + 1))
    echo "[train_all] 启动 ${preset} -> ${run_name} device=cuda:${dev} (config: ${CONFIG})"
    nohup python -m experiment.run train --config "${CONFIG}" --device "cuda:${dev}" > "${log}" 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "  PID=$pid  cuda:${dev}  日志=${log}"
  done

  echo ""
  echo "已启动 ${#ALL_PRESETS[@]} 个训练任务（最大并行 ${MAX_PARALLEL}）。当前 PIDs: ${PIDS[*]}"
  echo "等待: wait ${PIDS[*]}"
  echo "查看某日志: tail -f ${OUTS_DIR}/<run_name>_console.log"

  # 等待最后一批（或全部）训练进程结束
  for p in "${PIDS[@]}"; do wait "$p" || true; done

  if [[ "$DO_WAIT" == true ]]; then
    echo ""
    echo "全部训练已结束，开始对预设模型做 test 集评估（device=cuda:${FIS_DEVICE}）..."
    for preset in "${ALL_PRESETS[@]}"; do
      run_name="${PRESET_RUN_NAME[$preset]}"
      config="${PRESET_CONFIG[$preset]}"
      ckpt="${CKPT_DIR}/${run_name}/best.pt"
      if [[ -f "${ckpt}" ]]; then
        echo "[test_all] ${preset} -> ${ckpt}"
        python -m experiment.run eval --ckpt "${ckpt}" --config "${config}" --split test --device "cuda:${FIS_DEVICE}" || true
      else
        echo "[test_all] ${preset} 跳过（无 best.pt: ${ckpt}）"
      fi
    done
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
#  test_all：对 task1 / task2 用 best.pt 在 test 集上评估
# ---------------------------------------------------------------------------
if [[ "${CMD}" == "test_all" ]]; then
  for preset in "${ALL_PRESETS[@]}"; do
    run_name="${PRESET_RUN_NAME[$preset]}"
    config="${PRESET_CONFIG[$preset]}"
    ckpt="${CKPT_DIR}/${run_name}/best.pt"
    if [[ -f "${ckpt}" ]]; then
      echo "[test_all] ${preset} (${run_name}) device=cuda:${FIS_DEVICE}"
      python -m experiment.run eval --ckpt "${ckpt}" --config "${config}" --split test --device "cuda:${FIS_DEVICE}" || true
    else
      echo "[test_all] ${preset} 跳过（无 best.pt: ${ckpt}）"
    fi
  done
  exit 0
fi

# ---------------------------------------------------------------------------
#  单模型：train / eval
# ---------------------------------------------------------------------------
PRESET="${ARG2:-task1}"
EXTRA="${ARG3:-}"

if ! resolve_config_and_run_name "$PRESET"; then
  echo "ERROR: 未知 preset '${PRESET}'。可选:"
  echo "  ${ALL_PRESETS[*]}"
  echo "  或直接传入配置文件路径。"
  exit 1
fi

echo "== FIS-Net 实验 =="
echo "CMD    : ${CMD}"
echo "CONFIG : ${CONFIG}"
echo "DEVICE : cuda:${FIS_DEVICE}"
[[ -n "${RUN_NAME:-}" ]] && echo "RUN_NAME: ${RUN_NAME}"
echo ""

if [[ "${CMD}" == "train" ]]; then
  if [[ "${EXTRA}" == "--bg" ]]; then
    run_name="${RUN_NAME:-fisnet}"
    log="${OUTS_DIR}/${run_name}_console.log"
    mkdir -p "${OUTS_DIR}"
    echo "后台训练启动，device=cuda:${FIS_DEVICE}，终端日志: ${log}"
    nohup python -m experiment.run train --config "${CONFIG}" --device "cuda:${FIS_DEVICE}" > "${log}" 2>&1 &
    echo "PID: $!"
    echo "查看日志: tail -f ${log}"
  else
    python -m experiment.run train --config "${CONFIG}" --device "cuda:${FIS_DEVICE}"
  fi

elif [[ "${CMD}" == "eval" ]]; then
  CKPT="${ARG3}"
  SPLIT="${4:-test}"
  if [[ -z "${CKPT}" ]]; then
    if [[ -n "${RUN_NAME:-}" ]]; then
      CKPT="${CKPT_DIR}/${RUN_NAME}/best.pt"
      if [[ -f "${CKPT}" ]]; then
        echo "使用默认 checkpoint: ${CKPT}"
      else
        echo "ERROR: 未指定 checkpoint，且默认不存在: ${CKPT}"
        echo "用法: bash scripts/run_fisnet.sh eval ${PRESET} <ckpt_path> [split]"
        exit 1
      fi
    else
      echo "ERROR: eval 需指定 checkpoint 或使用预设名（如 ${PRESET}）。"
      exit 1
    fi
  fi
  python -m experiment.run eval --ckpt "${CKPT}" --config "${CONFIG}" --split "${SPLIT}" --device "cuda:${FIS_DEVICE}"

else
  echo "ERROR: 未知命令 '${CMD}'。可选: train, eval, train_all, test_all"
  exit 1
fi
