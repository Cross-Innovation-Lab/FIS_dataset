#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  FIS 基线模型 训练 / 评估 启动脚本
#
#  预设模型: BiLSTM+Attention, TimeSformer, TimeFilter, DLinear, SegRNN,
#           KANAD, Reformer, MSGNet（各 task1 / task2）
#
#  CUDA 设备（可覆盖）:
#    FIS_DEVICE=1                    # 单模型 train/eval 使用的 GPU（默认 1）
#    FIS_DEVICES=0,1,2,3             # train_all / train_new / train_task*_all 时轮询使用
#    FIS_MAX_PARALLEL=4              # train_all 时最多同时并行的训练进程数
#
#  用法:
#    单模型训练:   bash scripts/run_baselines.sh train bilstm_task1
#    单模型后台:   bash scripts/run_baselines.sh train timesformer_task2 --bg
#    单模型评估:   bash scripts/run_baselines.sh eval  bilstm_task1 <ckpt_path> [split]
#
#    并行训练全部: bash scripts/run_baselines.sh train_all
#    仅训新模型:   bash scripts/run_baselines.sh train_new
#    仅训 task1:   bash scripts/run_baselines.sh train_task1_all
#    仅训 task2:   bash scripts/run_baselines.sh train_task2_all
#    等待并测试:   bash scripts/run_baselines.sh train_all --wait
#    仅测试全部:   bash scripts/run_baselines.sh test_all
#    仅测新模型:   bash scripts/run_baselines.sh test_new
#    查看预设:     bash scripts/run_baselines.sh list
#
#  直接传配置路径: bash scripts/run_baselines.sh train /path/to/custom.json
# ============================================================================

PROJECT_ROOT="/CIL_PROJECTS/CODES/MM_FIS"
CONFIG_DIR="${PROJECT_ROOT}/experiment/configs"
OUTS_DIR="${PROJECT_ROOT}/outs-baselines"
CKPT_DIR="${PROJECT_ROOT}/checkpoints"

# 单模型 train/eval 使用的 GPU 编号（可设环境变量 FIS_DEVICE 覆盖，默认 0）
FIS_DEVICE="${FIS_DEVICE:-1}"
# train_all 时各进程使用的 GPU 列表，逗号分隔（可设 FIS_DEVICES 覆盖；不足预设数则循环使用）
FIS_DEVICES="${FIS_DEVICES:-1}"
FIS_MAX_PARALLEL="${FIS_MAX_PARALLEL:-3}"
IFS=',' read -ra DEVICE_LIST <<< "$FIS_DEVICES"

# 预设列表：preset_name -> config 文件名（不含路径）-> run_name 与 config 中 experiment.run_name 一致
ALL_PRESETS=(
  bilstm_task1 bilstm_task2
  timesformer_task1 timesformer_task2
  timefilter_task1 timefilter_task2
  dlinear_task1 dlinear_task2
  segrnn_task1 segrnn_task2
  kanad_task1 kanad_task2
  reformer_task1 reformer_task2
  msgnet_task1 msgnet_task2
)
declare -A PRESET_CONFIG
declare -A PRESET_RUN_NAME
PRESET_CONFIG[bilstm_task1]="${CONFIG_DIR}/bilstm_attn_task1.json"
PRESET_RUN_NAME[bilstm_task1]="bilstm_attn_task1"
PRESET_CONFIG[bilstm_task2]="${CONFIG_DIR}/bilstm_attn_task2.json"
PRESET_RUN_NAME[bilstm_task2]="bilstm_attn_task2"
PRESET_CONFIG[timesformer_task1]="${CONFIG_DIR}/timesformer_task1.json"
PRESET_RUN_NAME[timesformer_task1]="timesformer_task1"
PRESET_CONFIG[timesformer_task2]="${CONFIG_DIR}/timesformer_task2.json"
PRESET_RUN_NAME[timesformer_task2]="timesformer_task2"
PRESET_CONFIG[timefilter_task1]="${CONFIG_DIR}/timefilter_task1.json"
PRESET_RUN_NAME[timefilter_task1]="timefilter_task1"
PRESET_CONFIG[timefilter_task2]="${CONFIG_DIR}/timefilter_task2.json"
PRESET_RUN_NAME[timefilter_task2]="timefilter_task2"
PRESET_CONFIG[dlinear_task1]="${CONFIG_DIR}/dlinear_task1.json"
PRESET_RUN_NAME[dlinear_task1]="dlinear_task1"
PRESET_CONFIG[dlinear_task2]="${CONFIG_DIR}/dlinear_task2.json"
PRESET_RUN_NAME[dlinear_task2]="dlinear_task2"
PRESET_CONFIG[segrnn_task1]="${CONFIG_DIR}/segrnn_task1.json"
PRESET_RUN_NAME[segrnn_task1]="segrnn_task1"
PRESET_CONFIG[segrnn_task2]="${CONFIG_DIR}/segrnn_task2.json"
PRESET_RUN_NAME[segrnn_task2]="segrnn_task2"
PRESET_CONFIG[kanad_task1]="${CONFIG_DIR}/kanad_task1.json"
PRESET_RUN_NAME[kanad_task1]="kanad_task1"
PRESET_CONFIG[kanad_task2]="${CONFIG_DIR}/kanad_task2.json"
PRESET_RUN_NAME[kanad_task2]="kanad_task2"
PRESET_CONFIG[reformer_task1]="${CONFIG_DIR}/reformer_task1.json"
PRESET_RUN_NAME[reformer_task1]="reformer_task1"
PRESET_CONFIG[reformer_task2]="${CONFIG_DIR}/reformer_task2.json"
PRESET_RUN_NAME[reformer_task2]="reformer_task2"
PRESET_CONFIG[msgnet_task1]="${CONFIG_DIR}/msgnet_task1.json"
PRESET_RUN_NAME[msgnet_task1]="msgnet_task1"
PRESET_CONFIG[msgnet_task2]="${CONFIG_DIR}/msgnet_task2.json"
PRESET_RUN_NAME[msgnet_task2]="msgnet_task2"

NEW_PRESETS=(
  dlinear_task1 dlinear_task2
  segrnn_task1 segrnn_task2
  kanad_task1 kanad_task2
  reformer_task1 reformer_task2
  msgnet_task1 msgnet_task2
)

TASK1_PRESETS=(
  bilstm_task1 timesformer_task1 timefilter_task1
  dlinear_task1 segrnn_task1 kanad_task1 reformer_task1 msgnet_task1
)

TASK2_PRESETS=(
  bilstm_task2 timesformer_task2 timefilter_task2
  dlinear_task2 segrnn_task2 kanad_task2 reformer_task2 msgnet_task2
)

# 根据第二个参数解析：单 preset 或配置文件路径
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

ensure_config_exists() {
  local preset="$1"
  local config_path="${PRESET_CONFIG[$preset]:-}"
  if [[ -z "${config_path}" || ! -f "${config_path}" ]]; then
    echo "ERROR: preset '${preset}' 对应配置不存在: ${config_path:-<empty>}"
    exit 1
  fi
}

print_preset_group() {
  local title="$1"
  shift
  echo "${title}:"
  for preset in "$@"; do
    echo "  - ${preset}"
  done
}

run_train_group() {
  local group_name="$1"
  shift
  local do_wait="${1:-false}"
  shift || true
  local presets=("$@")

  local pids=()
  local active_pids=()
  local idx=0
  mkdir -p "${OUTS_DIR}"

  for preset in "${presets[@]}"; do
    if [[ "${group_name}" == "train_all" ]]; then
      while [[ "${#active_pids[@]}" -ge "${FIS_MAX_PARALLEL}" ]]; do
        local next_active=()
        for pid in "${active_pids[@]}"; do
          if kill -0 "${pid}" 2>/dev/null; then
            next_active+=("${pid}")
          fi
        done
        active_pids=("${next_active[@]}")
        if [[ "${#active_pids[@]}" -ge "${FIS_MAX_PARALLEL}" ]]; then
          sleep 2
        fi
      done
    fi

    ensure_config_exists "${preset}"
    resolve_config_and_run_name "${preset}"
    local run_name="${PRESET_RUN_NAME[$preset]}"
    local log="${OUTS_DIR}/${run_name}_console.log"
    local dev="${DEVICE_LIST[$((idx % ${#DEVICE_LIST[@]}))]}"
    idx=$((idx + 1))
    echo "[${group_name}] 启动 ${preset} -> ${run_name} device=cuda:${dev} (config: ${CONFIG})"
    nohup python -m experiment.run train --config "${CONFIG}" --device "cuda:${dev}" > "${log}" 2>&1 &
    local pid=$!
    pids+=($pid)
    active_pids+=($pid)
    echo "  PID=${pid}  cuda:${dev}  日志=${log}"
  done

  echo ""
  echo "已并行启动 ${#pids[@]} 个训练进程。PIDs: ${pids[*]}"
  echo "等待: wait ${pids[*]}"
  echo "查看某日志: tail -f ${OUTS_DIR}/<run_name>_console.log"

  if [[ "${do_wait}" == true ]]; then
    echo ""
    echo "等待全部训练完成..."
    for pid in "${pids[@]}"; do
      wait "${pid}" || true
    done
    echo "全部训练已结束，开始执行对应测试..."
    run_test_group "${group_name}" "${presets[@]}"
  fi
}

run_test_group() {
  local group_name="$1"
  shift
  local presets=("$@")
  for preset in "${presets[@]}"; do
    ensure_config_exists "${preset}"
    local run_name="${PRESET_RUN_NAME[$preset]}"
    local config="${PRESET_CONFIG[$preset]}"
    local ckpt="${CKPT_DIR}/${run_name}/best.pt"
    if [[ -f "${ckpt}" ]]; then
      echo "[${group_name}] ${preset} (${run_name}) device=cuda:${FIS_DEVICE}"
      python -m experiment.run eval --ckpt "${ckpt}" --config "${config}" --split test --device "cuda:${FIS_DEVICE}" || true
    else
      echo "[${group_name}] ${preset} 跳过（无 best.pt: ${ckpt}）"
    fi
  done
}

cd "${PROJECT_ROOT}"

CMD="${1:-}"
ARG2="${2:-}"
ARG3="${3:-}"

usage() {
  echo "用法:"
  echo "  单模型训练   train <preset> [--bg]"
  echo "  单模型评估   eval <preset> [ckpt_path] [split]   # 不写 ckpt 则用 checkpoints/<run_name>/best.pt"
  echo "  并行训练全部 train_all [--wait]"
  echo "  仅训新模型   train_new [--wait]"
  echo "  仅训 task1   train_task1_all [--wait]"
  echo "  仅训 task2   train_task2_all [--wait]"
  echo "  仅测试全部   test_all"
  echo "  仅测新模型   test_new"
  echo "  查看预设     list"
  echo ""
  print_preset_group "全部预设" "${ALL_PRESETS[@]}"
  echo ""
  print_preset_group "新加入模型预设" "${NEW_PRESETS[@]}"
  echo "或传入配置文件路径代替 preset。"
  echo "CUDA: 单模型用 FIS_DEVICE（默认 1）；train_all 用 FIS_DEVICES。样本由标签文件决定，不再使用 avalid.csv。"
  echo "并行上限: train_all 最多同时运行 FIS_MAX_PARALLEL 个训练进程（默认 4）。"
}

if [[ -z "${CMD}" || "${CMD}" == "--help" || "${CMD}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${CMD}" == "train" && -z "${ARG2}" ]]; then
  ARG2="bilstm_task1"
fi

if [[ "${CMD}" == "list" ]]; then
  usage
  exit 0
fi

# ---------------------------------------------------------------------------
#  train_all / train_new / train_task1_all / train_task2_all
# ---------------------------------------------------------------------------
if [[ "${CMD}" == "train_all" || "${CMD}" == "train_new" || "${CMD}" == "train_task1_all" || "${CMD}" == "train_task2_all" ]]; then
  DO_WAIT=false
  [[ "${ARG2:-}" == "--wait" ]] && DO_WAIT=true

  if [[ "${CMD}" == "train_all" ]]; then
    run_train_group "train_all" "${DO_WAIT}" "${ALL_PRESETS[@]}"
  elif [[ "${CMD}" == "train_new" ]]; then
    run_train_group "train_new" "${DO_WAIT}" "${NEW_PRESETS[@]}"
  elif [[ "${CMD}" == "train_task1_all" ]]; then
    run_train_group "train_task1_all" "${DO_WAIT}" "${TASK1_PRESETS[@]}"
  elif [[ "${CMD}" == "train_task2_all" ]]; then
    run_train_group "train_task2_all" "${DO_WAIT}" "${TASK2_PRESETS[@]}"
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
#  test_all / test_new：对预设用 best.pt 在 test 集上评估
# ---------------------------------------------------------------------------
if [[ "${CMD}" == "test_all" || "${CMD}" == "test_new" ]]; then
  if [[ "${CMD}" == "test_all" ]]; then
    run_test_group "test_all" "${ALL_PRESETS[@]}"
  else
    run_test_group "test_new" "${NEW_PRESETS[@]}"
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
#  单模型：train / eval
# ---------------------------------------------------------------------------
PRESET="${ARG2:-bilstm_task1}"
EXTRA="${ARG3:-}"

if ! resolve_config_and_run_name "$PRESET"; then
  echo "ERROR: 未知 preset '${PRESET}'。可选:"
  echo "  ${ALL_PRESETS[*]}"
  echo "  或直接传入配置文件路径。"
  exit 1
fi
if [[ -n "${RUN_NAME:-}" ]]; then
  ensure_config_exists "${PRESET}"
fi

echo "== FIS Baselines =="
echo "CMD    : ${CMD}"
echo "CONFIG : ${CONFIG}"
echo "DEVICE : cuda:${FIS_DEVICE}"
[[ -n "${RUN_NAME:-}" ]] && echo "RUN_NAME: ${RUN_NAME}"
echo ""

if [[ "${CMD}" == "train" ]]; then
  if [[ "${EXTRA}" == "--bg" ]]; then
    run_name="${RUN_NAME:-baseline}"
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
  CKPT="${EXTRA}"
  SPLIT="${4:-test}"
  if [[ -z "${CKPT}" ]]; then
    if [[ -n "${RUN_NAME:-}" ]]; then
      CKPT="${CKPT_DIR}/${RUN_NAME}/best.pt"
      if [[ -f "${CKPT}" ]]; then
        echo "使用默认 checkpoint: ${CKPT}"
      else
        echo "ERROR: 未指定 checkpoint，且默认不存在: ${CKPT}"
        echo "用法: bash scripts/run_baselines.sh eval ${PRESET} <ckpt_path> [split]"
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
