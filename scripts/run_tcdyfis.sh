#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  TCDyFIS training / evaluation launcher
#
#  Presets:
#    task1 -> experiment/configs/tcdyfis_task1.json
#    task2 -> experiment/configs/tcdyfis_task2.json
#
#  Usage:
#    bash scripts/run_tcdyfis.sh train task1
#    bash scripts/run_tcdyfis.sh train task2 --bg
#    bash scripts/run_tcdyfis.sh eval  task1 [ckpt_path] [split]
#    bash scripts/run_tcdyfis.sh train_all [--wait]
#    bash scripts/run_tcdyfis.sh test_all
# ============================================================================

PROJECT_ROOT="/CIL_PROJECTS/CODES/MM_FIS"
CONFIG_DIR="${PROJECT_ROOT}/experiment/configs"
OUTS_DIR="${PROJECT_ROOT}/outs-baselines"
CKPT_DIR="${PROJECT_ROOT}/checkpoints"

FIS_DEVICE="${FIS_DEVICE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
FIS_DEVICES="${FIS_DEVICES:-0,1}"
IFS=',' read -ra DEVICE_LIST <<< "$FIS_DEVICES"

ALL_PRESETS=(task1 task2)
declare -A PRESET_CONFIG
declare -A PRESET_RUN_NAME
PRESET_CONFIG[task1]="${CONFIG_DIR}/tcdyfis_task1.json"
PRESET_RUN_NAME[task1]="tcdyfis_task1_grouped"
PRESET_CONFIG[task2]="${CONFIG_DIR}/tcdyfis_task2.json"
PRESET_RUN_NAME[task2]="tcdyfis_task2_opt"

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
  echo "Usage:"
  echo "  train <preset> [--bg]"
  echo "  eval <preset> [ckpt_path] [split]"
  echo "  train_all [--wait]"
  echo "  test_all"
  echo "Presets: ${ALL_PRESETS[*]}"
  echo "You can also pass a config path directly."
}

if [[ -z "${CMD}" || "${CMD}" == "--help" || "${CMD}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${CMD}" == "train" && -z "${ARG2}" ]]; then
  ARG2="task1"
fi

if [[ "${CMD}" == "train_all" ]]; then
  DO_WAIT=false
  [[ "${ARG2:-}" == "--wait" ]] && DO_WAIT=true

  PIDS=()
  idx=0
  mkdir -p "${OUTS_DIR}"
  for preset in "${ALL_PRESETS[@]}"; do
    if [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]]; then
      for p in "${PIDS[@]}"; do wait "$p" || true; done
      PIDS=()
    fi
    resolve_config_and_run_name "$preset" || true
    run_name="${PRESET_RUN_NAME[$preset]}"
    log="${OUTS_DIR}/${run_name}_console.log"
    dev="${DEVICE_LIST[$((idx % ${#DEVICE_LIST[@]}))]}"
    idx=$((idx + 1))
    echo "[train_all] ${preset} -> ${run_name} device=cuda:${dev}"
    nohup python -m experiment.run train --config "${CONFIG}" --device "cuda:${dev}" > "${log}" 2>&1 &
    PIDS+=($!)
  done

  for p in "${PIDS[@]}"; do wait "$p" || true; done

  if [[ "${DO_WAIT}" == true ]]; then
    for preset in "${ALL_PRESETS[@]}"; do
      run_name="${PRESET_RUN_NAME[$preset]}"
      config="${PRESET_CONFIG[$preset]}"
      ckpt="${CKPT_DIR}/${run_name}/best.pt"
      if [[ -f "${ckpt}" ]]; then
        python -m experiment.run eval --ckpt "${ckpt}" --config "${config}" --split test --device "cuda:${FIS_DEVICE}" || true
      fi
    done
  fi
  exit 0
fi

if [[ "${CMD}" == "test_all" ]]; then
  for preset in "${ALL_PRESETS[@]}"; do
    run_name="${PRESET_RUN_NAME[$preset]}"
    config="${PRESET_CONFIG[$preset]}"
    ckpt="${CKPT_DIR}/${run_name}/best.pt"
    if [[ -f "${ckpt}" ]]; then
      echo "[test_all] ${preset} (${run_name}) device=cuda:${FIS_DEVICE}"
      python -m experiment.run eval --ckpt "${ckpt}" --config "${config}" --split test --device "cuda:${FIS_DEVICE}" || true
    else
      echo "[test_all] skip ${preset}, missing ${ckpt}"
    fi
  done
  exit 0
fi

PRESET="${ARG2:-task1}"
EXTRA="${ARG3:-}"

if ! resolve_config_and_run_name "$PRESET"; then
  echo "ERROR: unknown preset '${PRESET}'"
  exit 1
fi

echo "== TCDyFIS =="
echo "CMD    : ${CMD}"
echo "CONFIG : ${CONFIG}"
echo "DEVICE : cuda:${FIS_DEVICE}"
[[ -n "${RUN_NAME:-}" ]] && echo "RUN_NAME: ${RUN_NAME}"
echo ""

if [[ "${CMD}" == "train" ]]; then
  if [[ "${EXTRA}" == "--bg" ]]; then
    run_name="${RUN_NAME:-tcdyfis}"
    log="${OUTS_DIR}/${run_name}_console.log"
    mkdir -p "${OUTS_DIR}"
    nohup python -m experiment.run train --config "${CONFIG}" --device "cuda:${FIS_DEVICE}" > "${log}" 2>&1 &
    echo "PID: $!"
    echo "log: ${log}"
  else
    python -m experiment.run train --config "${CONFIG}" --device "cuda:${FIS_DEVICE}"
  fi
elif [[ "${CMD}" == "eval" ]]; then
  CKPT="${ARG3}"
  SPLIT="${4:-test}"
  if [[ -z "${CKPT}" ]]; then
    CKPT="${CKPT_DIR}/${RUN_NAME}/best.pt"
    if [[ ! -f "${CKPT}" ]]; then
      echo "ERROR: missing checkpoint ${CKPT}"
      exit 1
    fi
  fi
  python -m experiment.run eval --ckpt "${CKPT}" --config "${CONFIG}" --split "${SPLIT}" --device "cuda:${FIS_DEVICE}"
else
  echo "ERROR: unknown command '${CMD}'"
  exit 1
fi
