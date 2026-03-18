#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  TCDyFIS-v2 experiment launcher
#
#  Runs multiple hyperparameter configurations for task2 (and optionally task1),
#  then compares results across all runs.
#
#  Usage:
#    bash scripts/run_tcdyfis_v2.sh train_all          # train all task2 variants
#    bash scripts/run_tcdyfis_v2.sh train task1         # train task1 only
#    bash scripts/run_tcdyfis_v2.sh train A             # train variant A only
#    bash scripts/run_tcdyfis_v2.sh train A --bg        # background
#    bash scripts/run_tcdyfis_v2.sh compare             # compare all results
#    bash scripts/run_tcdyfis_v2.sh train_all --compare # train then compare
#
#  Environment:
#    FIS_DEVICE=0          default GPU
#    FIS_DEVICES=0,1       GPUs for parallel training
#    MAX_PARALLEL=2        max concurrent jobs
# ============================================================================

PROJECT_ROOT="/CIL_PROJECTS/CODES/MM_FIS"
CONFIG_DIR="${PROJECT_ROOT}/experiment/configs"
OUTS_DIR="${PROJECT_ROOT}/outs-baselines"
CKPT_DIR="${PROJECT_ROOT}/checkpoints"

FIS_DEVICE="${FIS_DEVICE:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
FIS_DEVICES="${FIS_DEVICES:-0,1}"
IFS=',' read -ra DEVICE_LIST <<< "$FIS_DEVICES"

# ---- Preset registry ----
# task2 variants A-E + task1
ALL_TASK2_VARIANTS=(A B C D E)
ALL_PRESETS=(task1 A B C D E)

declare -A PRESET_CONFIG
declare -A PRESET_RUN_NAME
declare -A PRESET_DESC

PRESET_CONFIG[task1]="${CONFIG_DIR}/tcdyfis_v2_task1.json"
PRESET_RUN_NAME[task1]="tcdyfis_v2_task1"
PRESET_DESC[task1]="task1 baseline"

PRESET_CONFIG[A]="${CONFIG_DIR}/tcdyfis_v2_task2_A.json"
PRESET_RUN_NAME[A]="tcdyfis_v2_task2_A"
PRESET_DESC[A]="d192 drop0.3 lr2e-4 wd0.02"

PRESET_CONFIG[B]="${CONFIG_DIR}/tcdyfis_v2_task2_B.json"
PRESET_RUN_NAME[B]="tcdyfis_v2_task2_B"
PRESET_DESC[B]="d192 drop0.4 wd0.05 noise0.05"

PRESET_CONFIG[C]="${CONFIG_DIR}/tcdyfis_v2_task2_C.json"
PRESET_RUN_NAME[C]="tcdyfis_v2_task2_C"
PRESET_DESC[C]="d128 comp48 lr1e-4 wd0.02"

PRESET_CONFIG[D]="${CONFIG_DIR}/tcdyfis_v2_task2_D.json"
PRESET_RUN_NAME[D]="tcdyfis_v2_task2_D"
PRESET_DESC[D]="d192 dyadic2 ccc0.4 noise0.02"

PRESET_CONFIG[E]="${CONFIG_DIR}/tcdyfis_v2_task2_E.json"
PRESET_RUN_NAME[E]="tcdyfis_v2_task2_E"
PRESET_DESC[E]="d192 comp96 kern7 noise0.02"

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
  echo "TCDyFIS-v2 Experiment Launcher"
  echo ""
  echo "Usage:"
  echo "  train <preset> [--bg]   Train a single preset"
  echo "  train_all [--compare]   Train all task2 variants (A-E) in parallel"
  echo "  compare                 Compare results of all completed runs"
  echo "  list                    List all presets and their descriptions"
  echo ""
  echo "Presets: ${ALL_PRESETS[*]}"
  echo ""
  echo "Variant descriptions:"
  for p in "${ALL_PRESETS[@]}"; do
    printf "  %-6s %s\n" "$p" "${PRESET_DESC[$p]}"
  done
}

if [[ -z "${CMD}" || "${CMD}" == "--help" || "${CMD}" == "-h" ]]; then
  usage
  exit 0
fi

# ---- list ----
if [[ "${CMD}" == "list" ]]; then
  printf "%-10s %-30s %s\n" "PRESET" "RUN_NAME" "DESCRIPTION"
  printf "%-10s %-30s %s\n" "------" "--------" "-----------"
  for p in "${ALL_PRESETS[@]}"; do
    printf "%-10s %-30s %s\n" "$p" "${PRESET_RUN_NAME[$p]}" "${PRESET_DESC[$p]}"
  done
  exit 0
fi

# ---- compare ----
if [[ "${CMD}" == "compare" ]]; then
  echo "== Comparing TCDyFIS-v2 results =="
  # Collect all v2 run names + the v1 baseline
  RUN_DIRS=()
  for p in "${ALL_PRESETS[@]}"; do
    d="${OUTS_DIR}/${PRESET_RUN_NAME[$p]}"
    [[ -d "$d" ]] && RUN_DIRS+=("$d")
  done
  # Also include v1 baselines for comparison
  for d in "${OUTS_DIR}/tcdyfis_task2_opt" "${OUTS_DIR}/tcdyfis_task2" "${OUTS_DIR}/tcdyfis_task1" "${OUTS_DIR}/tcdyfis_task1_grouped"; do
    [[ -d "$d" ]] && RUN_DIRS+=("$d")
  done

  if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "No completed runs found."
    exit 0
  fi

  python "${PROJECT_ROOT}/scripts/compare_results.py" "${RUN_DIRS[@]}"
  exit 0
fi

# ---- train_all ----
if [[ "${CMD}" == "train_all" ]]; then
  DO_COMPARE=false
  [[ "${ARG2:-}" == "--compare" ]] && DO_COMPARE=true

  PIDS=()
  idx=0
  mkdir -p "${OUTS_DIR}"

  echo "== Training all task2 variants: ${ALL_TASK2_VARIANTS[*]} =="
  echo "   Devices: ${FIS_DEVICES}, Max parallel: ${MAX_PARALLEL}"
  echo ""

  for variant in "${ALL_TASK2_VARIANTS[@]}"; do
    # Wait if at max parallel
    while [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]]; do
      NEW_PIDS=()
      for p in "${PIDS[@]}"; do
        if kill -0 "$p" 2>/dev/null; then
          NEW_PIDS+=("$p")
        fi
      done
      PIDS=("${NEW_PIDS[@]}")
      if [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]]; then
        sleep 5
      fi
    done

    resolve_config_and_run_name "$variant" || continue
    run_name="${PRESET_RUN_NAME[$variant]}"
    log="${OUTS_DIR}/${run_name}_console.log"
    dev="${DEVICE_LIST[$((idx % ${#DEVICE_LIST[@]}))]}"
    idx=$((idx + 1))

    echo "[train_all] ${variant} (${PRESET_DESC[$variant]}) -> ${run_name} device=cuda:${dev}"
    nohup python -m experiment.run train \
      --config "${CONFIG}" \
      --device "cuda:${dev}" \
      > "${log}" 2>&1 &
    PIDS+=($!)
    sleep 1
  done

  echo ""
  echo "Waiting for all jobs to finish..."
  for p in "${PIDS[@]}"; do
    wait "$p" || true
  done
  echo "All training jobs completed."

  if [[ "${DO_COMPARE}" == true ]]; then
    echo ""
    bash "$0" compare
  fi
  exit 0
fi

# ---- train single ----
if [[ "${CMD}" == "train" ]]; then
  PRESET="${ARG2:-A}"
  EXTRA="${ARG3:-}"

  if ! resolve_config_and_run_name "$PRESET"; then
    echo "ERROR: unknown preset '${PRESET}'"
    echo "Available: ${ALL_PRESETS[*]}"
    exit 1
  fi

  echo "== TCDyFIS-v2 =="
  echo "PRESET : ${PRESET} (${PRESET_DESC[$PRESET]:-})"
  echo "CONFIG : ${CONFIG}"
  echo "RUN    : ${RUN_NAME}"
  echo "DEVICE : cuda:${FIS_DEVICE}"
  echo ""

  if [[ "${EXTRA}" == "--bg" ]]; then
    log="${OUTS_DIR}/${RUN_NAME}_console.log"
    mkdir -p "${OUTS_DIR}"
    nohup python -m experiment.run train \
      --config "${CONFIG}" \
      --device "cuda:${FIS_DEVICE}" \
      > "${log}" 2>&1 &
    echo "PID: $!"
    echo "log: ${log}"
  else
    python -m experiment.run train \
      --config "${CONFIG}" \
      --device "cuda:${FIS_DEVICE}"
  fi
  exit 0
fi

echo "ERROR: unknown command '${CMD}'"
usage
exit 1
