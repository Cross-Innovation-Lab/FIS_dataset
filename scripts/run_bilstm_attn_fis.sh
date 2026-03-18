#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  BiLSTM + Attention FIS 基线 训练 / 评估 启动脚本
#
#  用法:
#    训练任务一:  bash scripts/run_bilstm_attn_fis.sh train task1
#    训练任务二:  bash scripts/run_bilstm_attn_fis.sh train task2
#    后台训练:    bash scripts/run_bilstm_attn_fis.sh train task1 --bg
#    评估:        bash scripts/run_bilstm_attn_fis.sh eval  task1  <ckpt_path> [split]
#
#  训练日志写入:   outs/<run_name>/train.log    （由 Python 端写入）
#  终端输出日志:   outs/<run_name>/console.log  （由 nohup 重定向）
#  最优 checkpoint: checkpoints/<run_name>/best.pt
# ============================================================================

PROJECT_ROOT="/CIL_PROJECTS/CODES/MM_FIS"
CONFIG_DIR="${PROJECT_ROOT}/experiment/configs"

CMD="${1:-train}"
TASK="${2:-task1}"
EXTRA="${3:-}"

case "${TASK}" in
  task1) CONFIG="${CONFIG_DIR}/bilstm_attn_task1.json" ;;
  task2) CONFIG="${CONFIG_DIR}/bilstm_attn_task2.json" ;;
  *)
    if [[ -f "${TASK}" ]]; then
      CONFIG="${TASK}"
    else
      echo "ERROR: 未知 task '${TASK}'。可选: task1, task2, 或直接传入配置文件路径。"
      exit 1
    fi
    ;;
esac

cd "${PROJECT_ROOT}"

echo "== BiLSTM-Attn FIS 实验 =="
echo "CMD    : ${CMD}"
echo "CONFIG : ${CONFIG}"
echo ""

if [[ "${CMD}" == "train" ]]; then
  if [[ "${EXTRA}" == "--bg" ]]; then
    # 后台运行：终端输出重定向到 console.log
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CONSOLE_LOG="${PROJECT_ROOT}/outs/bilstm_attn_fis_${TIMESTAMP}_console.log"
    mkdir -p "$(dirname "${CONSOLE_LOG}")"
    echo "后台训练启动，终端日志: ${CONSOLE_LOG}"
    nohup python -m experiment.run train --config "${CONFIG}" > "${CONSOLE_LOG}" 2>&1 &
    echo "PID: $!"
    echo "查看日志: tail -f ${CONSOLE_LOG}"
  else
    python -m experiment.run train --config "${CONFIG}"
  fi

elif [[ "${CMD}" == "eval" ]]; then
  CKPT="${EXTRA}"
  SPLIT="${4:-test}"
  if [[ -z "${CKPT}" ]]; then
    echo "ERROR: eval 模式需指定 checkpoint 路径。"
    echo "用法: bash scripts/run_bilstm_attn_fis.sh eval task1 checkpoints/xxx/best.pt [split]"
    exit 1
  fi
  python -m experiment.run eval --ckpt "${CKPT}" --config "${CONFIG}" --split "${SPLIT}"

else
  echo "ERROR: 未知命令 '${CMD}'。可选: train, eval"
  exit 1
fi

