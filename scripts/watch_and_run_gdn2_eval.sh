#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="${PIDFILE:-${ROOT_DIR}/runs/logs/gdn2_kla_10bt_4k_hf.pid}"
TRAIN_LOG="${TRAIN_LOG:-${ROOT_DIR}/runs/logs/gdn2_kla_10bt_4k_hf.log}"
EVAL_LOG="${EVAL_LOG:-${ROOT_DIR}/runs/eval/gdn2_paper/watch_eval.log}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/runs/eval/gdn2_paper}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

mkdir -p "$(dirname "${EVAL_LOG}")"

log() {
  echo "[$(TZ=Asia/Seoul date '+KST %Y-%m-%d %H:%M:%S')] $*"
}

if [[ ! -f "${PIDFILE}" ]]; then
  log "PID file not found: ${PIDFILE}"
  exit 1
fi

PID="$(cat "${PIDFILE}")"
log "Waiting for training PID ${PID} to finish before starting GDN-2 paper eval."

while kill -0 "${PID}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

log "Training PID ${PID} exited. Checking train log."
if grep -q "TRAIN_EXIT .* code=0" "${TRAIN_LOG}" || grep -q "Training time:" "${TRAIN_LOG}"; then
  log "Training appears complete. Launching evaluation on GPUs ${GPUS}."
else
  log "Training exit code marker not found. Launching evaluation only if final/checkpoint-10B exists."
fi

python "${ROOT_DIR}/scripts/run_gdn2_paper_eval.py" \
  --results_dir "${RESULTS_DIR}" \
  --gpus "${GPUS}" \
  --require_10b \
  --primary_checkpoint 10B \
  --summarize_after_phase \
  --summary_output "${RESULTS_DIR}/GDN2_PAPER_EVAL_RESULTS.md" \
  2>&1 | tee -a "${EVAL_LOG}"

python "${ROOT_DIR}/scripts/summarize_gdn2_paper_eval.py" \
  --results_dir "${RESULTS_DIR}" \
  --output "${RESULTS_DIR}/GDN2_PAPER_EVAL_RESULTS.md" \
  2>&1 | tee -a "${EVAL_LOG}"

log "Evaluation finished. Summary: ${RESULTS_DIR}/GDN2_PAPER_EVAL_RESULTS.md"
