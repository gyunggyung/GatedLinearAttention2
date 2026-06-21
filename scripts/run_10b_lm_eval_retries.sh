#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/outputs/tsz128x4k_10B_gdn2_kla_1.3B_fineweb_edu_10bt}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/runs/eval/gdn2_paper}"
CKPT="${CKPT:-${OUT_DIR}/hf_checkpoints/checkpoint-10B/model-ckpt.pth}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_ROOT}/10B}"
SPLIT_DIR="${SPLIT_DIR:-${CHECKPOINT_DIR}/splits}"
LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/logs}"
TOKENIZER_NAME="${TOKENIZER_NAME:-TinyLlama/TinyLlama_v1.1}"
MODEL_NAME="${MODEL_NAME:-gdn2_kla_1.3B}"
STANDARD_BATCH_SIZE="${STANDARD_BATCH_SIZE:-8}"
REAL_BATCH_SIZE="${REAL_BATCH_SIZE:-16}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-0}"

mkdir -p "${SPLIT_DIR}" "${LOG_DIR}"

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log="${LOG_DIR}/10B_retry_${name}_gpu${gpu}.log"
  echo "[$(TZ=Asia/Seoul date '+KST %Y-%m-%d %H:%M:%S')] launch gpu=${gpu} ${name}" >&2
  (
    set +e
    echo "[$(TZ=Asia/Seoul date '+KST %Y-%m-%d %H:%M:%S')] start gpu=${gpu} ${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      NUMEXPR_MAX_THREADS=256 \
      TOKENIZERS_PARALLELISM=false \
      "$@"
    rc=$?
    echo "[$(TZ=Asia/Seoul date '+KST %Y-%m-%d %H:%M:%S')] exit gpu=${gpu} ${name} rc=${rc}"
    exit "${rc}"
  ) > "${log}" 2>&1 &
  pids+=("$!")
}

rm -f \
  "${SPLIT_DIR}/standard_without_social_iqa.json" \
  "${SPLIT_DIR}/real_swde_squad.json" \
  "${SPLIT_DIR}/real_fda_triviaqa.json" \
  "${SPLIT_DIR}/real_nq_drop.json"

pids=()
launch 4 standard_without_social_iqa \
  python "${ROOT_DIR}/scripts/lm_eval_gla2.py" \
  --checkpoint "${CKPT}" --model_name "${MODEL_NAME}" --tokenizer_name "${TOKENIZER_NAME}" --dtype bf16 \
  --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq \
  --max_length 4096 --batch_size "${STANDARD_BATCH_SIZE}" \
  --output "${SPLIT_DIR}/standard_without_social_iqa.json" --bootstrap_iters "${BOOTSTRAP_ITERS}"

launch 5 real_swde_squad \
  python "${ROOT_DIR}/scripts/lm_eval_gla2.py" \
  --checkpoint "${CKPT}" --model_name "${MODEL_NAME}" --tokenizer_name "${TOKENIZER_NAME}" --dtype bf16 \
  --tasks swde,squad_completion --max_length 2048 --batch_size "${REAL_BATCH_SIZE}" \
  --output "${SPLIT_DIR}/real_swde_squad.json" --bootstrap_iters "${BOOTSTRAP_ITERS}"

launch 6 real_fda_triviaqa \
  python "${ROOT_DIR}/scripts/lm_eval_gla2.py" \
  --checkpoint "${CKPT}" --model_name "${MODEL_NAME}" --tokenizer_name "${TOKENIZER_NAME}" --dtype bf16 \
  --tasks fda,triviaqa --max_length 2048 --batch_size "${REAL_BATCH_SIZE}" \
  --output "${SPLIT_DIR}/real_fda_triviaqa.json" --bootstrap_iters "${BOOTSTRAP_ITERS}"

launch 7 real_nq_drop \
  python "${ROOT_DIR}/scripts/lm_eval_gla2.py" \
  --checkpoint "${CKPT}" --model_name "${MODEL_NAME}" --tokenizer_name "${TOKENIZER_NAME}" --dtype bf16 \
  --tasks nq_open,drop --max_length 2048 --batch_size "${REAL_BATCH_SIZE}" \
  --output "${SPLIT_DIR}/real_nq_drop.json" --bootstrap_iters "${BOOTSTRAP_ITERS}"

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

python "${ROOT_DIR}/scripts/merge_10b_eval_splits.py" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --split_dir "${SPLIT_DIR}"

python "${ROOT_DIR}/scripts/summarize_gdn2_paper_eval.py" \
  --results_dir "${RESULTS_ROOT}" \
  --output "${RESULTS_ROOT}/GDN2_PAPER_EVAL_RESULTS.md"

echo "[$(TZ=Asia/Seoul date '+KST %Y-%m-%d %H:%M:%S')] 10B lm-eval retries done status=${status}"
exit "${status}"
