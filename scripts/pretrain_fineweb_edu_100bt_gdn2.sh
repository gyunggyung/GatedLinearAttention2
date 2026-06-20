#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_cache}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT_DIR}/runs/triton/gdn2_100bt}"

TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/data/fineweb-edu/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/runs}"
MODEL="${MODEL:-gdn2_1.3B}"
TRAIN_CONFIG="${TRAIN_CONFIG:-tsz128x4k_100B}"
EXP_NAME="${EXP_NAME:-gdn2_1.3B_fineweb_edu_100bt}"
EXP_GROUP="${EXP_GROUP:-fineweb_edu_100bt}"
LR="${LR:-4e-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-0}"
GLOBAL_BATCH_TOKENS="${GLOBAL_BATCH_TOKENS:-524288}"
ACTUAL_TRAIN_TIME_MIN="${ACTUAL_TRAIN_TIME_MIN:-0}"
SAVE_STEP_INTERVAL="${SAVE_STEP_INTERVAL:-1000}"
EVAL_STEP_INTERVAL="${EVAL_STEP_INTERVAL:-1000}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-auto}"
EXPECTED_TOKENS_PER_SEC="${EXPECTED_TOKENS_PER_SEC:-250000}"
DATA_SHUFFLE_SEED="${DATA_SHUFFLE_SEED:-3407}"
DATA_SHUFFLE_BUFFER="${DATA_SHUFFLE_BUFFER:-0}"
HF_UPLOAD="${HF_UPLOAD:-0}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_UPLOAD_INTERVAL_TOKENS="${HF_UPLOAD_INTERVAL_TOKENS:-1000000000}"
HF_PRIVATE="${HF_PRIVATE:-true}"
HF_UPLOAD_BLOCKING="${HF_UPLOAD_BLOCKING:-false}"
DEVICES="${DEVICES:-}"

if [[ -n "${DEVICES}" ]]; then
  export DEVICES
fi

VAL_ARGS=()
if [[ -n "${VALIDATION_DATA:-}" ]]; then
  VAL_ARGS+=(--val_data_dir_raw "${VALIDATION_DATA}" --val_type val_sampled)
fi

HF_BOOL_ARGS=()
case "${HF_UPLOAD,,}" in
  1|true|yes|on) HF_BOOL_ARGS+=(--hf_upload) ;;
  *) HF_BOOL_ARGS+=(--no-hf_upload) ;;
esac
case "${HF_PRIVATE,,}" in
  1|true|yes|on) HF_BOOL_ARGS+=(--hf_private) ;;
  *) HF_BOOL_ARGS+=(--no-hf_private) ;;
esac
case "${HF_UPLOAD_BLOCKING,,}" in
  1|true|yes|on) HF_BOOL_ARGS+=(--hf_upload_blocking) ;;
  *) HF_BOOL_ARGS+=(--no-hf_upload_blocking) ;;
esac

python -u "${ROOT_DIR}/pretrain.py" \
  --output_root "${OUTPUT_ROOT}" \
  --train_data_dir_raw "${TRAIN_DATA}" \
  --train_data_dir "${TRAIN_DATA}" \
  --corpus_name fineweb-edu \
  --model_name "${MODEL}" \
  --exp_name "${EXP_NAME}" \
  --exp_group "${EXP_GROUP}" \
  --train_config "${TRAIN_CONFIG}" \
  --tokenizer_name TinyLlama/TinyLlama_v1.1 \
  --learning_rate "${LR}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --global_batch_tokens "${GLOBAL_BATCH_TOKENS}" \
  --actual_train_time "${ACTUAL_TRAIN_TIME_MIN}" \
  --save_step_interval "${SAVE_STEP_INTERVAL}" \
  --eval_step_interval "${EVAL_STEP_INTERVAL}" \
  --train_num_workers "${TRAIN_NUM_WORKERS}" \
  --activation_checkpointing "${ACTIVATION_CHECKPOINTING}" \
  --expected_tokens_per_sec "${EXPECTED_TOKENS_PER_SEC}" \
  --data_shuffle_seed "${DATA_SHUFFLE_SEED}" \
  --data_shuffle_buffer "${DATA_SHUFFLE_BUFFER}" \
  --hf_repo_id "${HF_REPO_ID}" \
  --hf_upload_interval_tokens "${HF_UPLOAD_INTERVAL_TOKENS}" \
  --use_stream_tok \
  "${HF_BOOL_ARGS[@]}" \
  "${VAL_ARGS[@]}"
