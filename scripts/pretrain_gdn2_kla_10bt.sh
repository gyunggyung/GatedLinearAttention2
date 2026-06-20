#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL="${MODEL:-gdn2_kla_1.3B}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-tsz128x4k_10B}"
export EXP_NAME="${EXP_NAME:-gdn2_kla_1.3B_fineweb_edu_10bt}"
export EXP_GROUP="${EXP_GROUP:-fineweb_edu_10bt}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT_DIR}/runs/triton/gdn2_kla_10bt}"
export EXPECTED_TOKENS_PER_SEC="${EXPECTED_TOKENS_PER_SEC:-250000}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
export DATA_SHUFFLE_SEED="${DATA_SHUFFLE_SEED:-3407}"
export DATA_SHUFFLE_BUFFER="${DATA_SHUFFLE_BUFFER:-100000}"
export HF_UPLOAD="${HF_UPLOAD:-1}"
export HF_REPO_ID="${HF_REPO_ID:-Gated_Linear_Attention2}"
export HF_UPLOAD_INTERVAL_TOKENS="${HF_UPLOAD_INTERVAL_TOKENS:-1000000000}"
export HF_PRIVATE="${HF_PRIVATE:-true}"
export HF_UPLOAD_BLOCKING="${HF_UPLOAD_BLOCKING:-false}"

exec "${SCRIPT_DIR}/pretrain_fineweb_edu_100bt_gdn2.sh"
