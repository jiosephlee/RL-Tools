#!/bin/bash
# Local TDC GRPO training script (no SLURM).
#
# Usage:
#   bash scripts/run_tdc_local.sh                          # defaults: 2 GPUs, 10 steps
#   NUM_GPUS=4 MAX_STEPS=100 bash scripts/run_tdc_local.sh # override
#   MULTI_TURN=0 bash scripts/run_tdc_local.sh             # single-turn TDC

set -euo pipefail

# Use a fresh temp dir
export RAY_TMPDIR="/tmp/ray_fresh_${USER}_$(date +%s)"
mkdir -p "$RAY_TMPDIR"

# Also try increasing the socket timeout
export RAY_BACKEND_LOG_LEVEL=warning

### ARGS ###
MODEL="${MODEL:-unsloth/gpt-oss-20b-BF16}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_TURNS="${MAX_TURNS:-30}"
MULTI_TURN="${MULTI_TURN:-1}"
LIGER_GRPO_LOSS="${LIGER_GRPO_LOSS:-0}"
TIS="${TIS:-0}"
TIS_TYPE="${TIS_TYPE:-tis}"
TIS_THRESHOLDS="${TIS_THRESHOLDS:-0.5 5.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
NUM_GENS="${NUM_GENS:-8}"
TRAIN_GLOBAL_BS="${TRAIN_GLOBAL_BS:-64}"
VAL_PERIOD="${VAL_PERIOD:-16}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

### PROJECT ROOT ###
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

### DATA ###
TRAIN_DATA="${TRAIN_DATA:-data/tdc/tdc_16task_train.jsonl}"
VAL_DATA="${VAL_DATA:-data/tdc/tdc_16task_val.jsonl}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA" >&2
    exit 1
fi

### CLEANUP — ensure no stale Ray processes or sockets ###
ray stop --force 2>/dev/null || true
# Kill any orphaned Ray processes
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
pkill -9 -f plasma_store 2>/dev/null || true
# Remove stale Ray temp dirs to avoid socket EOF errors
rm -rf /tmp/ray/ /tmp/ray_${USER}* 2>/dev/null || true
sleep 3

### ENV VARS ###
export RAY_TMPDIR="/tmp/ray_${USER}_$$"
mkdir -p "$RAY_TMPDIR"
export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_NVLS_ENABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}}"
mkdir -p "$TRITON_CACHE_DIR"

# Clear stale caches
# rm -rf ~/.cache/torch/inductor/ "/tmp/torchinductor_${USER}/" ~/.cache/vllm/torch_compile_cache/ 2>/dev/null || true

### BUILD OVERRIDES ###
OVERRIDES=(
    "policy.model_name=$MODEL"
    "policy.optimizer.kwargs.lr=$LEARNING_RATE"
    "policy.max_total_sequence_length=$MAX_SEQ_LEN"
    "policy.train_global_batch_size=$TRAIN_GLOBAL_BS"
    "grpo.num_prompts_per_step=$NUM_PROMPTS"
    "grpo.num_generations_per_prompt=$NUM_GENS"
    "grpo.max_num_steps=$MAX_STEPS"
    "grpo.val_period=$VAL_PERIOD"
    "cluster.gpus_per_node=$NUM_GPUS"
    "policy.generation.vllm_cfg.gpu_memory_utilization=$VLLM_GPU_MEM_UTIL"
    "data.train.data_path=$TRAIN_DATA"
    "data.validation.data_path=$VAL_DATA"
)

# Tensor parallelism for multi-GPU
if [ "$NUM_GPUS" -gt 1 ]; then
    OVERRIDES+=(
        "policy.dtensor_cfg.tensor_parallel_size=1"
        "policy.generation.vllm_cfg.tensor_parallel_size=1"
    )
fi

# Multi-turn tool calling
if [ "$MULTI_TURN" = "1" ]; then
    OVERRIDES+=(
        "grpo.max_rollout_turns=$MAX_TURNS"
        "data.default.env_name=tool_calling"
        "env.tool_calling.max_turns=$MAX_TURNS"
    )
    echo "Multi-turn tool calling enabled (max_turns=$MAX_TURNS)"
fi

# Liger Triton GRPO loss
if [ "$LIGER_GRPO_LOSS" = "1" ]; then
    OVERRIDES+=(
        "loss_fn.use_liger_grpo_loss=true"
        "loss_fn.clip_eps_low=0.2"
        "loss_fn.clip_eps_high=0.2"
        "loss_fn.beta=0.01"
        "loss_fn.temperature=1.0"
        "loss_fn.loss_type=grpo"
    )
    echo "Liger Triton GRPO loss enabled"
fi

# Truncated Importance Sampling
if [ "$TIS" = "1" ]; then
    read -r TIS_LOW TIS_HIGH <<< "$TIS_THRESHOLDS"
    OVERRIDES+=(
        "loss_fn.use_importance_sampling_correction=true"
        "loss_fn.truncated_importance_sampling_ratio=$TIS_HIGH"
        "loss_fn.truncated_importance_sampling_ratio_min=$TIS_LOW"
        "loss_fn.truncated_importance_sampling_type=$TIS_TYPE"
    )
    echo "TIS enabled (type=$TIS_TYPE, thresholds=$TIS_THRESHOLDS)"
fi

# Extra overrides
if [ -n "$EXTRA_OVERRIDES" ]; then
    # shellcheck disable=SC2206
    OVERRIDES+=($EXTRA_OVERRIDES)
fi

### PRINT CONFIG ###
echo "========================================"
echo "NeMo RL — TDC GRPO Training (local)"
echo "========================================"
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo "Train data: $TRAIN_DATA"
echo "Val data: ${VAL_DATA}"
echo "Max steps: $MAX_STEPS"
echo "Multi-turn: $MULTI_TURN  Liger: $LIGER_GRPO_LOSS  TIS: $TIS"
echo "----------------------------------------"
echo "Overrides:"
printf "  %s\n" "${OVERRIDES[@]}"
echo "========================================"

### RUN ###
python examples/run_grpo.py \
    --config examples/configs/grpo_tdc_tool_calling.yaml \
    "${OVERRIDES[@]}"
