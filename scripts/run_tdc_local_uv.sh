#!/bin/bash
# Local TDC GRPO training script using uv-managed environments (no SLURM).
# Ray is initialized internally by NeMo RL (init_ray in virtual_cluster.py).
#
# Usage:
#   bash scripts/run_tdc_local_uv.sh                          # defaults: 1 GPU, 10 steps
#   NUM_GPUS=4 MAX_STEPS=100 bash scripts/run_tdc_local_uv.sh # override
#   MULTI_TURN=0 bash scripts/run_tdc_local_uv.sh             # single-turn TDC

### ARGS ###
MODEL="${MODEL:-unsloth/gpt-oss-20b-BF16}"
NUM_GPUS="${NUM_GPUS:-1}"
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
# Per-task JSONL files are listed in the YAML config.
# Generate from merged JSONL: python scripts/split_tdc_by_task.py
BALANCED_SAMPLING="${BALANCED_SAMPLING:-0}"

### ENV VARS ###
# Unset VIRTUAL_ENV from conda activation so uv worker venv creation
# doesn't conflict (the warning "does not match the project environment path")
unset VIRTUAL_ENV

# Put venvs and cache on project disk to avoid home quota issues
export UV_PROJECT_ENVIRONMENT=/vast/projects/myatskar/design-documents/nemo-rl-venv
export UV_CACHE_DIR=/vast/projects/myatskar/design-documents/.uv-cache
export NEMO_RL_VENV_DIR=/vast/projects/myatskar/design-documents/nemo-rl-venvs
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1  # Skip worker venv creation; use current Python directly (same as NeMo RL Dockerfiles)

### RAY SETUP ###
export RAY_TMPDIR="/tmp/ray_rl"
mkdir -p "$RAY_TMPDIR"

NUM_GPUS_DETECTED="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

# Clear stale state
unset RAY_ADDRESS
uv run python -m ray.scripts.scripts stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR"/session_* 2>/dev/null || true
sleep 2

# Start Ray head externally to avoid EOF socket bug in init_ray()
RAY_PORT=$(( 6379 + (RANDOM % 1000) ))
RAY_NODE_IP=$(hostname -I | awk '{print $1}')
echo "Starting Ray head at $RAY_NODE_IP:$RAY_PORT with $NUM_GPUS_DETECTED GPUs"
uv run python -m ray.scripts.scripts start --head \
    --node-ip-address "$RAY_NODE_IP" \
    --port "$RAY_PORT" \
    --num-gpus "$NUM_GPUS_DETECTED" \
    --temp-dir "$RAY_TMPDIR"

export RAY_ADDRESS="$RAY_NODE_IP:$RAY_PORT"

# Wait for Ray
for i in {1..30}; do
    uv run python -m ray.scripts.scripts status >/dev/null 2>&1 && break
    sleep 1
done
uv run python -m ray.scripts.scripts status || { echo "Error: Ray failed to start" >&2; exit 1; }

export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_NVLS_ENABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}}"
mkdir -p "$TRITON_CACHE_DIR"

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
)

# Balanced per-task sampling (each task contributes equally per batch)
if [ "$BALANCED_SAMPLING" = "1" ]; then
    OVERRIDES+=(
        "data.use_multiple_dataloader=true"
        "grpo.num_prompts_per_step=16"
    )
    echo "Balanced per-task sampling enabled"
fi

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
echo "NeMo RL — TDC GRPO Training (local, uv-managed)"
echo "========================================"
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo "Balanced sampling: $BALANCED_SAMPLING"
echo "Max steps: $MAX_STEPS"
echo "Multi-turn: $MULTI_TURN  Liger: $LIGER_GRPO_LOSS  TIS: $TIS"
echo "----------------------------------------"
echo "Overrides:"
printf "  %s\n" "${OVERRIDES[@]}"
echo "========================================"

### RUN ###
echo "running"
uv run python examples/run_grpo.py \
    --config examples/configs/grpo_tdc_tool_calling.yaml \
    "${OVERRIDES[@]}"

### CLEANUP ###
echo "Training complete. Stopping Ray..."
uv run python -m ray.scripts.scripts stop --force 2>/dev/null || true
