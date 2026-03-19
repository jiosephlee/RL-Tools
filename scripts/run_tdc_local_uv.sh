#!/bin/bash
# Local TDC GRPO training script using uv-managed environments (no SLURM).
# Ray is initialized internally by NeMo RL (init_ray in virtual_cluster.py).
#
# Usage:
#   bash scripts/run_tdc_local_uv.sh                          # defaults: 1 GPU, 10 steps
#   NUM_GPUS=4 MAX_STEPS=100 bash scripts/run_tdc_local_uv.sh # override
#   MULTI_TURN=0 bash scripts/run_tdc_local_uv.sh             # single-turn TDC

set -euo pipefail

# Load CUDA 12.8 to match torch cu128; explicitly export CUDA_HOME
# so it propagates through Ray to _env_builder worker subprocesses
module load cuda/12.8.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}"
echo "CUDA_HOME=$CUDA_HOME  (nvcc: $(nvcc --version 2>&1 | grep release | awk '{print $5}' | tr -d ','))"

# cuDNN + NCCL headers needed by transformer-engine (megatron dependency)
CUDNN_HOME="/vast/parcc/spack/sw/apps/linux-sapphirerapids/cudnn-8.9.7.29-12-xfuy2r7bhizjohnv6gazm6ynxpjhmfxi"
# NCCL headers come from pip-installed nvidia-nccl-cu12 in the main venv
NCCL_INCLUDE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/lib/python3.12/site-packages/nvidia/nccl/include"
export CPLUS_INCLUDE_PATH="${CUDNN_HOME}/include:${NCCL_INCLUDE}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="${CUDNN_HOME}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Skip uv cache to avoid stale CUDA builds (e.g. deep_gemm built against wrong CUDA)
export UV_NO_CACHE=1

# Isolate JIT caches by CUDA version to avoid cross-contamination
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer-cu${CUDA_VER}"
export TRITON_CACHE_DIR="${HOME}/.cache/triton-cu${CUDA_VER}"
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch_inductor-cu${CUDA_VER}"

unset TMPDIR
unset RAY_TMPDIR
# Also try increasing the socket timeout
unset RAY_BACKEND_LOG_LEVEL  # unset so Raylet logs at INFO — needed to debug agent startup
# export NEMO_RL_PY_EXECUTABLES_SYSTEM=1 
### ARGS ###
MODEL="${MODEL:-unsloth/gpt-oss-20b-BF16}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_STEPS="${MAX_STEPS:-100000000}"
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
# unset VIRTUAL_ENV

# Put venvs and cache on project disk to avoid home quota issues
# export UV_PROJECT_ENVIRONMENT=/vast/projects/myatskar/design-documents/nemo-rl-venv
# export UV_CACHE_DIR=/vast/projects/myatskar/design-documents/.uv-cache
# export NEMO_RL_VENV_DIR=/vast/projects/myatskar/design-documents/nemo-rl-venvs
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

# CCCL headers (cuda/std/*) live under include/cccl/ in spack CUDA installs;
# deep-gemm/cutlass expects them at include/cuda/std/, so add the cccl path.
if [ -d "${CUDA_HOME}/include/cccl" ]; then
    export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi


NUM_GPUS_DETECTED="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

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

### CLEANUP STALE RAY STATE ###
# Stop any surviving Ray processes from a previous Ctrl+C
uv run python -m ray.scripts.scripts stop --force 2>/dev/null || true
sleep 1
rm -rf /tmp/ray/session_* /tmp/ray/session_latest 2>/dev/null || true
# Clean up stale Ray/Plasma shared memory objects in /dev/shm (accumulate after Ctrl+C)
rm -f /dev/shm/plasma_* /dev/shm/ray_* 2>/dev/null || true

### RUN ###
echo "running"
uv run examples/run_grpo.py \
    --config examples/configs/grpo_tdc_tool_calling.yaml \
    "${OVERRIDES[@]}"
