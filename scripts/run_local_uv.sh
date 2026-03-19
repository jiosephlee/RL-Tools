#!/bin/bash
# Local NeMo RL training script using uv-managed environments (no SLURM).
# Ray is initialized internally by NeMo RL (init_ray in virtual_cluster.py).
#
# Usage:
#   bash scripts/run_local_uv.sh --config examples/configs/your_config.yaml [hydra overrides...]
#
# Example:
#   NUM_GPUS=2 bash scripts/run_local_uv.sh --config examples/configs/grpo.yaml \
#       policy.model_name=meta-llama/Llama-3.1-8B cluster.gpus_per_node=2

set -euo pipefail

# ---------------------------------------------------------------------------
# CUDA setup — load CUDA 12.8 to match torch cu128 wheels
# ---------------------------------------------------------------------------
module load cuda/12.8.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}"
echo "CUDA_HOME=$CUDA_HOME  (nvcc: $(nvcc --version 2>&1 | grep release | awk '{print $5}' | tr -d ','))"

# cuDNN + NCCL headers needed by transformer-engine (megatron dependency).
# Adjust CUDNN_HOME to your cluster's cuDNN install path.
CUDNN_HOME="${CUDNN_HOME:-/usr/lib/x86_64-linux-gnu}"
# NCCL headers from pip-installed nvidia-nccl-cu12 in the main venv
NCCL_INCLUDE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/lib/python3.12/site-packages/nvidia/nccl/include"
if [ -d "$NCCL_INCLUDE" ]; then
    export CPLUS_INCLUDE_PATH="${CUDNN_HOME}/include:${NCCL_INCLUDE}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
else
    export CPLUS_INCLUDE_PATH="${CUDNN_HOME}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi
export LIBRARY_PATH="${CUDNN_HOME}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDNN_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---------------------------------------------------------------------------
# Cache isolation — avoid stale CUDA builds and cross-CUDA contamination
# ---------------------------------------------------------------------------
export UV_NO_CACHE=1

CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer-cu${CUDA_VER}"
export TRITON_CACHE_DIR="${HOME}/.cache/triton-cu${CUDA_VER}"
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch_inductor-cu${CUDA_VER}"

# ---------------------------------------------------------------------------
# Ray / misc environment
# ---------------------------------------------------------------------------
unset TMPDIR
unset RAY_TMPDIR
unset RAY_BACKEND_LOG_LEVEL

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

# CCCL headers (cuda/std/*) live under include/cccl/ in spack CUDA installs;
# deep-gemm/cutlass expects them at include/cuda/std/, so add the cccl path.
if [ -d "${CUDA_HOME}/include/cccl" ]; then
    export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Cleanup stale Ray state from previous runs (e.g. after Ctrl+C)
# ---------------------------------------------------------------------------
uv run python -m ray.scripts.scripts stop --force 2>/dev/null || true
sleep 1
rm -rf /tmp/ray/session_* /tmp/ray/session_latest 2>/dev/null || true
rm -f /dev/shm/plasma_* /dev/shm/ray_* 2>/dev/null || true

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================"
echo "NeMo RL — Local Training (uv-managed)"
echo "========================================"
echo "Args: $*"
echo "========================================"

uv run examples/run_grpo.py "$@"
