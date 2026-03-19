# Onboarding: Running NeMo RL with TDC Tool Calling

This document captures the lessons learned getting NeMo RL running with TDC (Tool-use Diversity Curriculum) and multi-turn tool calling on our local cluster. It covers dependency setup, Ray/CUDA pitfalls, and the code changes needed to make everything work end-to-end.

## Timeline of Changes (March 13–19, 2026)

### Phase 1: Initial TDC Integration (Mar 13)

**Commits:** `287cbb7` → `8bd3967` → `760bd24` → `58653ad` → `dc6d4a4`

- Added concatenated 16-task TDC train/val JSONL datasets
- Built the TDC tool-calling environment (`nemo_rl/environments/tool_calling/`)
- Added Liger GRPO loss, per-dataset eval, and tracing support
- Created `tdc_processor.py` for TDC data preprocessing
- Added the initial launch script `scripts/run_tdc_local.sh`
- Fixed: `tdc_processor.py` was inside a `processors/` package that shadowed `processors.py` — moved it up one level

### Phase 2: Dependency Hell — CUDA, PyTorch, vLLM (Mar 13)

**Commits:** `4ee3940` → `6844484` → `a3d8c30` → `43ff8ac` → `fcae0f8` → `df8340c`

- Bumped to torch 2.10.0 (cu128), vLLM 0.17.1, transformers 5.3.0, flash-attn 2.8.3
- Multiple rounds of `pyproject.toml` / `uv.lock` fixes to get consistent CUDA builds
- **Key lesson:** `pyproject.toml` PyTorch index must match the CUDA version on the machine. We're using `pytorch-cu128` index pointing to `https://download.pytorch.org/whl/cu128` with `module load cuda/12.8.1`.

**Uncommitted change (current):** switching pyproject.toml from `pytorch-cu130` back to `pytorch-cu128` to match our CUDA 12.8 environment.

### Phase 3: Ray Startup Crashes and Worker Environment Issues (Mar 13–14)

**Commits:** `f8c2fe4` → `6c59659` → `7a79769` (many "fixing ray" commits), `ae1cc75`, `b37a464`, `20c0dd2`, `ea03975`, `e33dd1b`

This was the hardest phase. Ray workers would crash on startup with opaque errors. Root causes discovered:

#### Problem 1: `runtime_env` payload too large
**Fix (uncommitted, `virtual_cluster.py`):** Do NOT pass `dict(os.environ)` as `env_vars` in Ray's `runtime_env`. On SLURM nodes the full environment has 200+ variables. This creates a massive payload that the `runtime_env_agent` must deserialize/validate on startup, causing it to crash or timeout before the 30-second Raylet deadline. Instead, pass only the specific env vars that workers need, and let workers inherit the rest naturally.

#### Problem 2: `VIRTUAL_ENV` / conda conflicts
**Commit `7ccdcdd`:** When conda is activated, `VIRTUAL_ENV` gets set, which confuses uv's worker venv creation (produces "does not match the project environment path" warnings). Fix: `unset VIRTUAL_ENV` in the launch script or ensure conda is deactivated.

#### Problem 3: Missing CUDA libraries in worker venvs
**Fix (uncommitted, `worker_groups.py`):** Ray workers running in uv-created venvs couldn't find `libcudart` and other NVIDIA libraries. Added logic to automatically discover nvidia package lib dirs under the venv's `site-packages/nvidia/` and prepend them to `LD_LIBRARY_PATH` in the worker's env vars.

#### Problem 4: Stale Ray state after Ctrl+C
**Fix (in launch script):** After killing a run with Ctrl+C, leftover Ray processes and `/dev/shm` shared memory objects prevent the next run from starting cleanly. The launch script now runs cleanup before each launch:
```bash
uv run python -m ray.scripts.scripts stop --force 2>/dev/null || true
sleep 1
rm -rf /tmp/ray/session_* /tmp/ray/session_latest 2>/dev/null || true
rm -f /dev/shm/plasma_* /dev/shm/ray_* 2>/dev/null || true
```

#### Problem 5: `TMPDIR` interfering with Ray
**Fix:** `unset TMPDIR` and `unset RAY_TMPDIR` in the launch script. Some cluster configurations set `TMPDIR` to a non-standard location that Ray doesn't handle well.

### Phase 4: DTensor → Megatron Backend Switch (Mar 14)

**Commit `0a335e0`:**

Switched from DTensor to Megatron backend for policy training. Key config changes in `grpo_tdc_tool_calling.yaml`:
- Disabled `dtensor_cfg`, enabled `megatron_cfg`
- Turned on activation checkpointing
- Enabled optimizer CPU offload (`optimizer_cpu_offload: true`, `optimizer_offload_fraction: 1.0`)
- Enabled context-parallel token dispatch (`enabled: true` for token-level bucketing)
- Set `expert_model_parallel_size: 2` for MoE models
- Enabled fusions: `moe_permute_fusion`, `defer_fp32_logits`, `moe_shared_expert_overlap`

### Phase 5: Multi-turn Tool Calling Fixes (Mar 14–16, + uncommitted)

**Commits:** `7960842`, `21f6bc3`, `876343b`, `e780133`

Key code changes to make multi-turn rollouts work:

1. **`rollouts.py` — Preserve special tokens during decode:**
   Changed `tokenizer.batch_decode(skip_special_tokens=True)` → `skip_special_tokens=False`. Environments need to parse protocol-specific markers like `<|channel|>`, `<|call|>`, `<|im_end|>`.

2. **`rollouts.py` — Canonical token IDs for observations:**
   Added `observation_token_ids` to `EnvironmentReturn`. When the environment provides pre-tokenized observation token IDs, the rollout code uses them directly instead of re-tokenizing text. This avoids lossy text→tokenize round-trips for special tokens (e.g., `<|start|>`, `<|im_start|>`).

3. **`chat_protocol.py` — `render_tool_feedback_token_ids`:**
   Added method to produce exact token IDs via the harmony encoder for GPT-OSS models, bypassing HuggingFace tokenization that mangles special tokens.

4. **`environment.py` — Config-based `reward_fn` loading:**
   Added dynamic import of reward functions specified in YAML config (`reward_fn: "nemo_rl.environments.tdc_environment.compute_reward"`). Also passes tokenizer to environment for token-level feedback rendering.

5. **`tdc_processor.py` — Added required metadata fields:**
   Added `turn_count`, `tool_results`, and `format_reward` to the datum spec, required by `ToolCallingEnvironment` for multi-turn rollouts.

6. **`interfaces.py` — `EnvironmentReturn` updates:**
   Added `observation_token_ids` field and made `answers` default to `None`.

### Phase 6: Observability (uncommitted)

- **`grpo.py` — GPU Peak Memory Tracker:** Added `GPUPeakMemoryTracker` class that calls `snapshot_and_reset_peak_gpu_memory_mb` on policy workers at each stage boundary (prepare_for_generation, generation, logprob_computation, policy_training). Logs max-across-workers peak memory per stage.
- **`base_policy_worker.py`:** Added `snapshot_and_reset_peak_gpu_memory_mb()` method using `torch.cuda.max_memory_allocated()`.
- **`vllm_worker.py`:** Set `VLLM_LOGGING_LEVEL=WARNING` by default to suppress noisy INFO logs from cluttering the driver terminal (full logs still in Ray worker log files).

## How to Launch

### Prerequisites
- CUDA 12.8 available via `module load cuda/12.8.1`
- cuDNN 8.9.x installed (path set in script)
- `uv` installed

### Quick Start
```bash
# Basic 2-GPU run with defaults
bash scripts/run_tdc_local_uv.sh

# Customize
NUM_GPUS=4 MAX_STEPS=100 bash scripts/run_tdc_local_uv.sh

# Single-turn mode
MULTI_TURN=0 bash scripts/run_tdc_local_uv.sh
```

### Key Environment Variables in the Launch Script

| Variable | Purpose |
|---|---|
| `CUDA_HOME` | Must point to CUDA 12.8 installation |
| `CPLUS_INCLUDE_PATH` | cuDNN + NCCL headers for transformer-engine compilation |
| `UV_NO_CACHE=1` | Prevents stale CUDA builds from cache (e.g., deep_gemm built against wrong CUDA) |
| `FLASHINFER_CACHE_DIR` | Per-CUDA-version JIT cache to avoid cross-contamination |
| `TRITON_CACHE_DIR` | Per-CUDA-version JIT cache |
| `TORCH_CUDA_ARCH_LIST` | Set to `10.0` for our GPU architecture |
| `CPLUS_INCLUDE_PATH` (cccl) | CCCL headers for deep-gemm/cutlass (spack CUDA installs put them under `include/cccl/`) |

### Debugging Tips

1. **Ray worker logs:** Check `/tmp/ray/session_latest/logs/` for full worker output
2. **If Ray hangs on startup:** Kill stale processes with `ray stop --force`, clean `/dev/shm/plasma_*` and `/tmp/ray/session_*`
3. **CUDA version mismatch errors:** Verify `nvcc --version` matches the PyTorch wheel index in `pyproject.toml` (cu128 vs cu130)
4. **"does not match the project environment path":** Run `unset VIRTUAL_ENV` before launching
5. **vLLM `_C` import errors in workers:** Check that `LD_LIBRARY_PATH` includes the nvidia package lib dirs — the fix in `worker_groups.py` handles this automatically
6. **JIT compilation errors (flashinfer, triton):** Clear the per-CUDA caches: `rm -rf ~/.cache/flashinfer-* ~/.cache/triton-*`

## File Map

| File | What Changed |
|---|---|
| `scripts/run_tdc_local_uv.sh` | Main launch script with all env setup |
| `examples/configs/grpo_tdc_tool_calling.yaml` | GRPO config for TDC (Megatron backend, multi-turn) |
| `nemo_rl/algorithms/grpo.py` | GPU peak memory tracking |
| `nemo_rl/data/tdc_processor.py` | TDC data preprocessing + multi-turn metadata fields |
| `nemo_rl/distributed/virtual_cluster.py` | Fixed runtime_env payload (don't pass full os.environ) |
| `nemo_rl/distributed/worker_groups.py` | Auto-inject nvidia lib dirs into worker LD_LIBRARY_PATH |
| `nemo_rl/environments/interfaces.py` | Added `observation_token_ids` to EnvironmentReturn |
| `nemo_rl/environments/tool_calling/chat_protocol.py` | Token-level feedback rendering for GPT-OSS |
| `nemo_rl/environments/tool_calling/environment.py` | Dynamic reward_fn, tokenizer passthrough, token IDs |
| `nemo_rl/experience/rollouts.py` | Special token preservation + canonical token ID support |
| `nemo_rl/models/generation/vllm/vllm_worker.py` | Suppress vLLM INFO logs |
| `nemo_rl/models/policy/workers/base_policy_worker.py` | Peak GPU memory snapshot method |
| `pyproject.toml` | CUDA index (cu128 vs cu130) |
