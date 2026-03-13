# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeMo RL is NVIDIA's scalable post-training library for reinforcement learning on multimodal models (1 GPU to thousands, tiny to >100B parameters). Built on Ray for distributed orchestration, PyTorch for training, and vLLM/SGLang for inference.

## Common Commands

```bash
# All commands use uv — never activate venv manually
uv run python examples/run_grpo.py           # Run GRPO training
uv run pytest tests/unit/                     # Run all unit tests
uv run pytest tests/unit/test_foo.py -v       # Run a single test file
uv run pytest tests/unit/test_foo.py::test_bar -v  # Run a single test
uv run pytest tests/unit/ --cov=nemo_rl       # Tests with coverage

# Linting and formatting (also runs automatically via pre-commit)
uv run ruff check nemo_rl/                    # Lint
uv run ruff format nemo_rl/                   # Format
uv run pyrefly check nemo_rl/                 # Type check

# Install pre-commit hooks
uv run --group dev pre-commit install
```

## Architecture

### Training Loop (GRPO as reference)

Entry points are in `examples/` (e.g., `run_grpo.py`). The flow:

1. **Config**: YAML → Hydra → `MasterConfig` TypedDict
2. **`setup(config)`** initializes all components: policy, generation, clusters, dataloader, loss function
3. **`grpo_train()`** runs the main loop:
   - Prepare batch → generate responses (rollout) → compute rewards → compute advantages → train (fwd/bwd/optimizer step) → checkpoint

Other algorithms (DPO, SFT, RM, distillation) follow the same pattern in `nemo_rl/algorithms/`.

### Distributed Execution via Ray

The system uses three key abstractions in `nemo_rl/distributed/`:

- **`RayVirtualCluster`** — creates Ray placement groups as logical compute nodes with GPU bundles
- **`RayWorkerGroup`** — manages a pool of Ray actors; distributes work via `run_all_workers_sharded_data()` (data-parallel sharding) or `run_all_workers_single_data()` (broadcast)
- **`NamedSharding`** — maps abstract parallelism axes (DP, TP, PP, CP) to global ranks

The main process orchestrates but never holds model weights — all compute is pushed to Ray actors.

### Policy (Training) — `nemo_rl/models/policy/`

`Policy` wraps a `RayWorkerGroup` with two backend options:
- **DTensor** (`DTensorPolicyWorker`) — PyTorch-native FSDP2/TP/SP/CP
- **Megatron** (`MegatronPolicyWorker`) — Megatron Core with 6D parallelism; includes built-in generation

Key methods: `get_logprobs()`, `train()`, `prepare_refit_info()` (sync weights to generation workers).

### Generation (Inference) — `nemo_rl/models/generation/`

Separate from policy when using DTensor. Backends:
- **vLLM** (`VllmGeneration`) — async engine, tensor parallelism, speculative decoding
- **SGLang** (`SGLangGeneration`) — compiled frontend
- **Megatron** — inline with policy (no separate generation workers)

**Colocated vs. non-colocated**: Colocated shares GPU memory between train/inference phases with weight refitting. Non-colocated uses separate clusters with collective communication for weight sync.

### Experience Collection — `nemo_rl/experience/rollouts.py`

Three paths: synchronous multi-turn rollout, async rollout (pipelined generation), and NeMo-Gym rollout. Data flows through: `LLMMessageLog` → `FlatMessages` → `GenerationDatumSpec` → `GenerationOutputSpec`.

### Data — `nemo_rl/data/`

- `DatumSpec` (TypedDict): message_log, length, loss_multiplier, task_name
- `BatchedDataDict`: dict-of-tensors wrapper with `select_indices()`, `shard_by_batch_size()`, `repeat_interleave()`
- Stateful dataloaders enable checkpoint resumption mid-epoch

### Loss Functions — `nemo_rl/algorithms/loss/`

`LossFunction` protocol takes data + logprobs, returns `(loss, metrics)`. `ClippedPGLossFn` implements clipped policy gradient with KL penalty against reference policy.

## Code Conventions

- **Style**: Google Python Style Guide; enforced by ruff (lint/format) + pyrefly (types)
- **Line length**: 120 chars
- **Config defaults**: YAML is the single source of truth. Use `config["key"]` — never `.get("key", default)` for config values. Optional config uses `TypedDict` with `NotRequired`.
- **Copyright header**: Required on all Python files outside `tests/` — NVIDIA Apache 2.0 with current year (see CODING_GUIDELINES.md for template)
- **Ray remote**: Add `# pragma: no cover` on `@ray.remote` decorated classes/functions
- **Docstrings**: Google-style; reserved for public interfaces. Comments for internal logic only.
- **Global variables**: Prefix with `G_` (e.g., `G_MY_GLOBAL`)

## CI

CI does not run automatically on PRs. Apply a label and comment `/ok to test <commit-sha>`:
- `CI:docs` — doctests only
- `CI:L0` — doctests + unit tests
- `CI:L1` — L0 + functional tests
- `CI:Lfast` — fast tests, skips container build

## Commits

All commits require DCO sign-off: `git commit -s -m "message"`
