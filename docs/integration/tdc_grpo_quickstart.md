# TDC GRPO Training on NeMo RL — Quickstart

## Installation

```bash
# In your RL-Tools environment:
pip install liger-kernel>=0.7.0

# Or add to your conda env setup:
# micromamba install -c conda-forge liger-kernel  (if available)
# Otherwise: pip install liger-kernel>=0.7.0 inside the conda env
```

## Data Preparation

The SLURM script expects TDC data at:
```
$OPENRLHF_ROOT/data/tdc/openai_format_gpt_oss/{TASK}_train.jsonl
$OPENRLHF_ROOT/data/tdc/openai_format_gpt_oss/{TASK}_val.jsonl
```

Where `$OPENRLHF_ROOT` defaults to `../OpenRLHF-Tools` (relative to the RL-Tools repo root). If your data lives elsewhere, set `OPENRLHF_ROOT` or pass `TRAIN_DATA`/`VAL_DATA` directly.

## Quick Local Test (no SLURM)

```bash
# Single-turn TDC, 1 GPU, minimal config:
uv run python examples/run_grpo.py \
    --config examples/configs/grpo_tdc_tool_calling.yaml \
    data.train.data_path=/path/to/BBB_Martins_train.jsonl \
    data.validation.data_path=/path/to/BBB_Martins_val.jsonl \
    grpo.max_num_steps=10 \
    cluster.gpus_per_node=1

# With Liger Triton GRPO loss:
uv run python examples/run_grpo.py \
    --config examples/configs/grpo_tdc_tool_calling.yaml \
    data.train.data_path=/path/to/BBB_Martins_train.jsonl \
    data.validation.data_path=/path/to/BBB_Martins_val.jsonl \
    loss_fn.use_liger_grpo_loss=true \
    grpo.max_num_steps=10

# Multi-turn tool calling:
uv run python examples/run_grpo.py \
    --config examples/configs/grpo_tdc_tool_calling.yaml \
    data.train.data_path=/path/to/BBB_Martins_train.jsonl \
    data.validation.data_path=/path/to/BBB_Martins_val.jsonl \
    data.default.env_name=tool_calling \
    grpo.max_rollout_turns=10 \
    grpo.max_num_steps=10
```

## SLURM Usage

```bash
# Basic single-turn TDC (uses defaults: Qwen 1.5B, 1 task, 1 GPU):
sbatch scripts/train_grpo_tdc_nemorl_slurm.sh

# Multi-turn tool calling with Liger loss on 2 GPUs:
LIGER_GRPO_LOSS=1 NUM_GPUS=2 MULTI_TURN=1 \
    sbatch scripts/train_grpo_tdc_nemorl_slurm.sh

# Custom model + TIS importance sampling:
MODEL=unsloth/gpt-oss-20b-BF16 NUM_GPUS=2 TIS=1 \
    sbatch scripts/train_grpo_tdc_nemorl_slurm.sh

# Multiple tasks (space-separated):
TASK_NAMES="BBB_Martins CYP2D6_Veith" NUM_GPUS=2 \
    sbatch scripts/train_grpo_tdc_nemorl_slurm.sh

# Point to a different OpenRLHF-Tools checkout:
OPENRLHF_ROOT=/path/to/OpenRLHF-Tools sbatch scripts/train_grpo_tdc_nemorl_slurm.sh
```

## Training on Multiple TDC Datasets

### SLURM Script (Auto-Concatenated Data)

The SLURM script supports multiple tasks via `TASK_NAMES`. It auto-concatenates train and val JSONL files:

```bash
# Two tasks:
TASK_NAMES="BBB_Martins CYP2D6_Veith" sbatch scripts/train_grpo_tdc_nemorl_slurm.sh

# All TDC tasks you have data for:
TASK_NAMES="BBB_Martins CYP2D6_Veith hERG_Karim DILI Caco2_Wang" \
    sbatch scripts/train_grpo_tdc_nemorl_slurm.sh
```

Each JSONL line retains its `"task"` field, so **per-dataset eval metrics are computed automatically** even from concatenated data. In wandb, you'll see under the `eval/` panel:
- `eval/eval_BBB_Martins_accuracy`
- `eval/eval_BBB_Martins_macro_f1`
- `eval/eval_CYP2D6_Veith_accuracy`
- `eval/eval_CYP2D6_Veith_macro_f1`
- `eval/eval_avg_accuracy`
- `eval/eval_avg_macro_f1`

The same breakdown is saved to `eval_metrics/eval_step_{N}.json` with a `per_task` dict.

### Manual Concatenation

If you prefer to manage data yourself:

```bash
# Concatenate multiple task JSONL files into one:
cat data/tdc/openai_format_gpt_oss/BBB_Martins_train.jsonl \
    data/tdc/openai_format_gpt_oss/CYP2D6_Veith_train.jsonl \
    data/tdc/openai_format_gpt_oss/hERG_Karim_train.jsonl \
    > data/tdc/openai_format_gpt_oss/multi_task_train.jsonl

# Same for validation:
cat data/tdc/openai_format_gpt_oss/BBB_Martins_val.jsonl \
    data/tdc/openai_format_gpt_oss/CYP2D6_Veith_val.jsonl \
    data/tdc/openai_format_gpt_oss/hERG_Karim_val.jsonl \
    > data/tdc/openai_format_gpt_oss/multi_task_val.jsonl

# Then train on the concatenated file:
TRAIN_DATA=data/tdc/openai_format_gpt_oss/multi_task_train.jsonl \
VAL_DATA=data/tdc/openai_format_gpt_oss/multi_task_val.jsonl \
    sbatch scripts/train_grpo_tdc_nemorl_slurm.sh
```

### Data Format

Each JSONL line must have these keys (from `tdc_data_processor`):
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "answer": "Yes",
  "smiles": "CCO...",
  "label": "1",
  "task": "BBB_Martins"
}
```

The `task` field identifies which TDC dataset each sample came from. This is preserved through training and appears in eval metrics, allowing per-task accuracy breakdown even when training on concatenated data.

## Feature Flags Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen2.5-1.5B` | Model name/path |
| `NUM_GPUS` | 1 | GPUs per node |
| `NUM_NODES` | 1 | Number of nodes |
| `MULTI_TURN` | 0 | Enable multi-turn tool calling |
| `MAX_TURNS` | 10 | Max tool-calling turns |
| `CHAT_PROTOCOL` | `gpt_oss` | Chat protocol for tool calling |
| `LIGER_GRPO_LOSS` | 0 | Enable Liger Triton GRPO loss |
| `TIS` | 0 | Enable truncated importance sampling |
| `TIS_TYPE` | `tis` | TIS variant: `tis`, `icepop`, `seq-mask-tis` |
| `TIS_THRESHOLDS` | `0.5 5.0` | Low/high IS thresholds |
| `LEARNING_RATE` | `1e-6` | Optimizer learning rate |
| `MAX_STEPS` | 500 | Max training steps |
| `MAX_EPOCHS` | 1 | Max epochs |
| `TASK_NAMES` | `BBB_Martins` | Space-separated task names |
| `TRAIN_DATA` | auto | Override training JSONL path |
| `VAL_DATA` | auto | Override validation JSONL path |
| `WANDB_PROJECT` | `nemorl-tdc-grpo` | WandB project name |
| `EXTRA_OVERRIDES` | (empty) | Additional Hydra overrides |

## Output Directory Structure

After a run, you'll find under `$LOG_DIR`:
```
traces/              — step traces, first_ever_trace.json, group traces
eval_metrics/        — per-step eval JSON
tool_usage_eval/     — per-dataset tool counts
vllm_stats/          — run_timing.jsonl
dataloader_logs/     — dataset_order.jsonl
```

WandB metrics are auto-routed into `train/`, `parse/`, `vllm/`, `system/` panels.

## Files Reference

| File | Purpose |
|------|---------|
| `nemo_rl/algorithms/loss/liger_grpo_loss.py` | Liger Triton GRPO loss (LossFunction protocol) |
| `nemo_rl/environments/tool_calling/environment.py` | Tool usage metrics in step() + global_post_process |
| `nemo_rl/algorithms/grpo.py` | Tracing helpers + Liger loss branch in setup() |
| `nemo_rl/utils/logger.py` | WandB panel routing + JSON/JSONL helpers |
| `examples/configs/grpo_tdc_tool_calling.yaml` | Full Hydra config for TDC GRPO |
| `scripts/train_grpo_tdc_nemorl_slurm.sh` | SLURM launch script (mirrors OpenRLHF pattern) |
