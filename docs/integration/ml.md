# NeMo RL (RL-Tools) — ML Training Mechanics

## Loss Functions

All loss functions implement `LossFunction` protocol:
```python
(logprobs/logits, data: BatchedDataDict, global_valid_seqs, global_valid_toks) → (loss, metrics)
```

### ClippedPGLossFn (`nemo_rl/algorithms/loss/loss_functions.py`)

The **unified** policy gradient loss covering PPO, GRPO, REINFORCE/RLOO, GSPO, and DAPO variants.

**Core formula:**
```
L(θ) = E_t[ min(r_t · A_t, clip(r_t, 1-ε_lo, 1+ε_hi) · A_t) ] + β · KL(π_θ || π_ref)
```

**Ratio computation modes:**
| Mode | Ratio r_t | Config |
|------|-----------|--------|
| PPO/GRPO (default) | `exp(log π_θ - log π_prev)` | `disable_ppo_ratio=False` |
| REINFORCE/RLOO | `log π_θ` directly (no ratio) | `disable_ppo_ratio=True` |
| GSPO | Sequence-level: `exp(mean_t(log π_θ - log π_prev))` | `sequence_level_importance_ratios=True` |
| Truly on-policy | Forced to 1.0 (gradient still flows) | `force_on_policy_ratio=True` |

**Dual-clipping (optional):**
When advantages < 0: `max(clip_loss, c · A_t)` where c > 1 (typically 3).

**Loss reduction:**
- `token_level_loss=True` → flat token mean (DAPO-style global normalization)
- `token_level_loss=False` → per-sequence mean, then batch mean (GRPO-style)

### KL Divergence

**Schulman approximations** (`nemo_rl/algorithms/utils.py:calculate_kl`):
- K1: `-log_ratio` (linear)
- K2: `log_ratio² / 2` (quadratic)
- K3: `exp(log_ratio) - 1 - log_ratio` (tight, default)

**Application:**
- Standard: KL added to loss `L = actor_loss + β · KL(π_θ || π_ref)`
- On-policy approximation: weighted by importance ratio `(π_curr / π_gen)`
- Reinforce++: KL added to **reward** instead of loss (`use_kl_in_reward=True`)

**Clamping:** Optional `kl_input_clamp_value` and `kl_output_clamp_value` for stability.

### Off-Policy Importance Sampling Correction

When the generation policy differs from the training policy (multi-epoch or async):

**Token-level IS (default):**
```
w_i = exp(log π_prev - log π_gen)
loss = w_i · clip_loss
```

**Sequence-level IS (GSPO):**
```
w_i = exp(Σ_t (log π_prev - log π_gen))
```

**Truncated IS variants:**
| Type | Behavior |
|------|----------|
| `tis` | Clamp weights to `[0, max_ratio]` |
| `icepop` | Zero out tokens with IS weight outside `[min, max]` |
| `seq-mask-tis` | Zero entire sequences by geometric-mean IS ratio; retained sequences keep raw token-level IS |

### NLLLossFn

Standard negative log-likelihood for SFT. Token-level masked mean with global normalization.

### DPOLossFn (`nemo_rl/algorithms/loss/loss_functions.py:723`)

```
L(θ) = w_p · L_pref + w_s · L_sft

L_pref = -E[log σ(β · (r_chosen - r_rejected))]
  where r = Σ_t (log π_θ - log π_ref)  [optionally averaged over tokens]

L_sft = NLL on chosen responses only
```

### DistillationLossFn

Top-k logit distillation with forward/reverse/mixed KL. Handles tail distribution via correction term when `zero_outside_topk=True`.

## Advantage Estimation

### GRPOAdvantageEstimator (`nemo_rl/algorithms/advantage_estimator.py:30`)

```python
baseline, std = calculate_baseline_and_std_per_prompt(prompt_ids, rewards)
advantages = (rewards - baseline) / (std + ε)  # if normalize_rewards=True
```

- Groups samples by prompt (via `prompt_ids` tensor)
- **Leave-one-out baseline** (optional, `use_leave_one_out_baseline`): baseline_i = mean(rewards excluding i)
- Standard deviation normalization: only applied to groups with std > 0
- Result broadcast to token-level shape `[batch, seq_len]`

### ReinforcePlusPlusAdvantageEstimator (`advantage_estimator.py:72`)

```python
# Optional per-prompt baseline subtraction
adv = rewards - mean(rewards_per_prompt)  # if minus_baseline=True

# Optional token-level KL penalty in reward
adv = adv - kl_coef * KL(π_policy, π_ref)  # if use_kl_in_reward=True

# Global normalization across entire batch
adv = (adv - global_mean) / global_std
```

Key difference from GRPO: **global batch normalization** instead of per-prompt normalization.

### What NeMo RL Does NOT Have

- **No GAE** (Generalized Advantage Estimation) — no temporal credit assignment
- **No critic/value network** — advantages derived purely from reward statistics
- **No cumulative discounted returns** — reward is a single scalar per sequence
- **No RLOO as a separate estimator** — RLOO is handled via `use_leave_one_out_baseline` in GRPOAdvantageEstimator

## Mini-Batch Gradient Steps

### NeMo RL: Single Pass

```
Collected experience (GBS samples)
  ↓ shard by DP rank
  ↓ split into micro-batches (MBS)
  ↓ forward → loss → backward (accumulated)
  ↓ all-reduce gradients
  ↓ optimizer step
  ↓ DONE — data seen exactly once
```

- `train_global_batch_size` (GBS): total samples per optimizer step
- `train_micro_batch_size` (MBS): samples per forward pass
- Gradient accumulation steps = GBS / (MBS × DP_size)
- **No multi-epoch training** over collected data
- **No replay buffer** — experience is discarded after one training step

### Config:
```yaml
policy:
  train_global_batch_size: 64    # total per optimizer step
  train_micro_batch_size: 4      # per forward pass per GPU
grpo:
  num_prompts_per_step: 8        # prompts per rollout
  num_generations_per_prompt: 8  # responses per prompt → 64 total samples
```

## Experience Collection

### No Replay Buffer

NeMo RL generates experience **synchronously** within the training loop:
1. Dataloader yields a batch of prompts
2. `repeat_interleave(num_generations_per_prompt)` → expand prompts
3. Generate responses via vLLM/SGLang/Megatron
4. Query environment for rewards
5. Compute logprobs, advantages, and train — in one shot
6. Move to next batch

### Dynamic Sampling (DAPO)

Optional filtering that only keeps prompts where responses have non-zero reward std:
- Prompts with all-same rewards provide no gradient signal
- Filtered-out prompts are cached and combined with next batch
- Continues generating until batch fills or `dynamic_sampling_max_gen_batches` hit

### Rollout Paths

1. **Synchronous multi-turn** (`run_multi_turn_rollout`): generate → env.score → maybe continue
2. **Async multi-turn** (`run_async_multi_turn_rollout`): pipelined via vLLM async engine
3. **NeMo-Gym** (`run_async_nemo_gym_rollout`): specialized for NeMo-Gym environments

### Data Representation

```
LLMMessageLog → list of dicts [{role, content, token_ids, generation_logprobs}, ...]
  ↓ batched_message_log_to_flat_message()
FlatMessages → {token_ids: [B, S], token_loss_mask: [B, S], generation_logprobs: [B, S], ...}
  ↓ wrapped in BatchedDataDict
train_data → {input_ids, token_mask, sample_mask, advantages, prev_logprobs, ...}
```

## Reward Processing

### Reward Scaling
Linear mapping: `reward = clamp(reward, src_min, src_max)` then linearly map to `[tgt_min, tgt_max]`.

### Reward Shaping
Custom reward functions applied after scaling. Configured via `RewardShapingConfig`.

### Sequence-Level Logprob Error Masking
Optional: mask out sequences where `exp(|gen_logprobs - prev_logprobs|)` exceeds threshold. Guards against stale generations.

## Metrics Tracked Per Step

**Ratio metrics:** mean, min, max, clamped versions
**KL metrics:** gen-kl (π_gen vs π_train), policy-kl (π_train vs π_gen), JS divergence
**Stability:** token_mult_prob_error, sampling_importance_ratio, approx_entropy
**Filtering:** is_oob_ratio (fraction filtered by IS truncation)
**Rewards:** raw, shaped, baseline, advantages (mean/min/max)
