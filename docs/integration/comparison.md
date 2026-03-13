# Comparison: NeMo RL (RL-Tools) vs OpenRLHF (OpenRLHF-Tools)

## Executive Summary

NeMo RL is a **production-grade NVIDIA framework** optimized for massive scale (100B+ params, thousands of GPUs) with Megatron/DTensor backends. OpenRLHF is a **research-flexible framework** with richer RL algorithm support (PPO with critic, GAE, replay buffers) built on DeepSpeed + Ray. The key integration targets are: critic/value networks, GAE, replay buffers, multi-epoch training, and the Liger fused loss.

---

## 1. Distributed Architecture

| Aspect | NeMo RL | OpenRLHF |
|--------|---------|----------|
| **Orchestration** | Ray actors (main process never holds weights) | Ray actors + DeepSpeed strategies |
| **Training backend** | DTensor (FSDP2/TP/SP/CP) or Megatron Core (6D parallelism) | DeepSpeed ZeRO-2/3 |
| **Inference backend** | vLLM, SGLang, or Megatron (built-in) | vLLM exclusively |
| **Worker abstraction** | `RayWorkerGroup` → `RayVirtualCluster` with `NamedSharding` | `BaseModelActor` (Ray actors with DeepSpeed) |
| **Weight sync** | `refit_policy_generation()` via state_dict or NCCL collectives | `broadcast_to_vllm()` via NCCL/CUDA IPC |
| **Colocated inference** | Yes — shared GPU with optimizer offload between phases | Yes — `colocate_actor_ref`, `colocate_critic_reward` |
| **Non-colocated** | Yes — separate `RayVirtualCluster` per component | Yes — separate Ray actor groups |
| **Config system** | Hydra/OmegaConf YAML → `MasterConfig` TypedDict | argparse CLI flags |

**Key difference:** NeMo RL's `RayVirtualCluster` + `NamedSharding` provides fine-grained GPU topology control that OpenRLHF doesn't have. OpenRLHF relies on DeepSpeed for parallelism rather than explicit sharding annotations.

---

## 2. Model Abstractions

| Component | NeMo RL | OpenRLHF |
|-----------|---------|----------|
| **Actor/Policy** | `Policy` → `RayWorkerGroup` of DTensor/Megatron workers | `Actor` (HF model + optional LoRA + DeepSpeed) |
| **Critic/Value** | **NOT PRESENT** | `CriticPPOTrainer` — separate value head model |
| **Reference** | Frozen weights inside policy workers; `get_reference_policy_logprobs()` | Separate `BaseModelActor` or colocated with actor |
| **Reward Model** | `EnvironmentInterface.score()` — external environments | `RewardModel` actor group OR custom Python function via `--remote_rm_url` |
| **Generation** | Separate `GenerationInterface` (vLLM/SGLang) or built-in (Megatron) | `LLMRayActor` wrapping vLLM engines |

**Critical gap:** NeMo RL has **no critic/value network**. All advantage estimation is reward-statistics-based (GRPO, Reinforce++). This means no GAE, no temporal credit assignment.

---

## 3. Advantage Estimation

| Estimator | NeMo RL | OpenRLHF |
|-----------|---------|----------|
| **GRPO (group_norm)** | `(r - mean) / std` per prompt group | `(r - mean) / (std + 1e-9)` per prompt group |
| **Dr. GRPO (dr_grpo)** | Not present (GRPO with `normalize_rewards=False` is close) | `r - mean` per prompt group |
| **RLOO** | Via `use_leave_one_out_baseline=True` in GRPO estimator | `(sum - r_i) / (n-1)` per prompt group |
| **Reinforce++** | Separate estimator with global batch normalization | `reinforce` mode with global normalization |
| **Reinforce++-baseline** | `minus_baseline=True` in Reinforce++ estimator | `reinforce_baseline` mode |
| **GAE** | **NOT PRESENT** | Full GAE: `δ_t + (γλ)δ_{t+1} + ...` with critic values |
| **KL in reward** | Reinforce++ supports `use_kl_in_reward` (token-level) | Applied in `compute_reward()` before advantage |

**NeMo RL advantages are always sequence-level scalars** broadcast to all tokens. OpenRLHF supports **token-level advantages** via GAE where each token has a different advantage value based on temporal credit assignment.

---

## 4. Loss Functions

### Policy Gradient Loss

| Feature | NeMo RL (`ClippedPGLossFn`) | OpenRLHF (`PolicyLoss`) |
|---------|---------------------------|------------------------|
| **PPO clipping** | `clip(r, 1-ε_lo, 1+ε_hi)` (asymmetric, DAPO) | `clip(r, 1-ε_lo, 1+ε_hi)` (asymmetric) |
| **Dual clipping** | Yes (`ratio_clip_c`) | Yes (`dual_clip`) |
| **GSPO** | `sequence_level_importance_ratios=True` | `policy_loss_type="gspo"` |
| **Token-level loss** | `token_level_loss=True/False` | `token_level_loss=None/"local_rank"/"global"` |
| **IS correction** | Token-level or sequence-level with TIS/ICE-POP/seq-mask-tis | Token-level with TIS/ICE-POP/seq-mask-tis |
| **Liger fused loss** | **NOT PRESENT** | `LigerPolicyLoss` — fused lm_head + loss (never materializes full logits) |
| **Entropy bonus** | **NOT PRESENT** (approximated for metrics only) | `entropy_loss_coef * entropy` added to loss |
| **MoE aux loss** | Logged but computed separately | `aux_loss_coef * aux_loss` added to loss |

### Other Losses

| Loss | NeMo RL | OpenRLHF |
|------|---------|----------|
| **DPO** | `DPOLossFn` with optional SFT component | `DPOLoss` with similar formulation |
| **Value/Critic** | **NOT PRESENT** | `ValueLoss` — clipped MSE: `max((V_clip - R)², (V - R)²)` |
| **SFT** | `NLLLossFn` | `SFTLoss` + `GPTLMLoss` |
| **Distillation** | `DistillationLossFn` (top-k, forward/reverse/mixed KL) | Not present as separate loss |
| **Reward Model** | `PreferenceLossFn` (Bradley-Terry) | Not in this fork (inherited from base) |

---

## 5. Training Loop & Experience Management

### Mini-Batch Steps

| Aspect | NeMo RL | OpenRLHF |
|--------|---------|----------|
| **Epochs over experience** | **1 (single pass)** | `max_epochs` (default configurable, typically 1-4) |
| **Replay buffer** | **NONE** — train immediately, discard | `NaiveReplayBuffer` — stores experience, DataLoader iterates |
| **Gradient accumulation** | GBS / (MBS × DP_size) micro-batches | DeepSpeed gradient accumulation |
| **Shuffling** | No (single pass, no shuffle) | Yes — DataLoader shuffle between epochs |
| **Dynamic batching** | Not present | `use_dynamic_batch`, `use_adaptive_batch` |
| **Sequence packing** | Handled in data pipeline | `packing_samples` in replay buffer collate |

**This is the biggest behavioral difference.** OpenRLHF can train for multiple epochs over each batch of collected experience (with shuffled mini-batches and a replay buffer). NeMo RL sees each experience exactly once.

### Training Step Comparison

**NeMo RL:**
```
policy.train(train_data, loss_fn):
  shard by DP → micro-batch accumulation → all-reduce → optimizer step
  (one pass, data discarded)
```

**OpenRLHF:**
```
replay_buffer.append(experience)
for epoch in range(max_epochs):
  dataloader = DataLoader(replay_buffer, shuffle=True)
  for micro_batch in dataloader:
    forward → loss → backward → optimizer step
replay_buffer.clear()
```

---

## 6. Experience Collection

| Aspect | NeMo RL | OpenRLHF |
|--------|---------|----------|
| **Rollout** | `run_multi_turn_rollout()` / async / NeMo-Gym | `AgentExecutor.execute()` via vLLM |
| **Multi-turn** | Native support (loop: generate → env.score → continue) | `MultiTurnAgentExecutor` with tool calling |
| **Tool calling** | Not present | Rich ChatProtocol system (GLMFlash, Qwen3, etc.) |
| **Action masking** | `token_loss_mask` on assistant messages | `action_ranges` tracking per-turn LLM outputs |
| **Rollout logprobs** | Stored in `generation_logprobs` per message | Stored in `experience.rollout_log_probs` |
| **Dynamic sampling (DAPO)** | Filter zero-std prompts, cache and accumulate | Not present as explicit feature |
| **ERL / Smart replay** | Not present | Reflection-augmented retry for hard prompts |

### Data Flow

**NeMo RL:**
```
LLMMessageLog → batched_message_log_to_flat_message() → FlatMessages → BatchedDataDict
```

**OpenRLHF:**
```
Prompts → vLLM generate → AgentExecutor → Experience(sequences, action_mask, rewards, ...)
```

---

## 7. Reference Policy & KL

| Aspect | NeMo RL | OpenRLHF |
|--------|---------|----------|
| **Storage** | Frozen weights inside policy workers | Separate Ray actor (or colocated) |
| **Computation** | `policy.get_reference_policy_logprobs()` | Async Ray call to reference actor |
| **KL types** | K1 (linear), K2 (quadratic), K3 (exponential, default) | `compute_approx_kl()` — log-ratio based |
| **KL application** | In loss (default) or in reward (Reinforce++) | In reward: `r_final = r - kl_coef * kl` |
| **Skip option** | `skip_reference_policy_logprobs_calculation` if KL=0 | Always computed if ref model exists |

---

## 8. Features NeMo RL Lacks (Integration Targets)

### High Priority

1. **Critic/Value Network** — Required for GAE and PPO. OpenRLHF has `CriticPPOTrainer` with:
   - Separate value head model
   - Clipped value loss: `0.5 * max((V_clip - R)², (V - R)²)`
   - Colocatable with reward model
   - Own DeepSpeed training loop

2. **GAE (Generalized Advantage Estimation)** — Token-level temporal credit assignment:
   ```
   δ_t = r_t + γ·V(t+1) - V(t)
   A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
   ```

3. **Replay Buffer + Multi-Epoch Training** — Train for N epochs over collected experience with shuffled mini-batches. Critical for sample efficiency.

4. **Liger Fused Loss** — Fuses lm_head projection with loss computation, never materializes full `[B, T, V]` logits tensor. Major memory savings for large vocab models. Two backends:
   - Chunked: processes vocabulary in chunks
   - Triton: fused kernel for log-softmax + loss + backward

### Medium Priority

5. **Entropy Bonus** — `-entropy_coef * H(π)` added to policy loss. NeMo RL computes entropy as a metric but doesn't use it in the loss.

6. **Token-Level Loss Normalization Modes** — OpenRLHF has three modes:
   - `None` → sequence mean then batch mean (GRPO)
   - `"local_rank"` → flat token mean within rank
   - `"global"` → cross-rank token normalization (DAPO)

   NeMo RL has `token_level_loss` bool (True/False) which covers the first two but not the cross-rank global normalization.

7. **vLLM Importance Sampling Correction** — Both have TIS/ICE-POP/seq-mask-tis, but OpenRLHF's operates on `rollout_log_probs` (from vLLM at generation time) vs `generation_logprobs` (from the policy at generation time). The distinction matters when vLLM weights lag behind the trained policy.

### Lower Priority

8. **Multi-Turn Tool Calling** — OpenRLHF has rich ChatProtocol implementations for structured tool calling (GLMFlash, InternS1, Qwen3, GPTOSS). NeMo RL has multi-turn rollout but no tool-calling protocol layer.

9. **ERL (Experiential RL)** — Smart replay with reflection-augmented retries for hard prompts. Variable group sizes per prompt.

10. **Distillation Loss in Policy Training** — OpenRLHF can add `distill_coef * (-sft_logps)` to the policy loss during RL training, not just as a separate algorithm.

11. **FP4/MXFP4 Quantization** — OpenRLHF supports FP4 weight compression for vLLM broadcast and MXFP4 QAT for expert weights.

---

## 9. Features OpenRLHF Lacks

1. **Megatron Core backend** — 6D parallelism for 100B+ models
2. **DTensor/FSDP2 backend** — PyTorch-native distributed tensors
3. **Pipeline parallelism in training** — OpenRLHF is DP + TP only
4. **NeMo-Gym integration** — Specialized environment interface
5. **Distillation algorithm** — Top-k logit distillation with forward/reverse/mixed KL
6. **Reward model training** — `PreferenceLossFn` for Bradley-Terry
7. **Hydra config system** — Type-safe YAML configs with TypedDict validation
8. **Dynamic sampling (DAPO)** — Filter zero-variance prompt groups
9. **Sequence-level logprob error masking** — Stability guard for stale generations
10. **Vision-Language Model support** — Multimodal data pipeline and generation

---

## 10. Terminology Mapping

| Concept | NeMo RL | OpenRLHF |
|---------|---------|----------|
| Policy model | `Policy` | `Actor` |
| Generation model | `VllmGeneration` / `SGLangGeneration` | `LLMRayActor` |
| Training step size | `train_global_batch_size` | `train_batch_size` |
| Forward pass size | `train_micro_batch_size` | `micro_train_batch_size` |
| Samples per prompt | `num_generations_per_prompt` | `n_samples_per_prompt` |
| Response mask | `token_mask` | `action_mask` |
| Sample validity | `sample_mask` | implicit (all valid) |
| Rollout logprobs | `generation_logprobs` | `rollout_log_probs` |
| Previous policy logprobs | `prev_logprobs` | `old_action_log_probs` |
| Reference logprobs | `reference_policy_logprobs` | `base_action_log_probs` |
| Weight sync | `refit_policy_generation()` | `broadcast_to_vllm()` |
| Loss reduction | `LossType.TOKEN_LEVEL / SEQUENCE_LEVEL` | `token_level_loss: None / "local_rank" / "global"` |
