# NeMo RL (RL-Tools) Architecture

## Execution Model

NeMo RL uses a **Ray-orchestrated, actor-based** architecture where the main process never holds model weights. All compute is pushed to Ray remote actors.

### Core Abstractions

```
Main Process (orchestrator)
  │
  ├── RayVirtualCluster (logical GPU allocation)
  │     └── PlacementGroup bundles → GPU binding
  │
  ├── RayWorkerGroup (pool of Ray actors)
  │     ├── run_all_workers_sharded_data()  → DP-sharded execution
  │     └── run_all_workers_single_data()   → broadcast execution
  │
  ├── Policy (training model)
  │     ├── Backend: DTensorPolicyWorker or MegatronPolicyWorker
  │     ├── Methods: get_logprobs(), train(), get_reference_policy_logprobs()
  │     └── Parallelism: DP × TP × PP × CP (Megatron) or DP × TP × CP (DTensor)
  │
  ├── GenerationInterface (inference model, separate from policy when using DTensor)
  │     ├── VllmGeneration (RayWorkerGroup of VllmGenerationWorker)
  │     ├── SGLangGeneration
  │     └── Or policy itself (Megatron backend has built-in generation)
  │
  └── StatefulDataLoader (checkpointable iteration)
```

### Colocated vs Non-Colocated Inference

**Colocated**: Policy and generation share the same GPUs. Between phases:
1. `policy.prepare_refit_info()` → extract state dict
2. `policy_generation.refit(state_dict)` → load weights into vLLM
3. `policy.offload_after_refit()` → unload optimizer to free GPU memory
4. Generation runs, then `policy.prepare_for_training()` reloads optimizer

**Non-colocated**: Separate GPU clusters for train and inference. Weight sync via NCCL collective communication.

### Data Sharding

```
BatchedDataDict (full batch on main process)
  ↓ NamedSharding splits by DP rank
SlicedDataDict (per-worker shard)
  ↓ Worker processes (forward/backward)
Results aggregated back to main
```

`NamedSharding` maps abstract axes (PP, DP, CP, TP) to global worker ranks, determining how data and model are partitioned.

## Training Loop Flow (GRPO)

Entry: `examples/run_grpo.py` → `setup(config)` → `grpo_train()`

```
grpo_train():
  for epoch in range(max_num_epochs):
    for batch in dataloader:

      1. PREPARE BATCH
         batch.repeat_interleave(num_generations_per_prompt)
         batched_message_log_to_flat_message() → input_ids

      2. REFIT & GENERATE
         refit_policy_generation()           # sync weights to vLLM
         run_multi_turn_rollout()            # generate + query environment
         OR run_async_multi_turn_rollout()   # pipelined async
         OR run_async_nemo_gym_rollout()     # NeMo-Gym

      3. REWARD PROCESSING
         scale_rewards()                     # linear [src_min,max] → [tgt_min,max]
         apply_reward_shaping()              # custom reward functions

      4. DYNAMIC SAMPLING (optional, DAPO)
         Filter prompts with zero-std rewards
         Cache and accumulate until batch fills

      5. COMPUTE LOGPROBS
         policy.prepare_for_lp_inference()
         prev_logprobs = policy.get_logprobs(data)
         ref_logprobs  = policy.get_reference_policy_logprobs(data)

      6. COMPUTE ADVANTAGES
         adv_estimator.compute_advantage(prompt_ids, rewards, mask, ...)

      7. TRAIN
         policy.prepare_for_training()       # reload optimizer
         policy.train(train_data, loss_fn)   # single pass: fwd → loss → bwd → step

      8. CHECKPOINT & VALIDATE (periodic)
```

### Key Architectural Note: Single Training Pass

NeMo RL does **one forward-backward pass per rollout batch**. There is no replay buffer and no multi-epoch training over collected experience. The training step is:

```
policy.train(train_data, loss_fn):
  → shard data by DP rank
  → accumulate over micro-batches (GBS / MBS steps)
  → each micro-batch: forward → loss → backward
  → all-reduce gradients
  → optimizer step
```

Gradient accumulation handles the GBS/MBS ratio, but the data is seen exactly once.

## Config System

YAML → Hydra resolvers → `MasterConfig` TypedDict

```
MasterConfig:
  policy:        # model name, backend (megatron/dtensor), parallelism
    generation:  # colocated settings, vLLM/SGLang config
  grpo:          # algorithm hyperparams, num_generations_per_prompt, max_num_steps
  data:          # dataset, tokenizer, batch sizes
  loss_fn:       # ClippedPGLossConfig
  cluster:       # GPU/node allocation
  logger:        # W&B, MLflow, TensorBoard
  checkpointing: # save path, frequency
```

All defaults live in YAML exemplar configs under `examples/configs/*.yaml`. Code must never set hidden defaults.

## Reference Policy Handling

- The reference policy is a **frozen snapshot** of the initial policy weights
- Stored internally within the policy workers (no separate model/actor)
- `policy.get_reference_policy_logprobs()` runs a separate forward pass without gradients
- For DPO: `init_reference_model=True` creates an explicit frozen copy
- Reference logprobs computed **every step** (can be skipped if KL penalty = 0)

## Model Worker Internals

### DTensorPolicyWorker
- Uses PyTorch's DTensor for FSDP2 + TP + SP + CP
- Data parallel is implicit: `DP = world_size / (TP * CP)`
- Separate vLLM/SGLang generation backend

### MegatronPolicyWorker
- Uses Megatron Core for 6D parallelism (DP, TP, PP, CP, SP, EP)
- Built-in generation (no separate generation backend needed)
- `policy_generation = None` → policy acts as both trainer and generator

## Environment Abstraction

```python
class EnvironmentInterface:
    def score(message_log, ...) → reward, metrics
```

- Environments are **not** gym-style step-by-step simulators (except NeMo-Gym)
- Default: single scoring call after full generation
- Multi-turn: loop of generate → environment.score → continue/stop
- Task-to-environment mapping: `task_to_env: dict[str, EnvironmentInterface]`
