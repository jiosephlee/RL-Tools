# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TDC (Therapeutics Data Commons) environment for molecular property prediction.

Provides a single-turn environment that scores model responses against
ground-truth binary labels (e.g., "(A)" or "(B)") for TDC tasks.
"""

import re
from collections import defaultdict
from typing import Any, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class TDCEnvironmentMetadata(TypedDict):
    ground_truth: str
    smiles: str
    label: str
    task: str


def extract_final_answer(text: str) -> str:
    """Extract the final answer from model output.

    Looks for patterns like "Answer: (A)", "Final answer: (B)", or bare "(A)"/"(B)".

    Args:
        text: Model-generated text.

    Returns:
        Extracted answer (e.g., "(A)") or empty string if not found.
    """
    # Pattern 1: "Answer: (A)" or "Answer:(A)"
    match = re.search(r"Answer:\s*\(([AB])\)", text, re.IGNORECASE)
    if match:
        return f"({match.group(1)})"

    # Pattern 2: "Final answer: (B)"
    match = re.search(r"Final\s+answer:\s*\(([AB])\)", text, re.IGNORECASE)
    if match:
        return f"({match.group(1)})"

    # Pattern 3: Last occurrence of (A) or (B)
    matches = list(re.finditer(r"\(([AB])\)", text))
    if matches:
        return f"({matches[-1].group(1)})"

    return ""


def compute_reward(generated_text: str, label: str) -> float:
    """Compute binary reward for a single generated response.

    Args:
        generated_text: Model-generated text.
        label: Ground truth label (e.g., "(A)" or "(B)").

    Returns:
        1.0 if correct, 0.0 if incorrect.
    """
    predicted = extract_final_answer(generated_text)
    if not predicted:
        return 0.0
    if predicted.upper() == label.upper():
        return 1.0
    return 0.0


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class TDCEnvironment(EnvironmentInterface[TDCEnvironmentMetadata]):
    """Environment for TDC molecular property prediction tasks.

    Single-turn: the model generates a response, and we check if its
    answer matches the ground-truth label.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[TDCEnvironmentMetadata],
    ) -> EnvironmentReturn[TDCEnvironmentMetadata]:
        """Score model responses against ground-truth TDC labels.

        Args:
            message_log_batch: Batch of conversation histories.
            metadata: Batch of TDC metadata with ground_truth labels.

        Returns:
            EnvironmentReturn with binary rewards and terminated=True.
        """
        results = []
        for conversation, meta in zip(message_log_batch, metadata):
            # Extract all assistant responses
            assistant_text = "".join(
                str(msg["content"]) for msg in conversation if msg["role"] == "assistant"
            )
            reward = compute_reward(assistant_text, meta["ground_truth"])
            results.append(reward)

        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        observations = [
            {"role": "environment", "content": "Environment: correct" if r == 1.0 else "Environment: incorrect"}
            for r in results
        ]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards,
            terminateds=done,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Compute per-task accuracy metrics."""
        batch["rewards"] = batch["rewards"] * batch["is_end"]

        metrics: dict[str, float | int] = {
            "accuracy": batch["rewards"].mean().item(),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }

        # Per-task accuracy if task info is available in extra_env_info
        if "extra_env_info" in batch and batch["extra_env_info"] is not None:
            task_rewards: dict[str, list[float]] = defaultdict(list)
            env_infos = batch["extra_env_info"]
            if isinstance(env_infos, list):
                for i, info in enumerate(env_infos):
                    if info and "task" in info:
                        task_rewards[info["task"]].append(batch["rewards"][i].item())
                for task_name, rewards_list in task_rewards.items():
                    metrics[f"accuracy/{task_name}"] = sum(rewards_list) / len(rewards_list)

        return batch, metrics
