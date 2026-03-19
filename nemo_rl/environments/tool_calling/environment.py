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

"""Tool-calling environment for multi-turn rollouts.

Implements EnvironmentInterface by parsing tool calls from assistant messages
via a ChatProtocol, executing tools via a ToolRegistry, and rendering feedback.
"""

import logging
from typing import Any, Callable, NotRequired, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.tool_calling.chat_protocol import ChatProtocol
from nemo_rl.environments.tool_calling.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolCallingMetadata(TypedDict):
    """Metadata tracked per sample across turns."""

    turn_count: int
    tool_results: list[dict[str, Any]]
    format_reward: float
    ground_truth: NotRequired[str]
    label: NotRequired[str]
    task: NotRequired[str]
    tool_counts: NotRequired[dict[str, int]]
    parse_method: NotRequired[str]
    tool_call_attempted: NotRequired[bool]
    parse_failed: NotRequired[bool]


class ToolCallingConfig(TypedDict):
    max_turns: int
    chat_protocol: str
    tools_file: NotRequired[str]
    reward_fn: NotRequired[str]


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class ToolCallingEnvironment(EnvironmentInterface[ToolCallingMetadata]):
    """Environment that executes tool calls parsed from assistant messages.

    Works with NeMo RL's run_multi_turn_rollout: each step() call processes
    the latest assistant message, executes any tool calls, and returns
    observations for the next turn.

    Accepts a config dict (matching create_env() convention) with keys:
        chat_protocol: Protocol name (default: "gpt_oss").
        max_turns: Maximum tool-calling turns (default: 10).
        tools_file: Optional path to tool schemas JSON.
        reward_fn: Optional reward function name.

    Or can be constructed directly with protocol/registry objects.
    """

    def __init__(
        self,
        cfg: Optional[dict[str, Any]] = None,
        protocol: Optional[ChatProtocol] = None,
        registry: Optional[ToolRegistry] = None,
        max_turns: int = 10,
        reward_fn: Optional[Callable[[str, str], float]] = None,
        tokenizer: Optional[Any] = None,
    ):
        if cfg is not None:
            # Config-based init (called by create_env)
            from nemo_rl.environments.tool_calling.chat_protocol import get_protocol
            from nemo_rl.environments.tool_calling.tools import get_default_tools

            self.protocol = get_protocol(cfg.get("chat_protocol", "gpt_oss"))
            self.registry = ToolRegistry()
            self.registry.register_callables(get_default_tools())
            if cfg.get("tools_file"):
                self.registry.load_schemas_from_file(cfg["tools_file"])
            self.max_turns = cfg.get("max_turns", 10)
            self.reward_fn = None
            if cfg.get("reward_fn"):
                import importlib

                module_path, fn_name = cfg["reward_fn"].rsplit(".", 1)
                mod = importlib.import_module(module_path)
                self.reward_fn = getattr(mod, fn_name)
            self.tokenizer = tokenizer
        else:
            # Direct init (for tests or programmatic use)
            assert protocol is not None and registry is not None
            self.protocol = protocol
            self.registry = registry
            self.max_turns = max_turns
            self.reward_fn = reward_fn
            self.tokenizer = tokenizer

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ToolCallingMetadata],
    ) -> EnvironmentReturn[ToolCallingMetadata]:
        """Process assistant messages: parse tool calls, execute, return feedback.

        Args:
            message_log_batch: Batch of conversation histories.
            metadata: Batch of per-sample metadata tracking turn counts etc.

        Returns:
            EnvironmentReturn with tool feedback observations or terminal rewards.
        """
        observations: list[dict[str, str]] = []
        observation_token_ids_list: list[list[int] | None] = []
        rewards_list: list[float] = []
        terminated_list: list[bool] = []
        updated_metadata: list[ToolCallingMetadata] = []
        next_stop_strings: list[list[str] | None] = []

        for conversation, meta in zip(message_log_batch, metadata):
            # Extract last assistant message
            assistant_text = ""
            for msg in reversed(conversation):
                if msg["role"] == "assistant":
                    assistant_text = str(msg["content"])
                    break

            parsed = self.protocol.parse_assistant_text(assistant_text)
            tool_calls = parsed.get("tool_calls", [])

            new_meta: ToolCallingMetadata = {
                "turn_count": meta["turn_count"] + 1,
                "tool_results": list(meta["tool_results"]),
                "format_reward": meta["format_reward"],
                "tool_counts": dict(meta.get("tool_counts", {})),
                "parse_method": parsed.get("parse_method", "unknown"),
                "tool_call_attempted": len(tool_calls) > 0,
                "parse_failed": parsed.get("parse_failed", False),
            }
            # Preserve optional fields
            for key in ("ground_truth", "label", "task"):
                if key in meta:
                    new_meta[key] = meta[key]  # type: ignore[literal-required]

            if tool_calls and new_meta["turn_count"] < self.max_turns:
                # Execute tool calls and track per-tool counts
                tool_results = []
                for tc in tool_calls:
                    name = tc.get("name", "")
                    arguments = tc.get("arguments", {})
                    result_json = self.registry.execute(name, arguments)
                    tool_results.append({"name": name, "content": result_json})
                    new_meta["tool_counts"][name] = (
                        new_meta["tool_counts"].get(name, 0) + 1
                    )

                new_meta["tool_results"].extend(tool_results)

                # Track parse quality for format reward
                if parsed.get("parse_failed"):
                    new_meta["format_reward"] += -0.0025
                else:
                    new_meta["format_reward"] += 0.0

                feedback = self.protocol.render_tool_feedback(tool_results)
                feedback_token_ids = self.protocol.render_tool_feedback_token_ids(
                    tool_results, tokenizer=self.tokenizer
                )
                observations.append({"role": "environment", "content": feedback})
                observation_token_ids_list.append(feedback_token_ids)
                rewards_list.append(new_meta["format_reward"])
                terminated_list.append(False)
                next_stop_strings.append(self.protocol.stop_strings or None)
            else:
                # No tool calls or max turns reached -> terminal
                content_text = parsed.get("content", assistant_text)
                reward = self._compute_terminal_reward(content_text, meta)

                # Add penalty for parse failures
                if parsed.get("parse_failed"):
                    new_meta["format_reward"] += -0.0025

                observations.append({"role": "environment", "content": ""})
                observation_token_ids_list.append(None)
                rewards_list.append(reward + new_meta["format_reward"])
                terminated_list.append(True)
                next_stop_strings.append(None)

            updated_metadata.append(new_meta)

        return EnvironmentReturn(
            observations=observations,
            metadata=updated_metadata,
            next_stop_strings=next_stop_strings,
            rewards=torch.tensor(rewards_list).cpu(),
            terminateds=torch.tensor(terminated_list, dtype=torch.float).cpu(),
            answers=None,
            observation_token_ids=observation_token_ids_list,
        )

    def _compute_terminal_reward(
        self, generated_text: str, meta: ToolCallingMetadata
    ) -> float:
        """Compute the terminal reward for a completed episode.

        Uses the injected reward_fn if available, otherwise defaults to 0.0.
        """
        label = meta.get("ground_truth", meta.get("label", ""))
        if self.reward_fn is not None and label:
            return self.reward_fn(generated_text, label)
        return 0.0

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Compute tool-calling metrics across the batch.

        Aggregates per-tool counts, parse method distribution, and parse failure
        rates across the batch. Metric keys use the naming convention expected by
        wandb panel routing (tool_count__*, parse_method__*, parse_failed, tool_call_attempted).
        """
        batch["rewards"] = batch["rewards"] * batch["is_end"]

        metrics: dict[str, float | int] = {
            "accuracy": batch["rewards"].mean().item(),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }

        # Aggregate tool usage metrics from metadata if available
        metadata_list = batch.get("metadata")
        if metadata_list and isinstance(metadata_list, list):
            total_tool_counts: dict[str, int] = {}
            parse_method_counts: dict[str, int] = {}
            num_parse_failed = 0
            num_tool_call_attempted = 0
            num_samples = len(metadata_list)

            for meta in metadata_list:
                if not isinstance(meta, dict):
                    continue
                # Aggregate per-tool counts
                for tool_name, count in meta.get("tool_counts", {}).items():
                    total_tool_counts[tool_name] = (
                        total_tool_counts.get(tool_name, 0) + count
                    )
                # Track parse method distribution
                pm = meta.get("parse_method", "unknown")
                parse_method_counts[pm] = parse_method_counts.get(pm, 0) + 1
                if meta.get("parse_failed"):
                    num_parse_failed += 1
                if meta.get("tool_call_attempted"):
                    num_tool_call_attempted += 1

            # Per-tool normalized frequencies
            total_calls = sum(total_tool_counts.values()) or 1
            for tool_name, count in total_tool_counts.items():
                metrics[f"tool_count__{tool_name}"] = count
                metrics[f"tool_freq__{tool_name}"] = count / total_calls

            # Parse method distribution
            for method, count in parse_method_counts.items():
                metrics[f"parse_method__{method}"] = (
                    count / num_samples if num_samples else 0
                )

            metrics["parse_failed"] = (
                num_parse_failed / num_samples if num_samples else 0
            )
            metrics["tool_call_attempted"] = (
                num_tool_call_attempted / num_samples if num_samples else 0
            )

        return batch, metrics

    @classmethod
    def collect_eval_tool_usage(
        cls,
        metadata_list: list[ToolCallingMetadata],
        step: int,
        dataset_name: str = "default",
    ) -> dict[str, Any]:
        """Collect per-dataset tool usage summary for eval-time JSON output.

        Args:
            metadata_list: List of metadata dicts from completed episodes.
            step: Current training step.
            dataset_name: Name of the evaluation dataset.

        Returns:
            Dict suitable for writing to tool_usage_eval/eval_step_{N}.json.
        """
        total_tool_counts: dict[str, int] = {}
        num_samples = len(metadata_list)

        for meta in metadata_list:
            for tool_name, count in meta.get("tool_counts", {}).items():
                total_tool_counts[tool_name] = (
                    total_tool_counts.get(tool_name, 0) + count
                )

        total_calls = sum(total_tool_counts.values()) or 1
        normalized = {
            name: count / total_calls for name, count in total_tool_counts.items()
        }

        return {
            "step": step,
            "dataset": dataset_name,
            "num_samples": num_samples,
            "total_tool_calls": sum(total_tool_counts.values()),
            "tool_counts": total_tool_counts,
            "tool_frequencies": normalized,
        }
