# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Liger Triton kernel-based GRPO loss function.

Uses liger_kernel's fused Triton kernels for log-softmax + GRPO loss + backward,
avoiding storage of the full log-softmax tensor by recomputing from saved LSE.
Operates on pre-computed logits (LossInputType.LOGIT).
"""

from typing import Any, NotRequired, Optional, TypedDict

import torch

from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType
from nemo_rl.algorithms.utils import masked_mean
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

try:
    from liger_kernel.transformers.grpo_loss import triton_grpo_loss

    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False


class LigerGRPOLossConfig(TypedDict):
    """Configuration for LigerGRPOLossFn."""

    clip_eps_low: float
    clip_eps_high: float
    beta: float
    temperature: float
    loss_type: str  # "grpo", "ipo", "simpo", etc.
    # IS correction settings
    enable_vllm_is_correction: NotRequired[bool]
    vllm_is_correction_type: NotRequired[str]  # "tis", "icepop", "seq-mask-tis"
    vllm_is_truncated_threshold: NotRequired[list[float]]  # [low, high]


class LigerGRPOLossFn:
    """GRPO loss using Liger Triton fused kernels.

    Implements the LossFunction protocol with LossInputType.LOGIT, so
    prepare_loss_input() passes raw logits directly to __call__.
    """

    input_type = LossInputType.LOGIT
    loss_type = LossType.TOKEN_LEVEL

    def __init__(self, cfg: LigerGRPOLossConfig):
        if not _LIGER_AVAILABLE:
            raise ImportError(
                "liger_kernel is required for LigerGRPOLossFn. "
                "Install with: pip install liger-kernel>=0.7.0"
            )
        self._triton_grpo_loss = triton_grpo_loss
        self._eps_low = cfg["clip_eps_low"]
        self._eps_high = cfg["clip_eps_high"]
        self._beta = cfg["beta"]
        self._temperature = cfg["temperature"]
        self._loss_type = cfg["loss_type"]
        self._enable_vllm_is_correction = cfg.get("enable_vllm_is_correction", False)
        self._vllm_is_correction_type = cfg.get("vllm_is_correction_type", "tis")
        self._vllm_is_truncated_threshold = cfg.get("vllm_is_truncated_threshold")

        if self._enable_vllm_is_correction and self._vllm_is_correction_type not in {
            "tis",
            "icepop",
            "seq-mask-tis",
        }:
            raise ValueError(
                f"Invalid vllm_is_correction_type: {self._vllm_is_correction_type}, "
                "must be one of tis/icepop/seq-mask-tis"
            )

    def _compute_vllm_is_ratio(
        self,
        old_log_probs: torch.Tensor,
        rollout_log_probs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute vLLM importance sampling ratio (TIS/ICEPOP/seq-mask-tis).

        Returns:
            (vllm_is_ratio, vllm_kl): Per-token IS ratio tensor and KL scalar.
        """
        if not self._enable_vllm_is_correction or rollout_log_probs is None:
            return None, None

        low_threshold, high_threshold = self._vllm_is_truncated_threshold
        log_ratio = old_log_probs - rollout_log_probs

        if self._vllm_is_correction_type == "icepop":
            vllm_is = torch.exp(log_ratio).detach()
            mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
            vllm_is_ratio = vllm_is * mask
        elif self._vllm_is_correction_type == "seq-mask-tis":
            seq_log_ratio = masked_mean(log_ratio, action_mask, dim=-1)
            seq_is = torch.exp(seq_log_ratio)
            seq_mask = (seq_is >= low_threshold) & (seq_is <= high_threshold)
            vllm_is = torch.exp(log_ratio).detach()
            vllm_is_ratio = seq_mask.unsqueeze(-1) * vllm_is
        else:
            # TIS: token-level clamp
            vllm_is_ratio = (
                torch.exp(log_ratio)
                .clamp(min=low_threshold, max=high_threshold)
                .detach()
            )

        vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)
        return vllm_is_ratio, vllm_kl

    def __call__(
        self,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        *,
        logits: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute Liger Triton GRPO loss from logits.

        Args:
            data: BatchedDataDict with keys: input_ids, prev_logprobs, reference_policy_logprobs,
                  token_mask, sample_mask, advantages, generation_logprobs.
            global_valid_seqs: Number of valid sequences for normalization.
            global_valid_toks: Number of valid tokens for normalization.
            logits: Model logits, shape [B, T, V].

        Returns:
            (loss, metrics_dict)
        """
        # Shifted targets for next-token prediction
        completion_ids = data["input_ids"][:, 1:].contiguous()

        # Build action mask: token_mask * sample_mask
        action_mask = (
            data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
        ).contiguous()

        old_logp = data["prev_logprobs"][:, 1:].contiguous()
        advantages = data["advantages"][:, 1:].contiguous()

        ref_logp = None
        if self._beta > 0 and "reference_policy_logprobs" in data:
            ref_logp = data["reference_policy_logprobs"][:, 1:].contiguous()

        # Compute vLLM IS ratio if enabled
        vllm_is_ratio = None
        vllm_kl = None
        if self._enable_vllm_is_correction and "generation_logprobs" in data:
            vllm_is_ratio, vllm_kl = self._compute_vllm_is_ratio(
                old_logp, data["generation_logprobs"][:, 1:], action_mask
            )

        loss, metrics = self._triton_grpo_loss(
            logits=logits.contiguous(),
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=action_mask,
            temperature=self._temperature,
            beta=self._beta,
            eps_low=self._eps_low,
            eps_high=self._eps_high,
            loss_type=self._loss_type,
            reduce=True,
            vllm_is_ratio=vllm_is_ratio,
        )

        # Unpack metrics: [kl, clip_ratio] if beta>0, else [clip_ratio]
        clip_ratio = metrics[-1]
        ppo_kl = metrics[0] if self._beta > 0 else torch.tensor(0.0, device=loss.device)

        num_valid = action_mask.sum().item()
        result_metrics = {
            "loss": loss.detach().item(),
            "clip_ratio": clip_ratio.detach().item()
            if isinstance(clip_ratio, torch.Tensor)
            else clip_ratio,
            "ppo_kl": ppo_kl.detach().item()
            if isinstance(ppo_kl, torch.Tensor)
            else ppo_kl,
            "num_valid_samples": num_valid,
        }
        if vllm_kl is not None:
            result_metrics["vllm_kl"] = vllm_kl.detach().item()

        return loss, result_metrics
