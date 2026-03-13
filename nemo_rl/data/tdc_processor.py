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

"""TDC (Therapeutics Data Commons) data processor for molecular property prediction tasks."""

from typing import Any

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log


def tdc_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: Any,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a TDC datum into a DatumSpec for RL training.

    TDC data is in OpenAI format: each record has a single user message
    (containing instructions + SMILES), plus metadata fields (answer, smiles,
    label, task). There is no assistant message to strip — the model generates
    its response during RL rollout. Metadata is stored in extra_env_info for
    the TDC environment to score against.

    Args:
        datum_dict: Raw datum with 'messages', 'answer', 'smiles', 'label', 'task' keys.
        task_data_spec: Task data specification with prompt template etc.
        tokenizer: HuggingFace tokenizer.
        max_seq_length: Maximum sequence length.
        idx: Index of this datum in the dataset.

    Returns:
        DatumSpec with tokenized prompt and TDC metadata in extra_env_info.
    """
    messages = datum_dict["messages"]

    message_log = get_formatted_message_log(
        messages,
        tokenizer,
        task_data_spec,
        add_bos_token=True,
        add_eos_token=False,
        add_generation_prompt=True,
        tools=datum_dict.get("tools", None),
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][: min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0

    extra_env_info = {
        "ground_truth": datum_dict.get("answer", datum_dict.get("label", "")),
        "smiles": datum_dict.get("smiles", ""),
        "label": datum_dict.get("label", ""),
        "task": datum_dict.get("task", ""),
    }

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "tdc"),
    }
    return output
