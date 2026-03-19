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

"""Chat protocol abstractions for format-specific tool-call parsing and feedback rendering.

Ported from OpenRLHF-Tools with adaptations for NeMo RL's EnvironmentInterface.
Provides:
- ChatProtocol ABC: Abstract interface for model-specific tool-call handling
- GPTOSSProtocol: Harmony token-ID parser for GPT-OSS models
- InternS1Protocol: JSON tool calling with action markers
- Qwen3Protocol: JSON tool calling with <tool_call> tags
- Qwen3CoderProtocol: XML tool calling with <function=name> syntax
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)

_VALID_JSON_ESC = set(['"', "\\", "/", "b", "f", "n", "r", "t", "u"])


def _repair_invalid_json_escapes(s: str) -> str:
    r"""Repair invalid escape sequences inside JSON strings (e.g. \C -> \\C).

    Needed for SMILES strings that contain backslash characters which are not
    valid JSON escapes.
    """
    out: list[str] = []
    in_str = False
    i = 0
    while i < len(s):
        c = s[i]
        if not in_str:
            if c == '"':
                in_str = True
            out.append(c)
            i += 1
            continue
        if c == '"':
            in_str = False
            out.append(c)
            i += 1
            continue
        if c == "\\":
            if i + 1 >= len(s):
                out.append("\\\\")
                i += 1
                continue
            nxt = s[i + 1]
            if nxt in _VALID_JSON_ESC:
                out.append("\\")
                out.append(nxt)
                i += 2
            else:
                out.append("\\\\")
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


class ChatProtocol(ABC):
    """Abstract protocol for format-specific parsing and feedback."""

    @abstractmethod
    def parse_assistant_text(self, text: str) -> dict[str, Any]:
        """Parse assistant text into structured action dict.

        Args:
            text: Raw assistant output text.

        Returns:
            Dictionary with:
                - content: str - Final answer text
                - tool_calls: list[dict] - List of tool calls with 'name' and 'arguments'
        """

    @abstractmethod
    def render_tool_feedback(self, tool_results: list[dict[str, str]]) -> str:
        """Render bridge text between assistant generation and next turn.

        Args:
            tool_results: List of dicts with "name" and "content" keys.

        Returns:
            Bridge text for concatenation.
        """

    def render_tool_feedback_token_ids(
        self,
        tool_results: list[dict[str, str]],
        tokenizer: Any = None,
    ) -> list[int] | None:
        """Return canonical token IDs for tool feedback, or None to fall back to text tokenization.

        When special tokens in the feedback text may not round-trip correctly
        through text → tokenize (e.g. harmony <|start|>, <|end|>), subclasses
        should override this to produce exact token IDs.

        Args:
            tool_results: List of dicts with "name" and "content" keys.
            tokenizer: HuggingFace tokenizer (needed to encode text segments).

        Returns:
            List of token IDs, or None if text tokenization is acceptable.
        """
        return None

    @property
    def stop_strings(self) -> list[str]:
        """Stop strings that signal end of assistant generation for this protocol."""
        return []


class InternS1Protocol(ChatProtocol):
    """Intern-S1-mini JSON tool calling format protocol.

    Tool call format:
        <|action_start|><|plugin|>
        {"name": "tool_name", "parameters": {"key": "value"}}
        <|action_end|>

    Observation format:
        <|im_start|>environment name=<|plugin|>
        {tool_result}
        <|im_end|>
    """

    _START_RE = re.compile(r"<\|action_start\|>\s*<\|plugin\|>")
    _END_RE = re.compile(r"<\|action_end\|>")

    def parse_assistant_text(self, text: str) -> dict[str, Any]:
        """Parse Intern-S1 tool call blocks from assistant output."""
        tool_calls: list[dict[str, Any]] = []
        content_parts: list[str] = []

        pos = 0
        while True:
            start_m = self._START_RE.search(text, pos)
            if start_m is None:
                content_parts.append(text[pos:])
                break

            content_parts.append(text[pos : start_m.start()])

            end_m = self._END_RE.search(text, start_m.end())
            if end_m is None:
                content_parts.append(text[start_m.start() :])
                break

            action = text[start_m.end() : end_m.start()].strip()
            j = action.find("{")
            if j != -1:
                action = action[j:]

            try:
                action_dict = json.loads(action)
            except json.JSONDecodeError as ex:
                if "Invalid \\escape" in str(ex):
                    try:
                        action_dict = json.loads(_repair_invalid_json_escapes(action))
                    except Exception:
                        content_parts.append(text[start_m.start() : end_m.end()])
                        pos = end_m.end()
                        continue
                else:
                    content_parts.append(text[start_m.start() : end_m.end()])
                    pos = end_m.end()
                    continue
            except Exception:
                content_parts.append(text[start_m.start() : end_m.end()])
                pos = end_m.end()
                continue

            name = action_dict.get("name")
            args = action_dict.get("parameters", action_dict.get("arguments", {}))
            if name:
                tool_calls.append({"name": name, "arguments": args})

            pos = end_m.end()

        content = "".join(content_parts).strip()
        if tool_calls:
            return {"content": content, "tool_calls": tool_calls}
        return {"content": content or text, "tool_calls": []}

    def render_tool_feedback(self, tool_results: list[dict[str, str]]) -> str:
        """Intern-S1 bridge: close assistant turn, render tool responses, open next turn."""
        feedback = "<|im_end|>\n"
        for tr in tool_results:
            feedback += f"<|im_start|>environment name=<|plugin|>\n\n{tr['content']}<|im_end|>\n"
        feedback += "<|im_start|>assistant\n\n<think>\n"
        return feedback

    @property
    def stop_strings(self) -> list[str]:
        return ["<|action_end|>", "<|im_end|>"]


class Qwen3Protocol(InternS1Protocol):
    """Qwen3 JSON tool calling format protocol.

    Tool call format:
        <tool_call>
        {"name": "tool_name", "arguments": {"key": "value"}}
        </tool_call>
    """

    _START_RE = re.compile(r"<tool_call>")
    _END_RE = re.compile(r"</tool_call>")

    def render_tool_feedback(self, tool_results: list[dict[str, str]]) -> str:
        """Qwen3 bridge: close assistant turn + tool responses + open next turn."""
        feedback = "\n<|im_end|>\n"
        for tr in tool_results:
            feedback += f"<|im_start|>tool\n<tool_response>\n{tr['content']}\n</tool_response>\n<|im_end|>\n"
        feedback += "<|im_start|>assistant\n"
        return feedback

    @property
    def stop_strings(self) -> list[str]:
        return ["</tool_call>", "<|im_end|>"]


class Qwen3CoderProtocol(Qwen3Protocol):
    """Qwen3.5 Coder XML tool calling format protocol.

    Tool call format:
        <tool_call>
        <function=tool_name>
        <parameter=key>value</parameter>
        </function>
        </tool_call>
    """

    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
    _FUNCTION_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
    _PARAMETER_RE = re.compile(
        r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
        re.DOTALL,
    )

    _START_RE = re.compile(r"<tool_call>")
    _END_RE = re.compile(r"</tool_call>")

    def parse_assistant_text(self, text: str) -> dict[str, Any]:
        """Parse Qwen3.5 Coder XML tool call format."""
        if "<function=" not in text:
            return {"content": text, "tool_calls": []}

        try:
            function_calls = self._get_function_calls(text)
            if not function_calls:
                return {"content": text, "tool_calls": []}

            tool_calls = []
            for fc_str in function_calls:
                tc = self._parse_xml_function_call(fc_str)
                if tc:
                    tool_calls.append(tc)

            content_idx = text.find("<tool_call>")
            if content_idx < 0:
                content_idx = text.find("<function=")
            content = text[:content_idx].strip() if content_idx > 0 else ""

            return {"content": content, "tool_calls": tool_calls}
        except Exception as e:
            logger.warning("Qwen3Coder parse error: %s", e)
            return {"content": text, "tool_calls": []}

    def _get_function_calls(self, model_output: str) -> list[str]:
        """Extract raw function call strings from model output."""
        matched_ranges = self._TOOL_CALL_RE.findall(model_output)
        raw_tool_calls = [m[0] if m[0] else m[1] for m in matched_ranges]

        if not raw_tool_calls:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tc in raw_tool_calls:
            raw_function_calls.extend(self._FUNCTION_RE.findall(tc))

        return [m[0] if m[0] else m[1] for m in raw_function_calls]

    def _parse_xml_function_call(self, function_call_str: str) -> dict[str, Any] | None:
        """Parse a single <function=name>..params..</function> block."""
        try:
            end_idx = function_call_str.index(">")
        except ValueError:
            return None

        function_name = function_call_str[:end_idx].strip()
        parameters_text = function_call_str[end_idx + 1 :]

        param_dict: dict[str, Any] = {}
        for match_text in self._PARAMETER_RE.findall(parameters_text):
            try:
                idx = match_text.index(">")
            except ValueError:
                continue
            param_name = match_text[:idx].strip()
            param_value = match_text[idx + 1 :]
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]
            param_dict[param_name] = self._coerce_param_value(param_value)

        if not function_name:
            return None
        return {"name": function_name, "arguments": param_dict}

    @staticmethod
    def _coerce_param_value(value: str) -> Any:
        """Best-effort type coercion for XML parameter values."""
        stripped = value.strip()
        if stripped.lower() == "null":
            return None
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass
        if stripped.lower() in ("true", "false"):
            return stripped.lower() == "true"
        try:
            return int(stripped)
        except ValueError:
            pass
        try:
            float_val = float(stripped)
            return int(float_val) if float_val == int(float_val) else float_val
        except ValueError:
            pass
        return value


class GPTOSSProtocol(ChatProtocol):
    """GPT-OSS Harmony protocol with regex-based text parsing.

    Harmony message format (tool call):
        <|start|>assistant<|channel|>analysis<|message|>reasoning...<|end|>
        <|start|>assistant<|channel|>commentary to=functions.tool_name
        <|constrain|>json<|message|>{"arg": "val"}<|call|>

    Tool feedback (function -> assistant):
        <|start|>functions.tool_name to=assistant<|channel|>commentary<|message|>result<|end|>

    Stop tokens: <|return|> and <|call|>.
    """

    _RE_TOOL_CALL = re.compile(
        r"(?:<\|channel\|>[^<]*?)?"
        r"to=functions\.(\S+?)"
        r"(?:\s*<\|(?:channel|constrain)\|>[^<{]*)*"
        r"\s*<\|message\|>(.*?)"
        r"(?:<\|call\|>|<\|end\|>|$)",
        re.DOTALL,
    )

    def parse_assistant_text(self, text: str) -> dict[str, Any]:
        """Parse GPT-OSS assistant output using regex on decoded text.

        Args:
            text: Decoded assistant text (with special tokens visible).

        Returns:
            Dict with content, tool_calls, and parse metadata.
        """
        tool_calls: list[dict[str, Any]] = []

        for m in self._RE_TOOL_CALL.finditer(text):
            name = m.group(1)
            args_text = m.group(2).strip()
            try:
                args: Any = json.loads(args_text)
            except json.JSONDecodeError:
                try:
                    args = json.loads(_repair_invalid_json_escapes(args_text))
                except Exception:
                    args = args_text
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    try:
                        args = json.loads(_repair_invalid_json_escapes(args))
                    except Exception:
                        args = {"raw": args}
            if not isinstance(args, dict):
                args = {"raw": args}
            tool_calls.append({"name": name, "arguments": args})

        attempted_call = "<|call|>" in text or "to=functions" in text
        parse_failed = attempted_call and len(tool_calls) == 0

        return {
            "content": text,
            "tool_calls": tool_calls,
            "parse_failed": parse_failed,
        }

    def render_tool_feedback(self, tool_results: list[dict[str, str]]) -> str:
        """Build bridge text: tool responses + generation prompt."""
        feedback = ""
        for tr in tool_results:
            tool_name = tr["name"]
            tool_content = tr["content"]
            feedback += (
                f"<|start|>functions.{tool_name} to=assistant"
                f"<|channel|>commentary<|message|>{tool_content}<|end|>"
            )
        feedback += "<|start|>assistant"
        return feedback

    def render_tool_feedback_token_ids(
        self,
        tool_results: list[dict[str, str]],
        tokenizer: Any = None,
    ) -> list[int] | None:
        """Canonical token IDs for GPT-OSS tool feedback via harmony encoder.

        Uses ``openai_harmony``'s encoder to produce the exact token IDs the
        model was trained with, avoiding the lossy text → HF tokenize
        round-trip for special tokens like ``<|start|>``, ``<|end|>``.
        """
        if tokenizer is None:
            return None
        try:
            from openai_harmony import (
                Author,
                HarmonyEncodingName,
                Message,
                Role,
                load_harmony_encoding,
            )
        except ImportError:
            logger.warning("openai_harmony not available; falling back to text tokenization")
            return None

        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        token_ids: list[int] = []
        for tr in tool_results:
            msg = (
                Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{tr['name']}"),
                    tr["content"],
                )
                .with_channel("commentary")
                .with_recipient("assistant")
            )
            token_ids.extend(encoding.render(msg))

        # Append <|start|>assistant generation prompt
        assistant_header_ids = tokenizer.encode("<|start|>assistant", add_special_tokens=False)
        token_ids.extend(assistant_header_ids)
        return token_ids

    @property
    def stop_strings(self) -> list[str]:
        return ["<|return|>", "<|call|>"]


PROTOCOL_REGISTRY: dict[str, type[ChatProtocol]] = {
    "gptoss": GPTOSSProtocol,
    "gpt_oss": GPTOSSProtocol,
    "intern_s1": InternS1Protocol,
    "qwen3": Qwen3Protocol,
    "qwen3_coder": Qwen3CoderProtocol,
}


def get_protocol(name: str) -> ChatProtocol:
    """Instantiate a ChatProtocol by name.

    Args:
        name: Protocol name (e.g., "gptoss", "intern_s1", "qwen3", "qwen3_coder").

    Returns:
        ChatProtocol instance.

    Raises:
        ValueError: If protocol name is not recognized.
    """
    if name not in PROTOCOL_REGISTRY:
        raise ValueError(f"Unknown chat protocol: {name!r}. Available: {list(PROTOCOL_REGISTRY.keys())}")
    return PROTOCOL_REGISTRY[name]()


__all__ = [
    "ChatProtocol",
    "GPTOSSProtocol",
    "InternS1Protocol",
    "Qwen3Protocol",
    "Qwen3CoderProtocol",
    "get_protocol",
    "PROTOCOL_REGISTRY",
]
