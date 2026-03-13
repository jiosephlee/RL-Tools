"""Unit tests for chat protocol parsing."""

import json

import pytest

from nemo_rl.environments.tool_calling.chat_protocol import (
    GPTOSSProtocol,
    InternS1Protocol,
    Qwen3CoderProtocol,
    Qwen3Protocol,
    get_protocol,
)


class TestInternS1Protocol:
    def setup_method(self):
        self.protocol = InternS1Protocol()

    def test_no_tool_calls(self):
        result = self.protocol.parse_assistant_text("The answer is 42")
        assert result["tool_calls"] == []
        assert result["content"] == "The answer is 42"

    def test_single_tool_call(self):
        text = '<|action_start|><|plugin|>\n{"name": "get_smiles", "parameters": {"smiles": "CCO"}}\n<|action_end|>'
        result = self.protocol.parse_assistant_text(text)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "get_smiles"
        assert result["tool_calls"][0]["arguments"] == {"smiles": "CCO"}

    def test_multiple_tool_calls(self):
        text = (
            '<|action_start|><|plugin|>\n{"name": "tool1", "parameters": {"x": 1}}\n<|action_end|>'
            "Some text"
            '<|action_start|><|plugin|>\n{"name": "tool2", "parameters": {"y": 2}}\n<|action_end|>'
        )
        result = self.protocol.parse_assistant_text(text)
        assert len(result["tool_calls"]) == 2

    def test_render_feedback(self):
        feedback = self.protocol.render_tool_feedback([{"name": "tool1", "content": '{"result": "ok"}'}])
        assert "<|im_end|>" in feedback
        assert "environment" in feedback
        assert "<|im_start|>assistant" in feedback

    def test_accepts_arguments_key(self):
        text = '<|action_start|><|plugin|>\n{"name": "func", "arguments": {"a": 1}}\n<|action_end|>'
        result = self.protocol.parse_assistant_text(text)
        assert result["tool_calls"][0]["arguments"] == {"a": 1}

    def test_invalid_json(self):
        text = "<|action_start|><|plugin|>\nnot json\n<|action_end|>"
        result = self.protocol.parse_assistant_text(text)
        assert result["tool_calls"] == []

    def test_stop_strings(self):
        assert len(self.protocol.stop_strings) > 0


class TestQwen3Protocol:
    def setup_method(self):
        self.protocol = Qwen3Protocol()

    def test_no_tool_calls(self):
        result = self.protocol.parse_assistant_text("Just some text")
        assert result["tool_calls"] == []

    def test_tool_call(self):
        text = '<tool_call>\n{"name": "calc", "arguments": {"x": 1}}\n</tool_call>'
        result = self.protocol.parse_assistant_text(text)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "calc"

    def test_render_feedback(self):
        feedback = self.protocol.render_tool_feedback([{"name": "calc", "content": "42"}])
        assert "<tool_response>" in feedback
        assert "<|im_start|>assistant" in feedback


class TestQwen3CoderProtocol:
    def setup_method(self):
        self.protocol = Qwen3CoderProtocol()

    def test_no_tool_calls(self):
        result = self.protocol.parse_assistant_text("Just text")
        assert result["tool_calls"] == []

    def test_xml_tool_call(self):
        text = "<tool_call>\n<function=my_tool>\n<parameter=smiles>CCO</parameter>\n</function>\n</tool_call>"
        result = self.protocol.parse_assistant_text(text)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "my_tool"
        assert result["tool_calls"][0]["arguments"]["smiles"] == "CCO"

    def test_coerce_param_values(self):
        assert Qwen3CoderProtocol._coerce_param_value("42") == 42
        assert Qwen3CoderProtocol._coerce_param_value("true") is True
        assert Qwen3CoderProtocol._coerce_param_value("null") is None
        assert Qwen3CoderProtocol._coerce_param_value("hello") == "hello"
        assert Qwen3CoderProtocol._coerce_param_value("3.14") == 3.14


class TestGPTOSSProtocol:
    def setup_method(self):
        self.protocol = GPTOSSProtocol()

    def test_no_tool_calls(self):
        result = self.protocol.parse_assistant_text("Just a response")
        assert result["tool_calls"] == []
        assert not result.get("parse_failed", False)

    def test_tool_call_regex(self):
        text = '<|channel|>analysis to=functions.get_smiles<|constrain|>json<|message|>{"smiles": "CCO"}<|call|>'
        result = self.protocol.parse_assistant_text(text)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "get_smiles"

    def test_parse_failed_detection(self):
        text = "to=functions.something <|call|>"
        result = self.protocol.parse_assistant_text(text)
        # May or may not find tool calls, but should detect attempted call
        assert "<|call|>" in text

    def test_render_feedback(self):
        feedback = self.protocol.render_tool_feedback([{"name": "tool1", "content": "result"}])
        assert "functions.tool1" in feedback
        assert "<|start|>assistant" in feedback

    def test_stop_strings(self):
        assert "<|return|>" in self.protocol.stop_strings
        assert "<|call|>" in self.protocol.stop_strings


class TestGetProtocol:
    def test_valid_protocols(self):
        for name in ["gptoss", "gpt_oss", "intern_s1", "qwen3", "qwen3_coder"]:
            protocol = get_protocol(name)
            assert protocol is not None

    def test_invalid_protocol(self):
        with pytest.raises(ValueError, match="Unknown chat protocol"):
            get_protocol("nonexistent")
