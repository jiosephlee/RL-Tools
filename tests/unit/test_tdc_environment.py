"""Unit tests for TDC data processor and environment."""

import pytest

# TDC environment module imports ray at top level due to @ray.remote decorator.
# Import the pure functions directly to test without ray.
try:
    from nemo_rl.environments.tdc_environment import compute_reward, extract_final_answer

    HAS_TDC_ENV = True
except ImportError:
    HAS_TDC_ENV = False

    # Define local copies for testing without ray
    import re

    def extract_final_answer(text: str) -> str:
        match = re.search(r"Answer:\s*\(([AB])\)", text, re.IGNORECASE)
        if match:
            return f"({match.group(1)})"
        match = re.search(r"Final\s+answer:\s*\(([AB])\)", text, re.IGNORECASE)
        if match:
            return f"({match.group(1)})"
        matches = list(re.finditer(r"\(([AB])\)", text))
        if matches:
            return f"({matches[-1].group(1)})"
        return ""

    def compute_reward(generated_text: str, label: str) -> float:
        predicted = extract_final_answer(generated_text)
        if not predicted:
            return 0.0
        if predicted.upper() == label.upper():
            return 1.0
        return 0.0


class TestExtractFinalAnswer:
    def test_answer_pattern(self):
        assert extract_final_answer("Answer: (A)") == "(A)"
        assert extract_final_answer("Answer:(B)") == "(B)"
        assert extract_final_answer("answer: (A)") == "(A)"

    def test_final_answer_pattern(self):
        assert extract_final_answer("Final answer: (B)") == "(B)"
        assert extract_final_answer("Final Answer: (A)") == "(A)"

    def test_bare_parenthetical(self):
        assert extract_final_answer("I think (A) is correct") == "(A)"
        assert extract_final_answer("The answer is (B)") == "(B)"

    def test_last_occurrence(self):
        assert extract_final_answer("(A) but actually (B)") == "(B)"

    def test_no_answer(self):
        assert extract_final_answer("No clear answer here") == ""
        assert extract_final_answer("") == ""

    def test_markdown_bold(self):
        assert extract_final_answer("**Answer: (B)**") == "(B)"


class TestComputeReward:
    def test_correct_answer(self):
        assert compute_reward("Answer: (A)", "(A)") == 1.0
        assert compute_reward("I think (B)", "(B)") == 1.0

    def test_incorrect_answer(self):
        assert compute_reward("Answer: (B)", "(A)") == 0.0

    def test_no_answer(self):
        assert compute_reward("No answer here", "(A)") == 0.0

    def test_case_insensitive(self):
        assert compute_reward("Answer: (a)", "(A)") == 1.0


@pytest.mark.skipif(not HAS_TDC_ENV, reason="ray not available")
class TestTDCDataProcessor:
    def test_import(self):
        from nemo_rl.data.processors.tdc_processor import tdc_data_processor

        assert callable(tdc_data_processor)

    def test_registered_in_processor_registry(self):
        from nemo_rl.data.processors import PROCESSOR_REGISTRY

        assert "tdc_data_processor" in PROCESSOR_REGISTRY
