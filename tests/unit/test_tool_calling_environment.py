"""Unit tests for tool-calling environment and tool registry."""

import json

import pytest

from nemo_rl.environments.tool_calling.tool_registry import ToolRegistry

# Environment imports require ray
try:
    from nemo_rl.environments.tool_calling import ToolCallingEnvironment
    from nemo_rl.environments.utils import ENV_REGISTRY

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()

    def test_register_and_execute(self):
        self.registry.register_callable("add", lambda x, y: x + y)
        result = json.loads(self.registry.execute("add", {"x": 1, "y": 2}))
        assert result["result"] == 3

    def test_unknown_tool(self):
        result = json.loads(self.registry.execute("nonexistent", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execution_error(self):
        def failing_tool(**kwargs):
            raise ValueError("intentional error")

        self.registry.register_callable("fail", failing_tool)
        result = json.loads(self.registry.execute("fail", {}))
        assert "error" in result
        assert "intentional error" in result["error"]

    def test_register_callables_batch(self):
        self.registry.register_callables({"a": lambda: 1, "b": lambda: 2})
        assert self.registry.has_tool("a")
        assert self.registry.has_tool("b")

    def test_tool_names(self):
        self.registry.register_callable("tool1", lambda: None)
        self.registry.register_callable("tool2", lambda: None)
        assert set(self.registry.tool_names) == {"tool1", "tool2"}

    def test_raw_argument_handling(self):
        self.registry.register_callable("echo", lambda msg: msg)
        result = json.loads(self.registry.execute("echo", {"raw": '{"msg": "hello"}'}))
        assert result["result"] == "hello"

    def test_unexpected_kwarg_fallback(self):
        def smiles_only(smiles: str) -> str:
            return f"processed:{smiles}"

        self.registry.register_callable("proc", smiles_only)
        result = json.loads(self.registry.execute("proc", {"smiles": "CCO", "extra": "val"}))
        assert result["result"] == "processed:CCO"

    def test_load_schemas_from_file(self, tmp_path):
        schema_file = tmp_path / "tools.json"
        schemas = [
            {"type": "function", "function": {"name": "tool1", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "tool2", "parameters": {"type": "object"}}},
        ]
        schema_file.write_text(json.dumps(schemas))

        self.registry.load_schemas_from_file(str(schema_file))
        assert len(self.registry.schemas) == 2


@pytest.mark.skipif(not HAS_RAY, reason="ray not available")
class TestToolCallingEnvironmentImports:
    def test_imports(self):
        from nemo_rl.environments.tool_calling import (
            ChatProtocol,
            ToolCallingEnvironment,
            ToolRegistry,
        )

        assert ChatProtocol is not None
        assert ToolCallingEnvironment is not None
        assert ToolRegistry is not None

    def test_registered_in_env_registry(self):
        assert "tool_calling" in ENV_REGISTRY
        assert "tdc" in ENV_REGISTRY
