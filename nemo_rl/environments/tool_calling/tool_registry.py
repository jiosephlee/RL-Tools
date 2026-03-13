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

"""Tool registry for tool-calling environments.

Manages tool schemas (OpenAI format) and their corresponding Python callables.
"""

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry mapping tool names to schemas and callable implementations.

    Supports loading tool schemas from JSON files (OpenAI function-calling format)
    and registering Python callables by name.
    """

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}
        self._callables: dict[str, Callable[..., Any]] = {}

    def load_schemas_from_file(self, path: str) -> None:
        """Load tool schemas from a JSON file.

        The file should contain a list of tool definitions in OpenAI format:
        [{"type": "function", "function": {"name": "...", "parameters": {...}}}]

        Args:
            path: Path to the JSON file containing tool schemas.
        """
        with open(path, "r") as f:
            tools = json.load(f)

        for tool in tools:
            if isinstance(tool, dict):
                func_def = tool.get("function", tool)
                name = func_def.get("name")
                if name:
                    self._schemas[name] = func_def

        logger.info("Loaded %d tool schemas from %s", len(tools), path)

    def register_callable(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a Python callable for a tool.

        Args:
            name: Tool name (must match a loaded schema name).
            fn: Callable implementing the tool.
        """
        self._callables[name] = fn

    def register_callables(self, callables: dict[str, Callable[..., Any]]) -> None:
        """Register multiple tool callables at once.

        Args:
            callables: Dict mapping tool names to callables.
        """
        self._callables.update(callables)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with given arguments.

        Args:
            name: Tool name.
            arguments: Tool arguments dict.

        Returns:
            JSON string with execution result or error.
        """
        if name not in self._callables:
            return json.dumps({
                "error": f"Unknown tool: {name}",
                "available_tools": list(self._callables.keys()),
            })

        # Handle "raw" argument from failed JSON parse
        if "raw" in arguments and len(arguments) == 1:
            raw_val = arguments["raw"]
            if isinstance(raw_val, str):
                try:
                    parsed = json.loads(raw_val)
                    if isinstance(parsed, dict):
                        arguments = parsed
                except Exception:
                    pass

        try:
            result = self._callables[name](**arguments)
            return json.dumps({"result": result, "function_name": name, "arguments": arguments})
        except TypeError as e:
            # Try fallback with just the primary argument (e.g., "smiles")
            error_str = str(e)
            if "got an unexpected keyword argument" in error_str:
                for key in ("smiles", "query_smiles"):
                    if key in arguments:
                        try:
                            result = self._callables[name](**{key: arguments[key]})
                            return json.dumps({"result": result, "function_name": name, "arguments": {key: arguments[key]}})
                        except Exception:
                            pass
            return json.dumps({"error": error_str, "function_name": name, "arguments": arguments})
        except Exception as e:
            return json.dumps({"error": str(e), "function_name": name, "arguments": arguments})

    @property
    def tool_names(self) -> list[str]:
        """List of registered tool names with callables."""
        return list(self._callables.keys())

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """List of all loaded tool schemas in OpenAI format."""
        return [{"type": "function", "function": schema} for schema in self._schemas.values()]

    def has_tool(self, name: str) -> bool:
        """Check if a tool callable is registered."""
        return name in self._callables
