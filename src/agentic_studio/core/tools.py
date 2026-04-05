"""Tool definitions and registry for the agentic orchestrator.

Each tool is a callable that the LLM can invoke based on its reasoning.
Tools have JSON schemas for structured tool calling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ToolParameter:
    """A parameter for a tool."""
    name: str
    description: str
    type: str  # "string", "number", "array", "object", etc.
    required: bool = True
    enum: list[str] | None = None
    items_type: str | None = None  # For arrays


@dataclass(slots=True)
class ToolSchema:
    """JSON schema for a tool."""
    name: str
    description: str
    parameters: list[ToolParameter]


class Tool:
    """Wrapper for an agent tool."""
    def __init__(
        self,
        name: str,
        description: str,
        schema: ToolSchema,
        func: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.schema = schema
        self.func = func

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        return self.func(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert tool to dictionary for API calls."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                        **({"enum": param.enum} if param.enum else {}),
                        **({"items": {"type": param.items_type}} if param.items_type else {}),
                    }
                    for param in self.schema.parameters
                },
                "required": [p.name for p in self.schema.parameters if p.required],
            },
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all tools."""
        return list(self.tools.values())

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        return tool(**kwargs)


# Global tool registry
_registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    parameters: list[ToolParameter],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a tool."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        schema = ToolSchema(name=name, description=description, parameters=parameters)
        tool = Tool(name=name, description=description, schema=schema, func=func)
        _registry.register(tool)
        return func
    return decorator


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry
