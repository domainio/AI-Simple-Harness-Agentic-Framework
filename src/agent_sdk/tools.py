from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pydantic import BaseModel

from agent_sdk.protocols import Tool
from agent_sdk.types import RunConfig


@dataclass
class FunctionTool:
    name: str
    description: str
    args_schema: type[BaseModel]
    fn: Callable[..., object]

    def invoke(self, args: dict, cfg: RunConfig) -> str:
        try:
            validated = self.args_schema(**args)
            return str(self.fn(**validated.model_dump()))
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"


def tool(args: type[BaseModel]) -> Callable[[Callable[..., object]], FunctionTool]:
    """Wrap a plain function as a validated tool."""

    def decorate(fn: Callable[..., object]) -> FunctionTool:
        return FunctionTool(
            name=fn.__name__,
            description=(fn.__doc__ or "").strip(),
            args_schema=args,
            fn=fn,
        )

    return decorate


def to_openai_tool(t: Tool) -> dict:
    params = t.args_schema.model_json_schema()
    params["additionalProperties"] = False
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": params,
        },
    }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def openai_schemas(self) -> list[dict]:
        return [to_openai_tool(t) for t in self._tools.values()]
