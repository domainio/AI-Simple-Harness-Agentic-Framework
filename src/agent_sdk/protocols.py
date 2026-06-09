from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from agent_sdk.types import Message, ModelResponse, RunConfig


@runtime_checkable
class ChatModel(Protocol):
    def invoke(self, messages: list["Message"], cfg: "RunConfig") -> "ModelResponse": ...


class Tool(Protocol):
    name: str
    description: str
    args_schema: "type[BaseModel]"

    def invoke(self, args: dict, cfg: "RunConfig") -> str: ...


@runtime_checkable
class Tracer(Protocol):
    def on_model_message(self, step: int, msg: "Message") -> None: ...

    def on_tool_start(self, step: int, name: str, args: str) -> None: ...

    def on_tool_end(self, step: int, name: str, result: str) -> None: ...

    def on_stop(self, reason: str, steps: int) -> None: ...
