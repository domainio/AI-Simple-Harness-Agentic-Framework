from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_sdk.types import Message, ModelResponse, RunConfig


WIRE_KEYS = ("role", "content", "tool_calls", "tool_call_id")


class BaseChatModel(ABC):
    """Provider template. Subclasses isolate wire shape, provider call, and decode logic."""

    def invoke(self, messages: "list[Message]", cfg: "RunConfig") -> "ModelResponse":
        return self._decode(self._call(self._encode(messages), cfg))

    @abstractmethod
    def _encode(self, messages: "list[Message]") -> object: ...

    @abstractmethod
    def _call(self, wire: object, cfg: "RunConfig") -> object: ...

    @abstractmethod
    def _decode(self, raw: object) -> "ModelResponse": ...
