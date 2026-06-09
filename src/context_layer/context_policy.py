from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ContextType(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DOC = "doc"
    MEMORY = "memory"
    SUMMARY = "summary"


@dataclass(frozen=True)
class TypePolicy:
    base_priority: int
    pinned: bool = False
    truncatable: bool = False


DEFAULT_POLICY: dict[ContextType, TypePolicy] = {
    ContextType.SYSTEM: TypePolicy(base_priority=1000, pinned=True),
    ContextType.USER: TypePolicy(base_priority=900),
    ContextType.ASSISTANT: TypePolicy(base_priority=500),
    ContextType.TOOL: TypePolicy(base_priority=500, truncatable=True),
    ContextType.DOC: TypePolicy(base_priority=400, truncatable=True),
    ContextType.MEMORY: TypePolicy(base_priority=600),
}
