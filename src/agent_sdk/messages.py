from __future__ import annotations

from agent_sdk.types import Message


def sys(content: str) -> Message:
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    return {"role": "user", "content": content}


def tool_msg(tool_call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def identity(history: list[Message]) -> list[Message]:
    """Default render seam: pass transcript through unchanged."""
    return history
