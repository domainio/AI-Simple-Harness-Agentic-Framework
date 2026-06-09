from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_sdk.types import Message


class NoopTracer:
    """Silent default: SDK does no output unless caller asks."""

    def on_model_message(self, step: int, msg: "Message") -> None: ...

    def on_tool_start(self, step: int, name: str, args: str) -> None: ...

    def on_tool_end(self, step: int, name: str, result: str) -> None: ...

    def on_stop(self, reason: str, steps: int) -> None: ...


def _clip(s: str, n: int = 500) -> str:
    return s if len(s) <= n else s[:n] + "..."


class StdoutTracer:
    """Human-readable trace of model decisions and tool I/O."""

    def on_model_message(self, step: int, msg: "Message") -> None:
        reasoning = msg.get("reasoning")
        if reasoning:
            print(f"[step {step}] reasoning: {_clip(reasoning)}")
        content = msg.get("content")
        if content:
            print(f"[step {step}] model: {_clip(content)}")
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            print(f"[step {step}] model wants tool: {fn['name']} {fn['arguments']}")

    def on_tool_start(self, step: int, name: str, args: str) -> None:
        print(f"[step {step}] tool -> {name} {args}")

    def on_tool_end(self, step: int, name: str, result: str) -> None:
        print(f"[step {step}] tool <- {name}: {_clip(result)}")

    def on_stop(self, reason: str, steps: int) -> None:
        print(f"[stop] reason={reason} steps={steps}")
