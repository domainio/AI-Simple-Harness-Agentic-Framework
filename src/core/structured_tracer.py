from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_sdk.types import Message


class StructuredTracer:
    """Emit agent and context events as one JSON record per line."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self._tool_started_at: float | None = None

    def _emit(self, kind: str, **fields) -> None:
        self.stream.write(json.dumps({"ts": time.time(), "kind": kind, **fields}) + "\n")
        self.stream.flush()

    def on_model_message(self, step: int, msg: "Message") -> None:
        self._emit(
            "model",
            step=step,
            tool_calls=len(msg.get("tool_calls") or []),
            has_content=bool(msg.get("content")),
            has_reasoning=bool(msg.get("reasoning")),
        )

    def on_tool_start(self, step: int, name: str, args: str) -> None:
        self._tool_started_at = time.perf_counter()
        self._emit("tool_start", step=step, name=name)

    def on_tool_end(self, step: int, name: str, result: str) -> None:
        now = time.perf_counter()
        started_at = self._tool_started_at if self._tool_started_at is not None else now
        self._emit(
            "tool_end",
            step=step,
            name=name,
            result_len=len(result),
            latency_ms=round((now - started_at) * 1000, 2),
        )

    def on_stop(self, reason: str, steps: int) -> None:
        self._emit("stop", reason=reason, steps=steps)

    def on_assembly(self, kept: int, evicted: int, truncated: int, tokens: int, budget: int) -> None:
        self._emit(
            "assembly",
            kept=kept,
            evicted=evicted,
            truncated=truncated,
            tokens=tokens,
            budget=budget,
        )


class MultiTracer:
    """Fan out tracer events, including optional context-assembly telemetry."""

    def __init__(self, tracers: list):
        self.tracers = tracers

    def _notify(self, event: str, *args, **kwargs):
        for tracer in self.tracers:
            fn = getattr(tracer, event, None)
            if fn is not None:
                fn(*args, **kwargs)

    def on_run_start(self, task, max_steps):
        self._notify("on_run_start", task, max_steps)

    def on_run_end(self, result):
        self._notify("on_run_end", result)

    def on_model_start(self, step, messages):
        self._notify("on_model_start", step, messages)

    def on_model_end(self, step, response):
        self._notify("on_model_end", step, response)

    def on_model_message(self, step, msg):
        self._notify("on_model_message", step, msg)

    def on_tool_start(self, step, name, args):
        self._notify("on_tool_start", step, name, args)

    def on_tool_end(self, step, name, result):
        self._notify("on_tool_end", step, name, result)

    def on_stop(self, reason, steps):
        self._notify("on_stop", reason, steps)

    def on_assembly(self, **kwargs):
        self._notify("on_assembly", **kwargs)
