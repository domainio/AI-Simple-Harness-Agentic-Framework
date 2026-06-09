from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_sdk.types import AgentResult, Message, ModelResponse

LANGFUSE_ENV = Path(__file__).resolve().parent / "observability" / ".env.langfuse"
LANGFUSE_KEYS = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_langfuse_env(env_path: Path = LANGFUSE_ENV) -> None:
    """Load local Langfuse SDK keys so get_client() can publish."""
    wanted = {k for k in LANGFUSE_KEYS if not os.environ.get(k)}
    if not wanted or not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if key in wanted and value.strip():
            os.environ[key] = _unquote(value)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _compact_message(msg: "Message", limit: int) -> dict:
    out: dict = {"role": msg.get("role")}
    if "content" in msg and msg.get("content") is not None:
        out["content"] = _clip(str(msg["content"]), limit)
    if msg.get("tool_call_id"):
        out["tool_call_id"] = msg["tool_call_id"]
    if msg.get("reasoning"):
        out["reasoning"] = _clip(str(msg["reasoning"]), limit)
    if msg.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": tc.get("id"),
                "name": (tc.get("function") or {}).get("name"),
                "arguments": _clip(str((tc.get("function") or {}).get("arguments", "")), limit),
            }
            for tc in msg["tool_calls"]
        ]
    return out


def _parse_args(raw: str, limit: int):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _clip(raw, limit)


class LangfuseTracer:
    """Langfuse adapter for the SDK tracer seam.

    Import stays optional: pass a fake client in tests, or install `langfuse`
    and configure LANGFUSE_* env vars for real traces.
    """

    def __init__(self, client=None, *, trace_name: str = "agent", flush_on_stop: bool = True, max_chars: int = 4000):
        if client is None:
            try:
                from langfuse import get_client
            except ImportError as exc:
                raise RuntimeError("Install langfuse to use LangfuseTracer: uv add langfuse") from exc
            _load_langfuse_env()
            client = get_client()
        self.client = client
        self.trace_name = trace_name
        self.flush_on_stop = flush_on_stop
        self.max_chars = max_chars
        self._run_cm = None
        self._run = None
        self._model_cms: dict[int, object] = {}
        self._models: dict[int, object] = {}
        self._tool_cm = None
        self._tool = None

    def on_run_start(self, task: str, max_steps: int) -> None:
        self._run_cm = self.client.start_as_current_observation(
            as_type="span",
            name=self.trace_name,
            input={"task": _clip(task, self.max_chars), "max_steps": max_steps},
        )
        self._run = self._run_cm.__enter__()

    def on_model_start(self, step: int, messages: list["Message"]) -> None:
        self._ensure_run()
        cm = self.client.start_as_current_observation(
            as_type="generation",
            name=f"model.step.{step}",
            input=[_compact_message(m, self.max_chars) for m in messages],
            metadata={"step": step},
        )
        self._model_cms[step] = cm
        self._models[step] = cm.__enter__()

    def on_model_end(self, step: int, response: "ModelResponse") -> None:
        gen = self._models.pop(step, None)
        if gen is not None:
            fields: dict = {
                "output": _compact_message(response.message, self.max_chars),
                "metadata": {"step": step, "finish_reason": response.finish_reason},
            }
            if response.usage is not None:
                fields["usage_details"] = response.usage
            if response.model is not None:
                fields["model"] = response.model
            gen.update(**fields)
        self._close(self._model_cms.pop(step, None))

    def on_model_message(self, step: int, msg: "Message") -> None:
        pass

    def on_tool_start(self, step: int, name: str, args: str) -> None:
        self._ensure_run()
        self._tool_cm = self.client.start_as_current_observation(
            as_type="span",
            name=f"tool.{name}",
            input=_parse_args(args, self.max_chars),
            metadata={"step": step, "tool": name},
        )
        self._tool = self._tool_cm.__enter__()

    def on_tool_end(self, step: int, name: str, result: str) -> None:
        if self._tool is not None:
            self._tool.update(output=_clip(result, self.max_chars), metadata={"step": step, "tool": name})
        self._close(self._tool_cm)
        self._tool = None
        self._tool_cm = None

    def on_stop(self, reason: str, steps: int) -> None:
        self._ensure_run()
        if self._run is not None:
            self._run.update(metadata={"reason": reason, "steps": steps})

    def on_run_end(self, result: "AgentResult") -> None:
        self._ensure_run()
        if self._run is not None:
            self._run.update(
                output=_clip(result.output, self.max_chars),
                metadata={"status": result.status, "reason": result.reason, "steps": result.steps},
            )
        self._close(self._run_cm)
        self._run = None
        self._run_cm = None
        if self.flush_on_stop:
            self.client.flush()

    def _ensure_run(self) -> None:
        if self._run_cm is None:
            self.on_run_start("", 0)

    @staticmethod
    def _close(cm) -> None:
        if cm is not None:
            cm.__exit__(None, None, None)
