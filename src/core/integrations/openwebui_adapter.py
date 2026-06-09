"""Pure adapter glue: run the agent SDK from an OpenWebUI chat body.

No OpenWebUI imports, no network at import time. The SDK is reused as-is; multi-turn
memory is injected through the existing `Agent.render` hook (`agent_sdk/agent.py`).
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator

from agent_sdk.agent import Agent
from agent_sdk.builtin_tools import read_file, run_command, write_file
from agent_sdk.openai_model import OpenAIChat
from agent_sdk.tools import ToolRegistry
from agent_sdk.tracer import NoopTracer
from agent_sdk.types import Message, RunConfig


def split_messages(
    messages: list[dict],
    default_system: str,
    max_prior_chars: int = 12000,
) -> tuple[str, list[Message], str]:
    """Split an OpenWebUI message list into (system, prior, task).

    - system: first ``role=="system"`` content, else ``default_system``.
    - task:   last ``role=="user"`` content (raises ValueError if none).
    - prior:  turns before the task, sanitized to a valid OpenAI sequence — only
      user/assistant with non-empty content, rebuilt as plain {role, content} dicts
      (drops tool_calls and role=="tool"), then capped to the most recent
      ``max_prior_chars`` of content, oldest whole turns dropped first.
    """
    system = next(
        (m["content"] for m in messages if m.get("role") == "system" and m.get("content")),
        default_system,
    )

    task_index = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
        None,
    )
    if task_index is None:
        raise ValueError("no user message to use as task")
    task = messages[task_index]["content"]

    prior: list[Message] = [
        {"role": m["role"], "content": m["content"]}
        for m in messages[:task_index]
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    return system, _cap_prior(prior, max_prior_chars), task


def _cap_prior(prior: list[Message], max_chars: int) -> list[Message]:
    kept: list[Message] = []
    total = 0
    for m in reversed(prior):
        total += len(m["content"])
        if total > max_chars:
            break
        kept.append(m)
    kept.reverse()
    return kept


def history_render(prior: list[Message]) -> Callable[[list[Message]], list[Message]]:
    """Render hook that splices ``prior`` between the system message and the live turn.

    ``history[0]`` is the system message; ``history[1:]`` is the live task + in-flight
    tool loop. Injecting between them adds memory every step without evicting the live
    tool turn. ``prior`` is pre-sanitized so the emitted sequence stays API-valid.
    """

    def render(history: list[Message]) -> list[Message]:
        return [history[0], *prior, *history[1:]]

    return render


def run_chat(
    messages: list[dict],
    *,
    model=None,
    model_name: str = "gpt-4o-mini",
    system: str,
    max_steps: int,
    enable_tools: bool,
) -> str:
    """Run one agent turn over an OpenWebUI message list, return the final answer."""
    _, prior, task = split_messages(messages, system)
    registry = ToolRegistry([read_file, write_file, run_command] if enable_tools else [])
    if model is None:
        model = OpenAIChat(model=model_name, tools=registry.openai_schemas() or None)
    agent = Agent(model, registry, system, render=history_render(prior))
    result = agent.invoke(task, RunConfig(max_steps=max_steps))
    return result.output or f"[no output: {result.reason}]"


def _clip(s: str, n: int = 300) -> str:
    return s if len(s) <= n else s[:n] + "…"


class _QueueTracer(NoopTracer):
    """Pushes per-tool-step lines onto a queue as the agent loop runs."""

    def __init__(self, q: "queue.Queue[str | None]"):
        self.q = q

    def on_tool_start(self, step: int, name: str, args: str) -> None:
        self.q.put(f"⚙ step {step} → {name} {args}")

    def on_tool_end(self, step: int, name: str, result: str) -> None:
        self.q.put(f"⚙ step {step} ← {_clip(result)}")


def stream_chat(
    messages: list[dict],
    *,
    model=None,
    model_name: str = "gpt-4o-mini",
    system: str,
    max_steps: int,
    enable_tools: bool,
) -> Iterator[str]:
    """Same as run_chat, but yields ⚙ step lines live, then the final answer.

    The SDK loop is synchronous, so it runs in a worker thread while a queue-backed
    tracer streams step lines out; the final answer is yielded once the loop finishes.
    """
    _, prior, task = split_messages(messages, system)
    registry = ToolRegistry([read_file, write_file, run_command] if enable_tools else [])
    if model is None:
        model = OpenAIChat(model=model_name, tools=registry.openai_schemas() or None)
    agent = Agent(model, registry, system, render=history_render(prior))

    q: "queue.Queue[str | None]" = queue.Queue()
    box: dict[str, str] = {}

    def worker() -> None:
        try:
            result = agent.invoke(task, RunConfig(max_steps=max_steps, tracer=_QueueTracer(q)))
            box["out"] = result.output or f"[no output: {result.reason}]"
        except Exception as e:
            box["out"] = f"error: {type(e).__name__}: {e}"
        finally:
            q.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    streamed_step = False
    for line in iter(q.get, None):
        streamed_step = True
        yield line + "\n"
    thread.join()
    yield ("\n" if streamed_step else "") + box.get("out", "")
