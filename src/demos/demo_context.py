from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_sdk.agent import Agent
from agent_sdk.builtin_tools import read_file, run_command, write_file
from agent_sdk.openai_model import OpenAIChat
from agent_sdk.tools import ToolRegistry
from agent_sdk.tracer import StdoutTracer
from agent_sdk.types import Message, RunConfig
from context_layer.manager import ContextItem, ContextManager
from context_layer.tokenizer import TiktokenCounter

BOOK = ROOT / "demos" / "resources" / "TheLittlePrince.txt"
SYSTEM = "You are a file agent. Use the read_file tool when asked, then stop with a one-line answer."
# Low-priority background context that does not fit once the big tool output lands.
LOW_PRIORITY_DOC = "Background note: earlier discussion covered logging and CI config. " * 50
CUSTOM_CONTEXT = "User preference: answer with one concise sentence."

BUDGET = 700
TRUNCATE_CAP = 300


class RenderProbe:
    """Wraps the manager's render to capture the last assembled prompt for verification."""

    def __init__(self, cm: ContextManager):
        self.cm = cm
        self.raw_in: list[Message] = []
        self.assembled: list[Message] = []

    def render(self, messages: list[Message]) -> list[Message]:
        self.raw_in = list(messages)
        self.assembled = self.cm.render(messages)
        return self.assembled


def build() -> tuple[Agent, ContextManager, RenderProbe]:
    registry = ToolRegistry([read_file, write_file, run_command])
    model = OpenAIChat(model="gpt-4o-mini", tools=registry.openai_schemas())
    cm = ContextManager(TiktokenCounter("gpt-4o-mini"), budget=BUDGET, truncate_cap=TRUNCATE_CAP)
    cm.register(
        ContextItem(
            type="user_preference",
            message={"role": "system", "content": CUSTOM_CONTEXT},
            priority=650,
        )
    )
    cm.register(
        ContextItem(
            type="doc",
            message={"role": "system", "content": LOW_PRIORITY_DOC},
            priority=400,
        )
    )
    probe = RenderProbe(cm)
    return Agent(model, registry, SYSTEM, render=probe.render), cm, probe


def _tool_tokens(messages: list[Message], counter: TiktokenCounter) -> int:
    return sum(counter.count(m) for m in messages if m.get("role") == "tool")


def main() -> int:
    agent, cm, probe = build()
    counter = TiktokenCounter("gpt-4o-mini")
    task = f"Use read_file to read {BOOK}, then quote its first non-empty line exactly."

    result = agent.invoke(task, RunConfig(tracer=StdoutTracer()))
    print("\n=== RESULT ===", result.status, result.reason)
    print(result.output)

    stats = cm.last_stats
    raw_tool = _tool_tokens(probe.raw_in, counter)
    kept_tool = _tool_tokens(probe.assembled, counter)
    system_kept = any(m.get("role") == "system" and m.get("content") == SYSTEM for m in probe.assembled)
    custom_context_seen = any(
        m.get("role") == "system" and m.get("content") == CUSTOM_CONTEXT for m in probe.assembled
    )

    print("\n=== CONTEXT BUDGETING PROOF ===")
    print(f"budget={BUDGET} truncate_cap={TRUNCATE_CAP}")
    print("last assembly stats:", stats)
    print(f"large tool output: {raw_tool} tok in history -> {kept_tool} tok in prompt")

    checks = [
        ("token budget respected", stats.get("tokens", 0) <= BUDGET, f"{stats.get('tokens')} <= {BUDGET}"),
        ("system instruction intact", system_kept, "system message present in final prompt"),
        ("custom context type supported", custom_context_seen, "user_preference added without core changes"),
        ("large tool output truncated", stats.get("truncated", 0) >= 1, f"truncated={stats.get('truncated')}"),
        ("low-priority context evicted", stats.get("evicted", 0) >= 1, f"evicted={stats.get('evicted')}"),
    ]
    print()
    ok = True
    for name, passed, detail in checks:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
