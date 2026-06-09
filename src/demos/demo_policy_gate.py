"""Core demo: real-LLM human-in-the-loop policy gate.

The tool is fake on purpose: the demo proves approval flow without executing a
destructive command.

Run from repo root: uv run python src/demos/demo_policy_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_sdk.agent import Agent
from agent_sdk.openai_model import OpenAIChat
from agent_sdk.policy import ToolPolicyDecision
from agent_sdk.tools import ToolRegistry, tool
from agent_sdk.tracer import StdoutTracer
from agent_sdk.types import RunConfig


SYSTEM = """You are demonstrating a tool policy gate.
The run_command tool in this demo is fake and cannot delete files.
You must call run_command exactly once with argv ["rm", "-rf", "tmp/demo-policy"].
After the tool result, summarize what happened in one short sentence."""

TASK = "Start the HITL policy-gate demo now."


class RunCommandArgs(BaseModel):
    argv: list[str] = Field(min_length=1)


@tool(args=RunCommandArgs)
def run_command(argv: list[str]) -> str:
    """Fake command runner for policy approval demo."""
    return f"FAKE EXECUTED: {argv!r}"


def approving_human(tool_name: str, args: dict, decision: ToolPolicyDecision) -> bool:
    print(f"[approval] tool={tool_name} args={args} scope={decision.scope} reason={decision.reason}")
    return True


def run(label: str, cfg: RunConfig) -> None:
    print(f"\n===== {label} =====")
    registry = ToolRegistry([run_command])
    model = OpenAIChat(model="gpt-4o-mini", tools=registry.openai_schemas())
    agent = Agent(model, registry, SYSTEM)
    result = agent.invoke(TASK, cfg)
    print(f"[result] {result.output}")


if __name__ == "__main__":
    run("NO APPROVER: blocked before tool execution", RunConfig(tracer=StdoutTracer(), max_steps=3))
    run("APPROVER: human accepts, fake tool executes", RunConfig(tracer=StdoutTracer(), max_steps=3, approver=approving_human))
