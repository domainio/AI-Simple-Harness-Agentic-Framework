"""Core demo: rolling summary + structured telemetry. Requires OPENAI_API_KEY.

Run from repo root: uv run python src/demos/demo_core.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_sdk.agent import Agent
from agent_sdk.openai_model import OpenAIChat
from agent_sdk.tools import ToolRegistry, tool
from agent_sdk.tracer import StdoutTracer
from agent_sdk.types import RunConfig
from core.langfuse_tracer import LangfuseTracer
from core.structured_tracer import MultiTracer, StructuredTracer
from core.summarizer import LLMSummarizer
from core.summarizing_manager import SummarizingContextManager
from context_layer.manager import ContextItem
from context_layer.tokenizer import TiktokenCounter
from pydantic import BaseModel, Field


class TicketArgs(BaseModel):
    ticket_id: str = Field(description="ticket id")


@tool(args=TicketArgs)
def lookup_ticket(ticket_id: str) -> str:
    return f"{ticket_id}: refund pending"


def build():
    telemetry = io.StringIO()
    structured = StructuredTracer(stream=telemetry)
    cm = SummarizingContextManager(
        TiktokenCounter("gpt-4o-mini"),
        budget=300,
        truncate_cap=40,
        summarizer=LLMSummarizer(OpenAIChat(model="gpt-4o-mini")),
        summary_reserve=80,
        on_assembly=structured.on_assembly,
    )
    cm.register(
        ContextItem(
            type="memory",
            message={"role": "user", "content": "old fact " * 120 + "customer paid 30 dollars"},
            priority=650,
        )
    )
    registry = ToolRegistry([lookup_ticket])
    tracer = MultiTracer([StdoutTracer(), structured, LangfuseTracer()])
    model = OpenAIChat(model="gpt-4o-mini", tools=registry.openai_schemas())
    system = "Use lookup_ticket when checking ticket status. Combine tool results with available context."
    return Agent(model, registry, system, render=cm.render), tracer, telemetry


if __name__ == "__main__":
    agent, tracer, telemetry = build()
    result = agent.invoke("Use lookup_ticket for ticket T-7, then answer with the refund status.", RunConfig(tracer=tracer))
    print("\n=== RESULT ===")
    print(result.output)
    print("\n=== JSONL TELEMETRY ===")
    print(telemetry.getvalue(), end="")
