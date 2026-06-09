from __future__ import annotations

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
from core.structured_tracer import MultiTracer
from pydantic import BaseModel, Field


class WeatherArgs(BaseModel):
    city: str = Field(description="city name")


@tool(args=WeatherArgs)
def get_weather(city: str) -> str:
    return f"{city}: 22C, clear"


if __name__ == "__main__":
    registry = ToolRegistry([get_weather])
    tracer = MultiTracer([StdoutTracer(), LangfuseTracer()])
    model = OpenAIChat(model="gpt-4o-mini", tools=registry.openai_schemas())
    agent = Agent(model, registry, "Use get_weather when asked for current weather, then answer briefly.")
    result = agent.invoke("Use get_weather for Tel Aviv and answer briefly.", RunConfig(tracer=tracer))
    print("\n=== RESULT ===")
    print(result.output)
    print("\nPublished to Langfuse. Open http://localhost:3000")
