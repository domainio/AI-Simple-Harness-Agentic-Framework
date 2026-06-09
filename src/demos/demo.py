from __future__ import annotations

import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_sdk.agent import Agent
from agent_sdk.builtin_tools import read_file, run_command, write_file
from agent_sdk.openai_model import OpenAIChat
from agent_sdk.tools import ToolRegistry
from agent_sdk.tracer import StdoutTracer
from agent_sdk.types import RunConfig
from core.langfuse_tracer import LangfuseTracer
from core.structured_tracer import MultiTracer

SYSTEM = """You are a minimal file-system agent.
Use tools when needed. After tool results, decide the next step from the conversation.
Keep final answers concise."""


def main() -> None:
    sample = DEMO_DIR / "resources" / "TheLittlePrince.txt"
    task = " ".join(sys.argv[1:]) or f"Read {sample} and summarize what its about?"
    registry = ToolRegistry([read_file, write_file, run_command])
    model = OpenAIChat(model="gpt-4o-mini", tools=registry.openai_schemas())
    agent = Agent(model, registry, SYSTEM)
    tracer = MultiTracer([StdoutTracer(), LangfuseTracer()])
    result = agent.invoke(task, RunConfig(tracer=tracer))
    print(result.output)


if __name__ == "__main__":
    main()
