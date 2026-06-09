from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_sdk.agent import Agent
from agent_sdk.builtin_tools import read_file, run_command, write_file
from agent_sdk.openai_model import OpenAIChat
from agent_sdk.tools import ToolRegistry
from agent_sdk.tracer import StdoutTracer
from agent_sdk.types import RunConfig

SYSTEM = """You are a shell agent.
Use tools to write the requested Python script, run it, then react to the tool result.
Use run_command only with argv ["python3", "<script path>"].
If the command exits non-zero, read the same script file before giving the final diagnosis.
Keep final answers concise."""


def run(label: str, script_body: str) -> None:
    script_path = str(Path(tempfile.mkdtemp()) / "check.py")
    registry = ToolRegistry([read_file, write_file, run_command])
    model = OpenAIChat(model="gpt-4o-mini", tools=registry.openai_schemas())
    agent = Agent(model, registry, SYSTEM)
    task = (
        f"Write this exact Python script to {script_path!r}: {script_body!r}. "
        f"Then run it with run_command argv ['python3', {script_path!r}]. "
        "If it fails, read the same file and identify the suspect source line."
    )
    print(f"\n===== {label} =====")
    result = agent.invoke(task, RunConfig(tracer=StdoutTracer(), max_steps=6))
    print(f"[result] {result.output}")


if __name__ == "__main__":
    run("HEALTHY SCRIPT (exit 0)", 'print("harness agent online")\n')
    run("INTENTIONAL FAILURE SCRIPT (exit 1)", 'raise SystemExit("intentional failure for diagnosis")\n')
