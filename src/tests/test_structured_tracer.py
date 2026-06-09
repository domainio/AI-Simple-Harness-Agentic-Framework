import io
import json

from agent_sdk.agent import Agent
from agent_sdk.tools import ToolRegistry, tool
from agent_sdk.types import ModelResponse, RunConfig
from core.structured_tracer import MultiTracer, StructuredTracer
from pydantic import BaseModel, Field


def test_structured_tracer_emits_one_json_line_per_event():
    buf = io.StringIO()
    tracer = StructuredTracer(stream=buf)

    tracer.on_model_message(
        1,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
    )
    tracer.on_tool_start(1, "f", "{}")
    tracer.on_tool_end(1, "f", "result")
    tracer.on_stop("stop", 2)

    recs = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert [rec["kind"] for rec in recs] == ["model", "tool_start", "tool_end", "stop"]
    assert recs[2]["result_len"] == 6
    assert "latency_ms" in recs[2]
    assert all("ts" in rec for rec in recs)


class Recording:
    def __init__(self):
        self.events = []

    def on_model_message(self, *args):
        self.events.append("model")

    def on_tool_start(self, *args):
        self.events.append("tool_start")

    def on_tool_end(self, *args):
        self.events.append("tool_end")

    def on_stop(self, *args):
        self.events.append("stop")


def test_multitracer_fans_out_to_all():
    a = Recording()
    b = Recording()
    tracer = MultiTracer([a, b])

    tracer.on_model_message(1, {"role": "assistant", "content": "x"})
    tracer.on_stop("stop", 1)

    assert a.events == ["model", "stop"]
    assert b.events == ["model", "stop"]


def test_multitracer_on_assembly_skips_tracers_without_it():
    rec = Recording()
    tracer = MultiTracer([rec])

    tracer.on_assembly(kept=1, evicted=0, truncated=0, tokens=5, budget=60)

    assert rec.events == []


class PathArgs(BaseModel):
    path: str = Field(description="path")


@tool(args=PathArgs)
def fake_read(path: str) -> str:
    return "HELLO"


class CallThenStop:
    def invoke(self, messages, cfg) -> ModelResponse:
        if not any(m.get("role") == "tool" for m in messages):
            return ModelResponse(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "fake_read", "arguments": '{"path":"x"}'},
                        }
                    ],
                },
                "tool_calls",
            )
        return ModelResponse({"role": "assistant", "content": "done"}, "stop")


def test_structured_tracer_through_agent():
    buf = io.StringIO()
    reg = ToolRegistry()
    reg.register(fake_read)

    Agent(CallThenStop(), reg, "sys").invoke("go", RunConfig(tracer=StructuredTracer(stream=buf)))

    kinds = [json.loads(line)["kind"] for line in buf.getvalue().splitlines()]
    assert "tool_start" in kinds
    assert "tool_end" in kinds
    assert "stop" in kinds
