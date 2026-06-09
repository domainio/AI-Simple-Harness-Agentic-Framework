from pydantic import BaseModel, Field

from agent_sdk.agent import Agent
from agent_sdk.tools import ToolRegistry, tool
from agent_sdk.types import ModelResponse, RunConfig
from context_layer.manager import ContextItem, ContextManager
from context_layer.tokenizer import TiktokenCounter


def test_render_returns_messages_within_budget():
    cm = ContextManager(TiktokenCounter("gpt-4o-mini"), budget=40, truncate_cap=20)
    cm.register(
        ContextItem(
            type="doc",
            message={"role": "system", "content": "handbook"},
            priority=40,
            truncatable=True,
        )
    )
    transcript = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "old " * 30},
        {"role": "user", "content": "current task"},
    ]
    out = cm.render(transcript)
    assert all(isinstance(m, dict) and "role" in m for m in out)
    assert any(m["content"] == "SYS" for m in out)
    assert any(m["content"] == "current task" for m in out)
    assert cm.last_stats["tokens"] <= cm.budget


class PathArgs(BaseModel):
    path: str = Field(description="path")


@tool(args=PathArgs)
def fake_read(path: str) -> str:
    return "HELLO"


class FeedbackModel:
    def invoke(self, messages, cfg) -> ModelResponse:
        assert messages[0]["role"] == "system"
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


def test_agent_runs_end_to_end_with_render_seam():
    reg = ToolRegistry()
    reg.register(fake_read)
    cm = ContextManager(TiktokenCounter("gpt-4o-mini"), budget=100_000)
    agent = Agent(FeedbackModel(), reg, "system prompt", render=cm.render)
    result = agent.invoke("do the task", RunConfig())
    assert result.status == "complete" and result.output == "done"
