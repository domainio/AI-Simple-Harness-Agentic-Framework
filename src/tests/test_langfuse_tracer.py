from agent_sdk.agent import Agent
from agent_sdk.tools import ToolRegistry, tool
from agent_sdk.types import ModelResponse, RunConfig
from core.langfuse_tracer import LangfuseTracer
from pydantic import BaseModel


class FakeObservation:
    def __init__(self, fields):
        self.fields = fields
        self.updates = []
        self.ended = False

    def update(self, **fields):
        self.updates.append(fields)


class FakeContext:
    def __init__(self, obs):
        self.obs = obs

    def __enter__(self):
        return self.obs

    def __exit__(self, exc_type, exc, tb):
        self.obs.ended = True


class FakeLangfuse:
    def __init__(self):
        self.observations = []
        self.flushes = 0

    def start_as_current_observation(self, **fields):
        obs = FakeObservation(fields)
        self.observations.append(obs)
        return FakeContext(obs)

    def flush(self):
        self.flushes += 1


class ReadArgs(BaseModel):
    path: str


@tool(args=ReadArgs)
def fake_read(path: str) -> str:
    return "HELLO"


class ToolThenStop:
    def invoke(self, messages, cfg):
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
                usage={"input": 100, "output": 10},
                model="gpt-4o-mini",
            )
        return ModelResponse(
            {"role": "assistant", "content": "done"},
            "stop",
            usage={"input": 120, "output": 5},
            model="gpt-4o-mini",
        )


def test_langfuse_tracer_records_agent_model_and_tool_spans():
    client = FakeLangfuse()
    registry = ToolRegistry()
    registry.register(fake_read)

    result = Agent(ToolThenStop(), registry, "sys").invoke(
        "go",
        RunConfig(tracer=LangfuseTracer(client=client), tool_policy_model=None),
    )

    assert result.output == "done"
    assert client.flushes == 1
    assert [obs.fields["name"] for obs in client.observations] == [
        "agent",
        "model.step.1",
        "tool.fake_read",
        "model.step.2",
    ]
    assert [obs.fields["as_type"] for obs in client.observations] == ["span", "generation", "span", "generation"]
    assert all(obs.ended for obs in client.observations)
    assert client.observations[0].updates[-1]["output"] == "done"
    assert client.observations[1].updates[-1]["metadata"]["finish_reason"] == "tool_calls"
    assert client.observations[1].updates[-1]["usage_details"] == {"input": 100, "output": 10}
    assert client.observations[1].updates[-1]["model"] == "gpt-4o-mini"
    assert client.observations[2].updates[-1]["output"] == "HELLO"
