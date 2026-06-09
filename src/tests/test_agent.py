import io
from contextlib import redirect_stdout

from pydantic import BaseModel

from agent_sdk.agent import Agent
from agent_sdk.tools import ToolRegistry, tool
from agent_sdk.tracer import StdoutTracer
from agent_sdk.types import ModelResponse, RunConfig


class AddArgs(BaseModel):
    x: int
    y: int


@tool(args=AddArgs)
def add(x: int, y: int) -> str:
    """Add two integers."""
    return str(x + y)


class SequenceModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def invoke(self, messages, cfg):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def tool_call(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def test_loop_is_feedback_driven_and_returns_final_answer():
    model = SequenceModel(
        [
            ModelResponse(
                {"role": "assistant", "content": None, "tool_calls": [tool_call("c1", "add", '{"x":2,"y":3}')]},
                "tool_calls",
            ),
            ModelResponse({"role": "assistant", "content": "5"}, "stop"),
        ]
    )
    result = Agent(model, ToolRegistry([add]), "sys").invoke("calc", RunConfig(tool_policy_model=None))
    assert result.status == "complete" and result.output == "5"
    assert model.seen[1][-1] == {"role": "tool", "tool_call_id": "c1", "content": "5"}


def test_unknown_tool_failure_flows_back_as_tool_message():
    model = SequenceModel(
        [
            ModelResponse(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("c1", "missing", "{}")],
                },
                "tool_calls",
            ),
            ModelResponse({"role": "assistant", "content": "handled"}, "stop"),
        ]
    )
    result = Agent(model, ToolRegistry(), "sys").invoke("go", RunConfig())
    assert result.status == "complete"
    assert result.history[3]["content"].startswith("error: KeyError:")


def test_malformed_json_failure_flows_back_as_tool_message():
    model = SequenceModel(
        [
            ModelResponse(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("c1", "add", "{bad")],
                },
                "tool_calls",
            ),
            ModelResponse({"role": "assistant", "content": "handled"}, "stop"),
        ]
    )
    result = Agent(model, ToolRegistry([add]), "sys").invoke("go", RunConfig(tool_policy_model=None))
    assert result.history[3]["content"].startswith("error: JSONDecodeError:")


def test_multi_tool_calls_run_in_order():
    model = SequenceModel(
        [
            ModelResponse(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("c1", "add", '{"x":1,"y":2}'),
                        tool_call("c2", "add", '{"x":3,"y":4}'),
                    ],
                },
                "tool_calls",
            ),
            ModelResponse({"role": "assistant", "content": "done"}, "stop"),
        ]
    )
    result = Agent(model, ToolRegistry([add]), "sys").invoke("go", RunConfig(tool_policy_model=None))
    assert result.history[3]["content"] == "3"
    assert result.history[4]["content"] == "7"


def test_max_steps_returns_incomplete():
    model = SequenceModel(
        [
            ModelResponse(
                {"role": "assistant", "content": None, "tool_calls": [tool_call("c1", "add", '{"x":1,"y":1}')]},
                "tool_calls",
            )
        ]
    )
    result = Agent(model, ToolRegistry([add]), "go").invoke("go", RunConfig(max_steps=1))
    assert result.status == "incomplete"
    assert result.reason == "max_steps"


def test_length_returns_incomplete():
    model = SequenceModel([ModelResponse({"role": "assistant", "content": "partial"}, "length")])
    result = Agent(model, ToolRegistry(), "sys").invoke("go", RunConfig())
    assert result.status == "incomplete"
    assert result.output == "partial"
    assert result.reason == "length"


def test_reasoning_flows_to_trace():
    class ReasoningModel:
        def invoke(self, messages, cfg) -> ModelResponse:
            return ModelResponse({"role": "assistant", "content": "answer", "reasoning": "MY-THOUGHTS"}, "stop")

    buf = io.StringIO()
    with redirect_stdout(buf):
        Agent(ReasoningModel(), ToolRegistry(), "sys").invoke("go", RunConfig(tracer=StdoutTracer()))
    assert "MY-THOUGHTS" in buf.getvalue()
