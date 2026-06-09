import pytest

from agent_sdk.protocols import ChatModel, Tool, Tracer
from agent_sdk.tracer import NoopTracer
from agent_sdk.types import (
    MAX_STEPS_CEILING,
    AgentResult,
    Message,
    ModelResponse,
    RunConfig,
)


def test_model_response_carries_finish_reason():
    msg: Message = {"role": "assistant", "content": "hi"}
    resp = ModelResponse(message=msg, finish_reason="stop")
    assert resp.finish_reason == "stop" and resp.message["content"] == "hi"


def test_message_can_carry_reasoning():
    m: Message = {"role": "assistant", "content": "x", "reasoning": "thought first"}
    assert m["reasoning"] == "thought first"


def test_agent_result_fields():
    r = AgentResult(status="complete", output="done", reason="stop", steps=2, history=[])
    assert (r.status, r.reason, r.steps) == ("complete", "stop", 2)
    assert r.output == "done" and r.history == []


def test_runconfig_defaults():
    cfg = RunConfig()
    assert cfg.max_steps == 10 and isinstance(cfg.tracer, NoopTracer)


def test_runconfig_rejects_over_ceiling():
    with pytest.raises(ValueError):
        RunConfig(max_steps=MAX_STEPS_CEILING + 1)


def test_runconfig_rejects_nonpositive():
    with pytest.raises(ValueError):
        RunConfig(max_steps=0)


def test_protocols_are_importable_and_runtime_tracer_works():
    assert isinstance(NoopTracer(), Tracer)
    assert ChatModel and Tool
