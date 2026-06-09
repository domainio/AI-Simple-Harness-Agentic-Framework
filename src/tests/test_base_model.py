from agent_sdk.base_model import BaseChatModel, WIRE_KEYS
from agent_sdk.protocols import ChatModel
from agent_sdk.types import ModelResponse, RunConfig


class FakeProvider(BaseChatModel):
    def _encode(self, messages):
        return {"sent": [{k: v for k, v in m.items() if k in WIRE_KEYS} for m in messages]}

    def _call(self, wire, cfg):
        return {"wire": wire, "text": "ok"}

    def _decode(self, raw):
        return ModelResponse({"role": "assistant", "content": raw["text"]}, "stop")


def test_template_wires_encode_call_decode():
    resp = FakeProvider().invoke([{"role": "user", "content": "hi", "reasoning": "noise"}], RunConfig())
    assert resp.finish_reason == "stop" and resp.message["content"] == "ok"


def test_subclass_is_structural_chatmodel():
    assert isinstance(FakeProvider(), ChatModel)
