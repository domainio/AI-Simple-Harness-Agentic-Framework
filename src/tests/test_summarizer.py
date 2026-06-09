from agent_sdk.types import ModelResponse, RunConfig
from core.summarizer import LLMSummarizer
from context_layer.manager import ContextItem, Turn


class FakeModel:
    def __init__(self):
        self.last_messages = None

    def invoke(self, messages, cfg) -> ModelResponse:
        self.last_messages = messages
        return ModelResponse({"role": "assistant", "content": "COMPRESSED"}, "stop")


def _turn(content):
    return Turn([ContextItem(type="user", message={"role": "user", "content": content}, priority=1)])


def test_summarizer_produces_pinned_summary_item():
    model = FakeModel()
    summarizer = LLMSummarizer(model)

    item = summarizer.extend(prior=None, evicted=[_turn("alice paid 30 dollars")])

    assert item.type == "summary"
    assert item.pinned is True
    assert item.priority == 950
    assert item.message["role"] == "system"
    assert item.message["content"].startswith("Earlier conversation summary:")
    assert "COMPRESSED" in item.message["content"]


def test_summarizer_folds_prior_summary_into_prompt():
    model = FakeModel()
    summarizer = LLMSummarizer(model)
    prior = ContextItem(
        type="summary",
        message={"role": "system", "content": "Earlier conversation summary:\nPRIOR"},
        priority=950,
        pinned=True,
    )

    summarizer.extend(prior=prior, evicted=[_turn("new fact: bob left")])

    prompt_text = " ".join(m.get("content") or "" for m in model.last_messages)
    assert "PRIOR" in prompt_text
    assert "bob left" in prompt_text
