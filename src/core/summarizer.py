from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from agent_sdk.messages import sys as sys_msg
from agent_sdk.messages import user
from agent_sdk.types import RunConfig
from context_layer.manager import ContextItem
from context_layer.context_policy import ContextType
from context_layer.tokenizer import _text_of

if TYPE_CHECKING:
    from context_layer.manager import Turn


SUMMARY_SYS = (
    "You compress earlier conversation turns into a terse factual summary. "
    "Preserve names, decisions, file paths, numbers, and open threads. "
    "Output only the summary text."
)


class Summarizer(Protocol):
    def extend(self, prior: ContextItem | None, evicted: list["Turn"]) -> ContextItem: ...


class LLMSummarizer:
    """Fold newly evicted turns into one rolling summary."""

    def __init__(self, model, cfg: RunConfig | None = None):
        self.model = model
        self.cfg = cfg or RunConfig()

    def extend(self, prior: ContextItem | None, evicted: list["Turn"]) -> ContextItem:
        prior_text = prior.message["content"] if prior else "(none)"
        evicted_text = "\n".join(_text_of(item.message) for turn in evicted for item in turn.items)
        prompt = [
            sys_msg(SUMMARY_SYS),
            user(f"Prior summary:\n{prior_text}\n\nNew turns to fold in:\n{evicted_text}"),
        ]
        text = self.model.invoke(prompt, self.cfg).message.get("content") or ""
        return ContextItem(
            type=ContextType.SUMMARY,
            message={"role": "system", "content": "Earlier conversation summary:\n" + text},
            priority=950,
            pinned=True,
        )
