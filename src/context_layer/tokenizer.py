from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import tiktoken

if TYPE_CHECKING:
    from agent_sdk.types import Message


def _text_of(msg: "Message") -> str:
    parts = [msg.get("content") or ""]
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        parts.append(fn["name"])
        parts.append(fn["arguments"])
    return "\n".join(parts)


class Tokenizer(Protocol):
    def count(self, msg: "Message") -> int: ...


class TiktokenCounter:
    """Approximate chat-token counter; exact framing differs by provider/model."""

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            self.enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self.enc = tiktoken.get_encoding("cl100k_base")

    def count(self, msg: "Message") -> int:
        return len(self.enc.encode(_text_of(msg))) + 4

    def truncate_content(self, msg: "Message", token_cap: int, marker: str) -> tuple[str, int]:
        content = msg.get("content") or ""
        tokens = self.enc.encode(content)
        fixed_cost = self.count({**msg, "content": marker})
        keep = max(0, min(len(tokens), token_cap - fixed_cost))

        while True:
            candidate = self.enc.decode(tokens[:keep]) + marker
            cost = self.count({**msg, "content": candidate})
            if cost <= token_cap or keep == 0:
                return candidate, cost
            keep -= max(1, cost - token_cap)
