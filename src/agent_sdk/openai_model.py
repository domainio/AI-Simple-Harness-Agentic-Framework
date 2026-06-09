from __future__ import annotations

import os
from pathlib import Path

from agent_sdk.base_model import BaseChatModel, WIRE_KEYS
from agent_sdk.types import Message, ModelResponse, RunConfig

ENV = Path(__file__).resolve().parents[2] / ".env"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_openai_api_key(env_path: Path = ENV) -> None:
    if os.environ.get("OPENAI_API_KEY") or not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if key == "OPENAI_API_KEY" and value.strip():
            os.environ["OPENAI_API_KEY"] = _unquote(value)
            return


class OpenAIChat(BaseChatModel):
    """Canonical-IR adapter. Pass tools=registry.openai_schemas()."""

    def __init__(self, model: str = "gpt-4o-mini", client=None, tools: list[dict] | None = None):
        if client is None:
            _load_openai_api_key()
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model
        self.tools = tools

    def _encode(self, messages: list[Message]) -> list[dict]:
        return [{k: v for k, v in m.items() if k in WIRE_KEYS} for m in messages]

    def _call(self, wire: list[dict], cfg: RunConfig):
        return self.client.chat.completions.create(
            model=self.model,
            messages=wire,
            tools=self.tools or None,
        )

    def _decode(self, raw) -> ModelResponse:
        choice = raw.choices[0]
        m = choice.message
        out: Message = {"role": "assistant", "content": m.content}
        if getattr(m, "tool_calls", None):
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in m.tool_calls
            ]
        reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
        if reasoning:
            out["reasoning"] = reasoning
        raw_usage = getattr(raw, "usage", None)
        usage = None
        if raw_usage is not None:
            usage = {"input": raw_usage.prompt_tokens, "output": raw_usage.completion_tokens}
        return ModelResponse(
            message=out,
            finish_reason=choice.finish_reason,
            usage=usage,
            model=getattr(raw, "model", None) or self.model,
        )
