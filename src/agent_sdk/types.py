from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from agent_sdk.policy import (
    PolicyApprover,
    PolicyEvaluator,
    PolicyModel,
    default_semantic_consistency_evaluator,
    default_tool_policy_model,
)
from agent_sdk.tracer import NoopTracer


class Message(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[dict]
    tool_call_id: str
    reasoning: str | None


@dataclass
class ModelResponse:
    message: Message
    finish_reason: str
    usage: dict | None = None
    model: str | None = None


@dataclass
class AgentResult:
    status: Literal["complete", "incomplete"]
    output: str
    reason: str
    steps: int
    history: list[Message]


MAX_STEPS_CEILING = 50


@dataclass
class RunConfig:
    tracer: object = field(default_factory=NoopTracer)
    max_steps: int = 10
    tool_policy_model: PolicyModel | None = default_tool_policy_model
    semantic_policy_evaluator: PolicyEvaluator | None = default_semantic_consistency_evaluator
    approver: PolicyApprover | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= MAX_STEPS_CEILING:
            raise ValueError(f"max_steps must be in 1..{MAX_STEPS_CEILING}, got {self.max_steps}")
