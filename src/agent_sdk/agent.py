from __future__ import annotations

import inspect
import json
from collections.abc import Callable

from agent_sdk.messages import identity, sys, tool_msg, user
from agent_sdk.policy import PolicyEvaluation, ToolPolicyDecision
from agent_sdk.protocols import ChatModel
from agent_sdk.tools import ToolRegistry
from agent_sdk.types import AgentResult, Message, RunConfig


def _notify(tracer: object, event: str, *args) -> None:
    fn = getattr(tracer, event, None)
    if fn is not None:
        fn(*args)


def _call_approver(approver: Callable[..., bool], name: str, args: dict, decision: ToolPolicyDecision) -> bool:
    try:
        signature = inspect.signature(approver)
    except (TypeError, ValueError):
        return bool(approver(name, args, decision))

    try:
        signature.bind(name, args, decision)
    except TypeError:
        signature.bind(decision)
        return bool(approver(decision))
    return bool(approver(name, args, decision))


class Agent:
    """Model-driven tool loop with one prompt-assembly hook before each LLM call."""

    def __init__(
        self,
        model: ChatModel,
        registry: ToolRegistry,
        system_prompt: str,
        render: Callable[[list[Message]], list[Message]] = identity,
    ):
        self.model = model
        self.registry = registry
        self.system_prompt = system_prompt
        self.render = render

    def invoke(self, task: str, cfg: RunConfig) -> AgentResult:
        history: list[Message] = [sys(self.system_prompt), user(task)]
        _notify(cfg.tracer, "on_run_start", task, cfg.max_steps)
        for step in range(1, cfg.max_steps + 1):
            messages = self.render(history)
            _notify(cfg.tracer, "on_model_start", step, messages)
            resp = self.model.invoke(messages, cfg)
            _notify(cfg.tracer, "on_model_end", step, resp)
            history.append(resp.message)
            cfg.tracer.on_model_message(step, resp.message)
            finish_reason = resp.finish_reason

            if finish_reason == "tool_calls":
                for tc in resp.message["tool_calls"]:
                    result = self._run_tool(step, tc, cfg)
                    history.append(tool_msg(tc["id"], result))
                continue
            if finish_reason == "stop":
                cfg.tracer.on_stop("stop", step)
                return self._finish("complete", resp.message.get("content") or "", "stop", step, history, cfg)

            cfg.tracer.on_stop(finish_reason, step)
            return self._finish(
                "incomplete",
                resp.message.get("content") or "",
                finish_reason,
                step,
                history,
                cfg,
            )

        cfg.tracer.on_stop("max_steps", cfg.max_steps)
        return self._finish("incomplete", "", "max_steps", cfg.max_steps, history, cfg)

    def _finish(
        self,
        status: str,
        output: str,
        reason: str,
        steps: int,
        history: list[Message],
        cfg: RunConfig,
    ) -> AgentResult:
        result = AgentResult(status, output, reason, steps, history)
        _notify(cfg.tracer, "on_run_end", result)
        return result

    def _run_tool(self, step: int, tc: dict, cfg: RunConfig) -> str:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments") or "{}"
        cfg.tracer.on_tool_start(step, name, raw_args)
        try:
            args = json.loads(raw_args)
            tool = self.registry.get(name)
            result = self._check_policy(name, args, cfg)
            if result is None:
                result = tool.invoke(args, cfg)
        except Exception as e:
            result = f"error: {type(e).__name__}: {e}"
        cfg.tracer.on_tool_end(step, name, result)
        return result

    def _check_policy(self, name: str, args: dict, cfg: RunConfig) -> str | None:
        if cfg.tool_policy_model is None:
            return None

        try:
            decision = ToolPolicyDecision.model_validate(cfg.tool_policy_model(name, args))
        except Exception as e:
            return f"error: policy: invalid policy model: {type(e).__name__}: {e}"

        if cfg.semantic_policy_evaluator is not None:
            try:
                evaluation = PolicyEvaluation.model_validate(cfg.semantic_policy_evaluator(name, args, decision))
            except Exception as e:
                return f"error: policy: invalid semantic consistency evaluator: {type(e).__name__}: {e}"
            if not evaluation.consistent:
                return f"error: policy: inconsistent policy model: {evaluation.reason}"

        if decision.action == "deny":
            return f"error: policy: denied: {decision.reason}"

        approval_required = (
            decision.action == "require_approval"
            or (decision.risk == "high" and decision.scope != "readonly")
            or (decision.confidence == "low" and decision.scope in {"destructive", "network"})
        )
        if approval_required and not (cfg.approver and _call_approver(cfg.approver, name, args, decision)):
            return f"error: policy: approval required: {decision.reason}"

        return None
