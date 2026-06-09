from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

PolicyAction = Literal["allow", "deny", "require_approval"]
Risk = Literal["low", "medium", "high"]
Confidence = Literal["low", "medium", "high"]
Scope = Literal["readonly", "workspace_write", "destructive", "network", "unknown"]


class ToolPolicyDecision(BaseModel):
    """Policy label proposed for a tool call.

    `action` is not the only authority: risk, scope, and confidence can still
    escalate a call to approval inside the agent gate.
    """

    model_config = ConfigDict(frozen=True)

    action: PolicyAction
    risk: Risk
    confidence: Confidence
    scope: Scope
    reason: str


class PolicyEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    consistent: bool
    reason: str = ""


PolicyModel = Callable[[str, dict], ToolPolicyDecision | dict]
PolicyEvaluator = Callable[[str, dict, ToolPolicyDecision], PolicyEvaluation | dict]
PolicyApprover = Callable[[ToolPolicyDecision], bool] | Callable[[str, dict, ToolPolicyDecision], bool]

DESTRUCTIVE_COMMANDS = {
    "bash",
    "chmod",
    "chown",
    "dd",
    "kill",
    "killall",
    "mv",
    "rm",
    "rmdir",
    "sh",
    "sudo",
    "zsh",
}
NETWORK_COMMANDS = {"curl", "scp", "ssh", "wget"}
WRAPPER_COMMANDS = {"busybox", "env", "xargs"}
ENV_OPTIONS_WITH_OPERANDS = {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}
XARGS_OPTIONS_WITH_OPERANDS = {
    "-a",
    "--arg-file",
    "-d",
    "--delimiter",
    "-E",
    "-I",
    "-i",
    "-L",
    "-l",
    "-n",
    "--max-args",
    "-P",
    "--max-procs",
    "-s",
    "--max-chars",
}


def default_tool_policy_model(tool_name: str, args: dict) -> ToolPolicyDecision:
    """Infer baseline tool risk from tool name and args."""
    if tool_name == "read_file":
        return ToolPolicyDecision(
            action="allow",
            risk="low",
            confidence="high",
            scope="readonly",
            reason="read_file only reads content",
        )
    if tool_name == "write_file":
        return ToolPolicyDecision(
            action="allow",
            risk="medium",
            confidence="high",
            scope="workspace_write",
            reason="write_file changes file content",
        )
    if tool_name == "run_command":
        return _classify_command(args.get("argv"))
    return ToolPolicyDecision(
        action="require_approval",
        risk="low",
        confidence="low",
        scope="unknown",
        reason="no built-in policy rule for tool",
    )


def default_semantic_consistency_evaluator(tool_name: str, args: dict, decision: ToolPolicyDecision) -> PolicyEvaluation:
    """Check whether policy labels match obvious tool semantics."""
    expected = default_tool_policy_model(tool_name, args)
    if expected.risk == "high" and expected.scope == "destructive":
        if decision.risk != "high" or decision.scope != "destructive":
            return PolicyEvaluation(
                consistent=False,
                reason="destructive command was not labeled high risk/destructive scope",
            )
        if decision.action == "allow":
            return PolicyEvaluation(
                consistent=False,
                reason="destructive command cannot be directly allowed",
            )

    if expected.scope == "network":
        if decision.scope != "network":
            return PolicyEvaluation(consistent=False, reason="network command was not labeled network scope")
        if decision.action == "allow":
            return PolicyEvaluation(consistent=False, reason="network command cannot be directly allowed")

    if tool_name == "read_file" and decision.scope != "readonly":
        return PolicyEvaluation(consistent=False, reason="read_file must be readonly scope")

    if tool_name == "write_file" and decision.scope == "readonly":
        return PolicyEvaluation(consistent=False, reason="write_file cannot be readonly scope")

    return PolicyEvaluation(consistent=True)


def _classify_command(argv: object) -> ToolPolicyDecision:
    if not isinstance(argv, list) or not argv:
        return ToolPolicyDecision(
            action="deny",
            risk="high",
            confidence="high",
            scope="destructive",
            reason="run_command argv must be a non-empty list",
        )

    words = [str(x) for x in argv]
    parsed = _effective_command(words)
    joined = " ".join(words)
    if parsed is None:
        return ToolPolicyDecision(
            action="require_approval",
            risk="high",
            confidence="low",
            scope="unknown",
            reason=f"command wrapper could not be parsed safely: {joined}",
        )

    exe, effective = parsed
    inline_code = _is_python(exe) and "-c" in effective
    if exe in DESTRUCTIVE_COMMANDS or inline_code:
        return ToolPolicyDecision(
            action="require_approval",
            risk="high",
            confidence="high",
            scope="destructive",
            reason=f"command may destroy or alter files/processes: {joined}",
        )

    if exe in NETWORK_COMMANDS:
        return ToolPolicyDecision(
            action="require_approval",
            risk="medium",
            confidence="medium",
            scope="network",
            reason=f"command may access network: {joined}",
        )

    if Path(words[0]).name in WRAPPER_COMMANDS and exe == Path(words[0]).name:
        return ToolPolicyDecision(
            action="require_approval",
            risk="high",
            confidence="low",
            scope="unknown",
            reason=f"command wrapper could not be parsed safely: {joined}",
        )

    return ToolPolicyDecision(
        action="allow",
        risk="medium",
        confidence="medium",
        scope="workspace_write",
        reason=f"command execution can affect local state: {joined}",
    )


def _effective_command(words: list[str]) -> tuple[str, list[str]] | None:
    exe = Path(words[0]).name
    if exe == "env":
        i = 1
        while i < len(words):
            word = words[i]
            if word == "--":
                i += 1
                break
            if word in ENV_OPTIONS_WITH_OPERANDS:
                i += 2
                continue
            if word.startswith("--unset=") or word.startswith("--chdir=") or word.startswith("--split-string="):
                i += 1
                continue
            if word.startswith("-"):
                i += 1
                continue
            if "=" in word:
                i += 1
                continue
            break
        if i >= len(words):
            return None
        return Path(words[i]).name, words[i:]
    if exe == "busybox" and len(words) > 1:
        return Path(words[1]).name, words[1:]
    if exe == "busybox":
        return None
    if exe == "xargs":
        i = 1
        while i < len(words):
            word = words[i]
            if word == "--":
                i += 1
                break
            if word in XARGS_OPTIONS_WITH_OPERANDS:
                i += 2
                continue
            if word.startswith("--") and "=" in word:
                i += 1
                continue
            if word.startswith("-"):
                i += 1
                continue
            break
        if i >= len(words):
            return None
        return Path(words[i]).name, words[i:]
    return exe, words


def _is_python(exe: str) -> bool:
    if exe == "python":
        return True
    suffix = exe.removeprefix("python")
    return exe.startswith("python") and bool(suffix) and all(c.isdigit() or c == "." for c in suffix)
