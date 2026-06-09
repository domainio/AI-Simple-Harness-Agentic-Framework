import pytest

from agent_sdk.types import ModelResponse
from core.integrations.openwebui_adapter import history_render, run_chat, split_messages, stream_chat


class SequenceModel:
    """FakeModel: replays canned ModelResponses, records messages seen (no network)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def invoke(self, messages, cfg):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def tool_call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


# --- split_messages -----------------------------------------------------------

def test_split_messages_extracts_system_prior_task():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2 task"},
    ]
    system, prior, task = split_messages(messages, "DEFAULT")
    assert system == "SYS"
    assert task == "u2 task"
    assert prior == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_split_messages_no_system_uses_default():
    system, prior, task = split_messages([{"role": "user", "content": "hi"}], "DEFAULT")
    assert system == "DEFAULT"
    assert prior == []
    assert task == "hi"


def test_split_messages_raises_without_user():
    with pytest.raises(ValueError):
        split_messages([{"role": "system", "content": "S"}], "D")


def test_split_messages_sanitizes_tool_calls_and_tool_role():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "tool_calls": [tool_call("c1", "x", "{}")]},
        {"role": "tool", "tool_call_id": "c1", "content": "toolresult"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "task"},
    ]
    system, prior, task = split_messages(messages, "D")
    assert prior == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
    ]
    assert all("tool_calls" not in m for m in prior)
    assert all(m["role"] in {"user", "assistant"} for m in prior)


def test_split_messages_drops_empty_content():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "task"},
    ]
    _, prior, _ = split_messages(messages, "D")
    assert prior == [{"role": "user", "content": "u1"}]


def test_split_messages_caps_oldest_turns_first():
    messages = [
        {"role": "user", "content": "AAAA"},
        {"role": "assistant", "content": "BBBB"},
        {"role": "user", "content": "CCCC"},
        {"role": "user", "content": "task"},
    ]
    _, prior, _ = split_messages(messages, "D", max_prior_chars=8)
    assert prior == [
        {"role": "assistant", "content": "BBBB"},
        {"role": "user", "content": "CCCC"},
    ]


# --- history_render -----------------------------------------------------------

def test_history_render_injects_prior_between_system_and_live():
    prior = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    history = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call("c1", "read_file", "{}")]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    out = history_render(prior)(history)
    assert out == [history[0], *prior, *history[1:]]
    assert out[0]["role"] == "system"


# --- run_chat (model-injection seam, no network) ------------------------------

def test_run_chat_executes_tool_when_enabled(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("FILE-BODY", encoding="utf-8")
    model = SequenceModel([
        ModelResponse(
            {"role": "assistant", "content": None,
             "tool_calls": [tool_call("c1", "read_file", f'{{"path":"{f}"}}')]},
            "tool_calls",
        ),
        ModelResponse({"role": "assistant", "content": "done"}, "stop"),
    ])
    out = run_chat(
        [{"role": "user", "content": "read it"}],
        model=model, system="S", max_steps=4, enable_tools=True,
    )
    assert out == "done"
    tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "FILE-BODY"


def test_run_chat_without_tools_has_empty_registry():
    model = SequenceModel([
        ModelResponse(
            {"role": "assistant", "content": None,
             "tool_calls": [tool_call("c1", "read_file", '{"path":"x"}')]},
            "tool_calls",
        ),
        ModelResponse({"role": "assistant", "content": "done"}, "stop"),
    ])
    run_chat(
        [{"role": "user", "content": "go"}],
        model=model, system="S", max_steps=4, enable_tools=False,
    )
    tool_msgs = [m for m in model.seen[1] if m.get("role") == "tool"]
    assert tool_msgs[0]["content"].startswith("error: KeyError")


def test_run_chat_injects_prior_each_step():
    messages = [
        {"role": "user", "content": "old-q"},
        {"role": "assistant", "content": "old-a"},
        {"role": "user", "content": "task"},
    ]
    model = SequenceModel([
        ModelResponse(
            {"role": "assistant", "content": None,
             "tool_calls": [tool_call("c1", "read_file", '{"path":"x"}')]},
            "tool_calls",
        ),
        ModelResponse({"role": "assistant", "content": "ok"}, "stop"),
    ])
    run_chat(messages, model=model, system="S", max_steps=4, enable_tools=False)
    for seen in model.seen:
        assert seen[0] == {"role": "system", "content": "S"}
        assert {"role": "user", "content": "old-q"} in seen
        assert {"role": "assistant", "content": "old-a"} in seen
        assert {"role": "user", "content": "task"} in seen
        # no orphan tool messages without a preceding assistant tool_calls
        for i, m in enumerate(seen):
            if m.get("role") == "tool":
                assert seen[i - 1].get("tool_calls")


def test_stream_chat_yields_steps_then_answer(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("BODY", encoding="utf-8")
    model = SequenceModel([
        ModelResponse(
            {"role": "assistant", "content": None,
             "tool_calls": [tool_call("c1", "read_file", f'{{"path":"{f}"}}')]},
            "tool_calls",
        ),
        ModelResponse({"role": "assistant", "content": "done"}, "stop"),
    ])
    chunks = list(stream_chat(
        [{"role": "user", "content": "read it"}],
        model=model, system="S", max_steps=4, enable_tools=True,
    ))
    text = "".join(chunks)
    assert "→ read_file" in text          # tool start line
    assert "BODY" in text                       # tool result streamed
    assert text.rstrip().endswith("done")       # final answer last


def test_stream_chat_no_tools_just_answer():
    model = SequenceModel([ModelResponse({"role": "assistant", "content": "hi"}, "stop")])
    text = "".join(stream_chat(
        [{"role": "user", "content": "go"}],
        model=model, system="S", max_steps=3, enable_tools=False,
    ))
    assert "⚙" not in text                 # no step lines when no tools run
    assert text.strip() == "hi"


def test_run_chat_no_output_fallback():
    model = SequenceModel([
        ModelResponse(
            {"role": "assistant", "content": None,
             "tool_calls": [tool_call("c1", "read_file", '{"path":"x"}')]},
            "tool_calls",
        ),
    ])
    out = run_chat(
        [{"role": "user", "content": "go"}],
        model=model, system="S", max_steps=1, enable_tools=False,
    )
    assert out == "[no output: max_steps]"
