from agent_sdk.tracer import NoopTracer, StdoutTracer


def test_noop_tracer_is_silent(capsys):
    t = NoopTracer()
    t.on_model_message(1, {"role": "assistant", "content": "x"})
    t.on_tool_start(1, "read_file", "{}")
    t.on_tool_end(1, "read_file", "ok")
    t.on_stop("stop", 1)
    assert capsys.readouterr().out == ""


def test_stdout_tracer_emits_ordered_lines(capsys):
    t = StdoutTracer()
    t.on_model_message(
        1,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
    )
    t.on_tool_start(1, "read_file", "{}")
    t.on_tool_end(1, "read_file", "HELLO")
    t.on_stop("stop", 2)
    out = capsys.readouterr().out
    assert "step 1" in out and "read_file" in out and "HELLO" in out
    assert out.index("tool ->") < out.index("tool <-") and "stop" in out


def test_stdout_tracer_prints_reasoning_when_present(capsys):
    StdoutTracer().on_model_message(1, {"role": "assistant", "content": "ans", "reasoning": "let me think"})
    out = capsys.readouterr().out
    assert "reasoning" in out and "let me think" in out
