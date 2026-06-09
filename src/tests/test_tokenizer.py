from context_layer.tokenizer import TiktokenCounter, _text_of


def test_text_of_includes_content_and_tool_calls():
    msg = {
        "role": "assistant",
        "content": "hi",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"x"}'},
            }
        ],
    }
    text = _text_of(msg)
    assert "hi" in text and "read_file" in text and "path" in text


def test_counter_adds_framing_overhead():
    c = TiktokenCounter("gpt-4o-mini")
    n = c.count({"role": "user", "content": "hello world"})
    assert n > 4
    empty = c.count({"role": "user", "content": ""})
    assert empty == 4


def test_counter_grows_with_content():
    c = TiktokenCounter("gpt-4o-mini")
    short = c.count({"role": "user", "content": "a"})
    long = c.count({"role": "user", "content": "a " * 200})
    assert long > short
