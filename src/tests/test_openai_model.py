import os
from pathlib import Path
from types import SimpleNamespace

from agent_sdk.openai_model import ENV, OpenAIChat, _load_openai_api_key
from agent_sdk.types import RunConfig


class FakeCompletions:
    def __init__(self, choice, usage=None, model=None):
        self._choice = choice
        self._usage = usage
        self._model = model
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(choices=[self._choice], usage=self._usage, model=self._model)


def _client_returning(message, finish_reason, usage=None, model=None):
    raw = SimpleNamespace(message=message, finish_reason=finish_reason)
    completions = FakeCompletions(raw, usage=usage, model=model)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_decode_extracts_token_usage_and_model():
    msg = SimpleNamespace(content="ans", tool_calls=None, reasoning=None, reasoning_content=None)
    usage = SimpleNamespace(prompt_tokens=312, completion_tokens=18)
    client, _ = _client_returning(msg, "stop", usage=usage, model="gpt-4o-mini")
    resp = OpenAIChat(client=client).invoke([{"role": "user", "content": "hi"}], RunConfig())
    assert resp.usage == {"input": 312, "output": 18}
    assert resp.model == "gpt-4o-mini"


def test_decode_without_usage_leaves_fields_none():
    msg = SimpleNamespace(content="ans", tool_calls=None, reasoning=None, reasoning_content=None)
    client, _ = _client_returning(msg, "stop")
    resp = OpenAIChat(model="gpt-4o-mini", client=client).invoke(
        [{"role": "user", "content": "hi"}], RunConfig()
    )
    assert resp.usage is None
    assert resp.model == "gpt-4o-mini"


def test_maps_plain_stop_message():
    msg = SimpleNamespace(content="hello", tool_calls=None, reasoning=None, reasoning_content=None)
    client, _ = _client_returning(msg, "stop")
    resp = OpenAIChat(model="gpt-4o-mini", client=client).invoke(
        [{"role": "user", "content": "hi"}],
        RunConfig(),
    )
    assert resp.finish_reason == "stop" and resp.message == {"role": "assistant", "content": "hello"}


def test_maps_tool_calls_message_to_wire_dict():
    tc = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path":"x"}'),
    )
    msg = SimpleNamespace(content=None, tool_calls=[tc], reasoning=None, reasoning_content=None)
    client, comp = _client_returning(msg, "tool_calls")
    resp = OpenAIChat("gpt-4o-mini", client=client, tools=[{"type": "function"}]).invoke(
        [{"role": "user", "content": "hi"}],
        RunConfig(),
    )
    assert resp.finish_reason == "tool_calls"
    assert resp.message["tool_calls"][0]["function"]["name"] == "read_file"
    assert comp.last_kwargs["tools"] == [{"type": "function"}]


def test_decode_extracts_reasoning():
    msg = SimpleNamespace(content="ans", tool_calls=None, reasoning=None, reasoning_content="because")
    client, _ = _client_returning(msg, "stop")
    resp = OpenAIChat(client=client).invoke([{"role": "user", "content": "hi"}], RunConfig())
    assert resp.message["reasoning"] == "because"


def test_encode_strips_reasoning_before_wire():
    msg = SimpleNamespace(content="x", tool_calls=None, reasoning=None, reasoning_content=None)
    client, comp = _client_returning(msg, "stop")
    OpenAIChat(client=client).invoke(
        [{"role": "assistant", "content": "prev", "reasoning": "SECRET"}],
        RunConfig(),
    )
    assert all("reasoning" not in m for m in comp.last_kwargs["messages"])


def test_loads_openai_api_key_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY='test-key'\n", encoding="utf-8")

    _load_openai_api_key(env_path)

    assert os.environ["OPENAI_API_KEY"] == "test-key"


def test_default_env_path_is_project_root():
    assert ENV == Path(__file__).resolve().parents[2] / ".env"


def test_existing_openai_api_key_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY='file-key'\n", encoding="utf-8")

    _load_openai_api_key(env_path)

    assert os.environ["OPENAI_API_KEY"] == "existing-key"
