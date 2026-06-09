from context_layer.manager import ContextItem, ContextManager
from context_layer.tokenizer import TiktokenCounter


def _cm():
    return ContextManager(tokenizer=TiktokenCounter("gpt-4o-mini"), budget=10_000)


def test_context_item_defaults():
    it = ContextItem(type="doc", message={"role": "system", "content": "x"}, priority=40)
    assert it.token_cost == 0 and it.pinned is False and it.truncatable is False


def test_register_costs_item_once():
    cm = _cm()
    it = ContextItem(
        type="doc",
        message={"role": "system", "content": "hello world"},
        priority=40,
        truncatable=True,
    )
    cm.register(it)
    assert it.token_cost > 4
    assert it.source_id == id(it.message)
    assert cm.registered == [it]


def test_registered_truncatable_item_is_cached_after_truncation():
    class CountingTokenizer:
        def __init__(self):
            self.calls = 0

        def count(self, msg):
            self.calls += 1
            return len(msg.get("content") or "")

    tok = CountingTokenizer()
    cm = ContextManager(tokenizer=tok, budget=10_000, truncate_cap=20)
    doc = ContextItem(
        type="doc",
        message={"role": "system", "content": "x" * 100},
        priority=40,
        truncatable=True,
    )

    cm.register(doc)
    calls_after_register = tok.calls
    cm.render([])
    cm.render([])

    assert doc.token_cost <= cm.truncate_cap
    assert tok.calls == calls_after_register


def test_ingest_maps_roles_and_pins_current_task():
    cm = _cm()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "current"},
    ]
    items = cm._ingest(msgs)
    by_type = {it.type: it for it in items}
    assert by_type["system"].pinned is True
    users = [it for it in items if it.type == "user"]
    assert users[0].priority == users[1].priority == 900
    assert users[1].pinned is True and users[0].pinned is False
    assert by_type["assistant"].pinned is False


def test_ingest_costs_each_item():
    cm = _cm()
    msg = {"role": "user", "content": "hello world"}
    items = cm._ingest([msg])
    assert items[0].token_cost > 4
    assert items[0].source_id == id(msg)


def test_ingest_caches_token_cost_per_message():
    class CountingTokenizer:
        def __init__(self):
            self.calls = 0

        def count(self, msg):
            self.calls += 1
            return 5

    tok = CountingTokenizer()
    cm = ContextManager(tokenizer=tok, budget=10_000)
    msgs = [{"role": "user", "content": "hello"}]
    cm._ingest(msgs)
    cm._ingest(msgs)
    assert tok.calls == 1


def test_ingest_unknown_role_has_explicit_error():
    cm = _cm()
    try:
        cm._ingest([{"role": "future_role", "content": "x"}])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unsupported message role" in str(exc)
