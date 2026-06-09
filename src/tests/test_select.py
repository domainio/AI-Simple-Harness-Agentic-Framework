from context_layer.manager import ContextItem, ContextManager, ContextOverflow, Turn
from context_layer.tokenizer import TiktokenCounter


def _cm(budget):
    return ContextManager(tokenizer=TiktokenCounter("gpt-4o-mini"), budget=budget, truncate_cap=20)


def _item(role, content, prio, **kw):
    return ContextItem(type=role, message={"role": role, "content": content}, priority=prio, **kw)


def _price(cm, *items):
    for it in items:
        it.token_cost = cm.tokenizer.count(it.message)


def test_group_turns_keeps_assistant_with_its_tools():
    cm = _cm(1000)
    a = ContextItem(
        type="assistant",
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        priority=501,
    )
    t = ContextItem(
        type="tool",
        message={"role": "tool", "tool_call_id": "c1", "content": "r"},
        priority=502,
    )
    u = _item("user", "hi", 900)
    turns = cm._group_turns([a, t, u])
    sizes = sorted(len(turn.items) for turn in turns)
    assert sizes == [1, 2]
    pair = [turn for turn in turns if len(turn.items) == 2][0]
    assert pair.rank == 502


def test_truncate_shrinks_oversized_truncatable_item():
    cm = _cm(1000)
    big = _item("tool", "x " * 500, 500, truncatable=True)
    _price(cm, big)
    assert big.token_cost > cm.truncate_cap
    cm._truncate(big)
    assert big.token_cost <= cm.truncate_cap
    assert "[truncated]" in big.message["content"]


def test_over_budget_keeps_highest_priority_turns():
    cm = _cm(20)
    sysit = _item("system", "S", 1000, pinned=True)
    old = _item("user", "old " * 40, 901)
    new = _item("user", "new", 950)
    _price(cm, sysit, old, new)
    kept = cm._select([sysit, old, new])
    contents = [k.message["content"] for k in kept]
    assert contents[0] == "S"
    assert any("new" in c for c in contents)
    assert not any("old" in c for c in contents)
    assert sum(k.token_cost for k in kept) <= cm.budget


def test_recency_breaks_ties_among_equal_priority_turns():
    cm = _cm(10)
    sysit = _item("system", "S", 1000, pinned=True)
    older = _item("user", "older", 900)
    newer = _item("user", "newer", 900)
    _price(cm, sysit, older, newer)
    cm.budget = sysit.token_cost + max(older.token_cost, newer.token_cost)
    kept = cm._select([sysit, older, newer])
    contents = " ".join(k.message["content"] for k in kept)
    assert "newer" in contents and "older" not in contents


def test_sdk_system_message_assembled_first():
    cm = _cm(10_000)
    doc = ContextItem(type="doc", message={"role": "system", "content": "HANDBOOK"}, priority=400)
    sysit = _item("system", "SYS", 1000, pinned=True)
    task = _item("user", "task", 950, pinned=True)
    _price(cm, doc, sysit, task)
    kept = cm._select([doc, sysit, task])
    assert kept[0].type == "system"
    assert kept[0].message["content"] == "SYS"


def test_system_and_current_task_never_dropped():
    cm = _cm(15)
    sysit = _item("system", "S", 1000, pinned=True)
    task = _item("user", "current task", 950, pinned=True)
    filler = _item("assistant", "filler " * 50, 500)
    _price(cm, sysit, task, filler)
    kept = cm._select([sysit, task, filler])
    types = {k.type for k in kept}
    assert "system" in types and "user" in types
    assert filler not in kept


def test_tool_pairing_preserved_on_eviction():
    cm = _cm(40)
    sysit = _item("system", "S", 1000, pinned=True)
    a = ContextItem(
        type="assistant",
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        priority=501,
    )
    tl = ContextItem(
        type="tool",
        message={"role": "tool", "tool_call_id": "c1", "content": "huge " * 40},
        priority=502,
    )
    _price(cm, sysit, a, tl)
    kept = cm._select([sysit, a, tl])
    has_assistant = any(k.message.get("tool_calls") for k in kept)
    has_tool = any(k.type == "tool" for k in kept)
    assert has_assistant == has_tool


def test_truncatable_shrinks_before_evict():
    cm = _cm(80)
    sysit = _item("system", "S", 1000, pinned=True)
    big = _item("tool", "data " * 300, 500, truncatable=True)
    _price(cm, sysit, big)
    kept = cm._select([sysit, big])
    tool_kept = [k for k in kept if k.type == "tool"]
    assert tool_kept and "[truncated]" in tool_kept[0].message["content"]


def test_per_item_priority_override_wins():
    cm = _cm(40)
    sysit = _item("system", "S", 1000, pinned=True)
    low = _item("tool", "low " * 30, 500, truncatable=True)
    high = _item("tool", "high " * 30, 999, truncatable=True)
    _price(cm, sysit, low, high)
    kept = cm._select([sysit, low, high])
    kept_contents = " ".join(k.message["content"] for k in kept)
    assert "high" in kept_contents and "low" not in kept_contents


def test_open_closed_new_type_needs_no_core_change():
    cm = _cm(10_000)
    doc = ContextItem(
        type="retrieved_doc",
        message={"role": "system", "content": "retrieved fact"},
        priority=700,
        truncatable=True,
    )
    _price(cm, doc)
    kept = cm._select([doc])
    assert doc in kept


def test_on_evict_extra_cannot_exceed_final_budget():
    class ExtraManager(ContextManager):
        def _on_evict(self, evicted):
            extra = _item("summary", "summary " * 50, 950)
            extra.token_cost = self.tokenizer.count(extra.message)
            return [extra]

    cm = ExtraManager(tokenizer=TiktokenCounter("gpt-4o-mini"), budget=20, truncate_cap=20)
    sysit = _item("system", "S", 1000, pinned=True)
    old = _item("user", "old " * 40, 901)
    _price(cm, sysit, old)

    try:
        cm._select([sysit, old])
        assert False, "expected ContextOverflow"
    except ContextOverflow:
        pass


def test_on_evict_extra_uses_evicted_turn_position():
    class ExtraManager(ContextManager):
        def _on_evict(self, evicted):
            extra = _item("memory", "compressed old", 600)
            extra.token_cost = self.tokenizer.count(extra.message)
            return [extra]

    cm = ExtraManager(tokenizer=TiktokenCounter("gpt-4o-mini"), budget=20, truncate_cap=20)
    sysit = _item("system", "S", 1000, pinned=True)
    old = _item("user", "old " * 40, 901)
    current = _item("user", "current", 950, pinned=True)
    _price(cm, sysit, old, current)

    kept = cm._select([sysit, old, current])
    contents = [item.message["content"] for item in kept]

    assert contents.index("compressed old") < contents.index("current")


def test_fail_fast_when_pins_exceed_budget():
    cm = _cm(3)
    sysit = _item("system", "way too long " * 20, 1000, pinned=True)
    _price(cm, sysit)
    try:
        cm._select([sysit])
        assert False, "expected ContextOverflow"
    except ContextOverflow:
        pass


def test_last_stats_recorded():
    cm = _cm(60)
    sysit = _item("system", "S", 1000, pinned=True)
    old = _item("user", "old " * 10, 901)
    _price(cm, sysit, old)
    cm._select([sysit, old])
    assert set(cm.last_stats) == {"kept", "evicted", "truncated", "tokens", "budget"}
    assert cm.last_stats["budget"] == 60


def test_turn_cost_sums_member_costs():
    cm = _cm(1000)
    a = _item("assistant", "a", 1)
    b = _item("tool", "b", 2)
    _price(cm, a, b)
    turn = Turn([a, b])
    assert turn.cost == a.token_cost + b.token_cost
