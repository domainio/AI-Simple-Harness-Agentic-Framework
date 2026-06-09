from core.summarizing_manager import SummarizingContextManager
from context_layer.manager import ContextItem, ContextManager, Turn
from context_layer.tokenizer import TiktokenCounter


class FakeSummarizer:
    def __init__(self):
        self.calls = 0

    def extend(self, prior, evicted):
        self.calls += 1
        prior_text = prior.message["content"] if prior else ""
        return ContextItem(
            type="summary",
            message={"role": "system", "content": prior_text + f"|sum{self.calls}"},
            priority=950,
            pinned=True,
        )


def _item(role, content, prio, **kw):
    return ContextItem(type=role, message={"role": role, "content": content}, priority=prio, **kw)


def _items(cm):
    sysit = _item("system", "S", 1000, pinned=True)
    old = _item("user", "old fact " * 20, 901)
    new = _item("user", "current", 950, pinned=True)
    for it in (sysit, old, new):
        it.token_cost = cm.tokenizer.count(it.message)
    return [sysit, old, new]


def test_evicted_turns_are_summarized_and_reinjected():
    summarizer = FakeSummarizer()
    cm = SummarizingContextManager(
        TiktokenCounter("gpt-4o-mini"),
        budget=40,
        truncate_cap=20,
        summarizer=summarizer,
        summary_reserve=10,
    )

    kept = cm._select(_items(cm))

    assert summarizer.calls == 1
    summaries = [item for item in kept if item.type == "summary"]
    assert summaries
    assert summaries[0].pinned is True


def test_summary_is_placed_before_recent_turns():
    summarizer = FakeSummarizer()
    cm = SummarizingContextManager(
        TiktokenCounter("gpt-4o-mini"),
        budget=40,
        truncate_cap=20,
        summarizer=summarizer,
        summary_reserve=10,
    )

    kept = cm._select(_items(cm))
    types = [item.type for item in kept]

    assert types.index("system") < types.index("summary") < types.index("user")


def test_summary_is_cached_not_recomputed_for_same_turns():
    summarizer = FakeSummarizer()
    cm = SummarizingContextManager(
        TiktokenCounter("gpt-4o-mini"),
        budget=40,
        truncate_cap=20,
        summarizer=summarizer,
        summary_reserve=10,
    )
    items = _items(cm)

    cm._select(items)
    cm._select(items)

    assert summarizer.calls == 1


def test_summary_reserve_does_not_reduce_final_budget_guard():
    class CostTokenizer:
        def count(self, msg):
            return int(msg["content"])

    class FixedSummarizer:
        def extend(self, prior, evicted):
            return ContextItem(
                type="summary",
                message={"role": "system", "content": "350"},
                priority=950,
                pinned=True,
            )

    def item(role, cost, priority, pinned=False):
        return ContextItem(
            type=role,
            message={"role": role, "content": str(cost)},
            priority=priority,
            pinned=pinned,
            token_cost=cost,
        )

    cm = SummarizingContextManager(
        CostTokenizer(),
        budget=1000,
        summarizer=FixedSummarizer(),
        summary_reserve=512,
    )

    kept = cm._select(
        [
            item("system", 75, 1000, True),
            item("user", 300, 900),
            item("assistant", 100, 800),
            item("user", 75, 950, True),
        ]
    )

    assert sum(item.token_cost for item in kept) == 800
    assert cm.last_stats["budget"] == 1000


def test_base_manager_drops_without_summarizing():
    cm = ContextManager(TiktokenCounter("gpt-4o-mini"), budget=40, truncate_cap=20)
    kept = cm._select(_items(cm))

    assert not any(item.type == "summary" for item in kept)
