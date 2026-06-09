from __future__ import annotations

from context_layer.manager import ContextItem, ContextManager, Turn


def _key(item: ContextItem) -> int:
    return item.source_id if item.source_id is not None else id(item.message)


class SummarizingContextManager(ContextManager):
    """Compress evicted turns by overriding only the context-layer eviction hook."""

    def __init__(self, *args, summarizer, summary_reserve: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.summarizer = summarizer
        self.summary_reserve = summary_reserve
        self._summary: ContextItem | None = None
        self._summarized_ids: set[int] = set()

    def _eviction_reserve(self) -> int:
        return self.summary_reserve

    def _on_evict(self, evicted: list[Turn]) -> list[ContextItem]:
        new_turns: list[Turn] = []
        for turn in evicted:
            fresh = [item for item in turn.items if _key(item) not in self._summarized_ids]
            if fresh:
                new_turns.append(Turn(fresh))

        if new_turns:
            self._summary = self._cap(self.summarizer.extend(self._summary, new_turns))
            for turn in new_turns:
                for item in turn.items:
                    self._summarized_ids.add(_key(item))

        return [self._summary] if self._summary else []

    def _cap(self, item: ContextItem) -> ContextItem:
        item.token_cost = self.tokenizer.count(item.message)
        if item.token_cost <= self.summary_reserve:
            return item

        content = item.message.get("content") or ""
        marker = "\n...[summary truncated]"
        lo = 0
        hi = len(content)
        best = ""
        best_cost = self.tokenizer.count({**item.message, "content": ""})
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = content[:mid] + marker
            cost = self.tokenizer.count({**item.message, "content": candidate})
            if cost <= self.summary_reserve:
                best = candidate
                best_cost = cost
                lo = mid + 1
            else:
                hi = mid - 1

        item.message = {**item.message, "content": best}
        item.token_cost = best_cost
        return item
