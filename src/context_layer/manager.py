from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from context_layer.context_policy import DEFAULT_POLICY, ContextType, TypePolicy

if TYPE_CHECKING:
    from agent_sdk.types import Message
    from context_layer.tokenizer import Tokenizer


class ContextOverflow(Exception):
    """Mandatory pinned context exceeds the token budget."""


@dataclass
class ContextItem:
    type: ContextType | str
    message: "Message"
    priority: int
    token_cost: int = 0
    pinned: bool = False
    truncatable: bool = False
    source_id: int | None = None


@dataclass
class Turn:
    items: list[ContextItem] = field(default_factory=list)

    @property
    def cost(self) -> int:
        return sum(i.token_cost for i in self.items)

    @property
    def rank(self) -> int:
        return max(i.priority for i in self.items)


def _last_user_index(messages: "list[Message]") -> int:
    idx = -1
    for i, m in enumerate(messages):
        if m["role"] == "user":
            idx = i
    return idx


class ContextManager:
    def __init__(
        self,
        tokenizer: "Tokenizer",
        budget: int,
        policy: dict[ContextType | str, TypePolicy] | None = None,
        truncate_cap: int = 2_000,
        on_assembly=None,
    ):
        self.tokenizer = tokenizer
        self.budget = budget
        self.policy = DEFAULT_POLICY if policy is None else policy
        self.truncate_cap = truncate_cap
        self.on_assembly = on_assembly
        self.registered: list[ContextItem] = []
        self.last_stats: dict = {}
        self._cost: dict[int, int] = {}

    def register(self, item: ContextItem) -> None:
        item.token_cost = self.tokenizer.count(item.message)
        item.source_id = id(item.message)
        if item.truncatable and item.token_cost > self.truncate_cap:
            self._truncate(item)
        self.registered.append(item)

    def render(self, messages: "list[Message]") -> "list[Message]":
        registered = [replace(it) for it in self.registered]
        items = registered + self._ingest(messages)
        out = [it.message for it in self._select(items)]
        if self.on_assembly:
            self.on_assembly(**self.last_stats)
        return out

    def _ingest(self, messages: "list[Message]") -> list[ContextItem]:
        items: list[ContextItem] = []
        last_user = _last_user_index(messages)
        for i, m in enumerate(messages):
            role = m["role"]
            p = self.policy.get(role)
            if p is None:
                raise ValueError(f"unsupported message role {role!r}; add a TypePolicy")
            items.append(
                ContextItem(
                    type=role,
                    message=m,
                    priority=p.base_priority,
                    token_cost=self._count_cached(m),
                    pinned=p.pinned or (role == "user" and i == last_user),
                    truncatable=p.truncatable,
                    source_id=id(m),
                )
            )
        return items

    def _count_cached(self, m: "Message") -> int:
        key = id(m)
        if key not in self._cost:
            self._cost[key] = self.tokenizer.count(m)
        return self._cost[key]

    def _group_turns(self, items: list[ContextItem]) -> list[Turn]:
        turns: list[Turn] = []
        current: Turn | None = None
        for it in items:
            if it.message.get("tool_calls"):
                current = Turn([it])
                turns.append(current)
            elif it.type == ContextType.TOOL and current is not None:
                current.items.append(it)
            else:
                turns.append(Turn([it]))
                current = None
        return turns

    def _truncate(self, item: ContextItem) -> None:
        content = item.message.get("content") or ""
        marker = "\n...[truncated]"
        if item.token_cost <= self.truncate_cap or not content:
            return

        fast_truncate = getattr(self.tokenizer, "truncate_content", None)
        if fast_truncate:
            best_content, best_cost = fast_truncate(item.message, self.truncate_cap, marker)
            item.message = {**item.message, "content": best_content}
            item.token_cost = best_cost
            return

        lo = 0
        hi = len(content)
        best_content = marker
        best_cost = self.tokenizer.count({**item.message, "content": marker})
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = content[:mid] + marker
            msg = {**item.message, "content": candidate}
            cost = self.tokenizer.count(msg)
            if cost <= self.truncate_cap:
                best_content = candidate
                best_cost = cost
                lo = mid + 1
            else:
                hi = mid - 1

        item.message = {**item.message, "content": best_content}
        item.token_cost = best_cost

    def _on_evict(self, evicted: list[Turn]) -> list[ContextItem]:
        return []

    def _eviction_reserve(self) -> int:
        return 0

    def _select(self, items: list[ContextItem]) -> list[ContextItem]:
        order = {id(it): n for n, it in enumerate(items)}

        pinned = [it for it in items if it.pinned]
        pinned_cost = sum(it.token_cost for it in pinned)
        if pinned_cost > self.budget:
            raise ContextOverflow(f"pinned items cost {pinned_cost} > budget {self.budget}")

        rest = [it for it in items if not it.pinned]
        truncated = 0
        for it in rest:
            if it.truncatable and it.token_cost > self.truncate_cap:
                self._truncate(it)
                truncated += 1

        turns = self._group_turns(rest)
        budget_left = max(0, self.budget - self._eviction_reserve() - pinned_cost)
        kept_turns: list[Turn] = []
        total = 0

        def turn_key(turn: Turn) -> tuple[int, int]:
            recency = max(order[id(i)] for i in turn.items)
            return (turn.rank, recency)

        for turn in sorted(turns, key=turn_key, reverse=True):
            if total + turn.cost <= budget_left:
                kept_turns.append(turn)
                total += turn.cost

        kept_turn_ids = {id(t) for t in kept_turns}
        evicted = [t for t in turns if id(t) not in kept_turn_ids]

        extra = self._on_evict(evicted)
        extra_order = min((order[id(i)] for t in evicted for i in t.items), default=len(order))
        for offset, it in enumerate(extra):
            if it.token_cost == 0:
                it.token_cost = self.tokenizer.count(it.message)
            order.setdefault(id(it), extra_order + offset)

        kept_items = pinned + extra + [i for t in kept_turns for i in t.items]
        kept_items.sort(key=lambda it: (self._sort_group(it), order[id(it)]))
        token_total = sum(it.token_cost for it in kept_items)
        if token_total > self.budget:
            raise ContextOverflow(f"selected items cost {token_total} > budget {self.budget}")

        self.last_stats = {
            "kept": len(kept_items),
            "evicted": len(evicted),
            "truncated": truncated,
            "tokens": token_total,
            "budget": self.budget,
        }
        return kept_items

    def _sort_group(self, item: ContextItem) -> int:
        if item.type == ContextType.SYSTEM:
            return 0
        if item.type == ContextType.SUMMARY:
            return 1
        if item.message.get("role") == "system":
            return 2
        return 3
