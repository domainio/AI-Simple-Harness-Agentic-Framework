import io
import json

from core.structured_tracer import StructuredTracer
from core.summarizing_manager import SummarizingContextManager
from context_layer.manager import ContextItem, ContextManager
from context_layer.tokenizer import TiktokenCounter


class FakeSummarizer:
    def __init__(self):
        self.calls = 0

    def extend(self, prior, evicted):
        self.calls += 1
        return ContextItem(
            type="summary",
            message={"role": "system", "content": f"sum{self.calls}"},
            priority=950,
            pinned=True,
        )


def _transcript():
    return [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "old fact " * 20},
        {"role": "user", "content": "current"},
    ]


def test_on_assembly_hook_emits_record_on_any_manager():
    buf = io.StringIO()
    tracer = StructuredTracer(stream=buf)
    cm = ContextManager(
        TiktokenCounter("gpt-4o-mini"),
        budget=40,
        truncate_cap=20,
        on_assembly=tracer.on_assembly,
    )

    out = cm.render(_transcript())

    assert all("role" in msg for msg in out)
    recs = [json.loads(line) for line in buf.getvalue().splitlines()]
    assembly = [rec for rec in recs if rec["kind"] == "assembly"]
    assert assembly
    assert assembly[0]["budget"] == 40
    assert "evicted" in assembly[0]
    assert "tokens" in assembly[0]


def test_compose_summarizing_with_assembly_hook_no_multiple_inheritance():
    buf = io.StringIO()
    summarizer = FakeSummarizer()
    tracer = StructuredTracer(stream=buf)
    cm = SummarizingContextManager(
        TiktokenCounter("gpt-4o-mini"),
        budget=40,
        truncate_cap=20,
        summarizer=summarizer,
        summary_reserve=10,
        on_assembly=tracer.on_assembly,
    )

    out = cm.render(_transcript())

    assert any("sum" in (msg.get("content") or "") for msg in out)
    recs = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert any(rec["kind"] == "assembly" for rec in recs)
    assert summarizer.calls == 1
