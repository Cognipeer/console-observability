"""OpenTelemetry normalizer and exporter.

Two things are pinned here. First, the mapping itself — three conventions, one
event model. Second, and less obvious: **the two SDKs must agree**. A run traced
from Python and the same run traced from TypeScript have to produce the same
event, so several tests below assert a specific shape purely because the JS
normalizer produces that shape.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

import pytest

import cognipeer_observability as cognipeer
from cognipeer_observability import _transport
from cognipeer_observability.otel import normalize as n

pytest.importorskip("opentelemetry.sdk.trace")

from cognipeer_observability.otel import CognipeerSpanExporter  # noqa: E402


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, Any]]:
    sent: List[Tuple[str, Any]] = []
    monkeypatch.setattr(
        _transport.Transport,
        "_submit",
        lambda self, path, payload: sent.append((path, payload)),
    )
    monkeypatch.setenv("COGNIPEER_API_KEY", "cpeer_test_key_000000000000")
    # A closed port, so that anything escaping the patch — the atexit hook ends
    # sessions after the fixture is torn down — cannot reach a real Console.
    monkeypatch.setenv("COGNIPEER_BASE_URL", "http://127.0.0.1:9")
    cognipeer.reset_client()
    return sent


def span(name: str = "span", **attributes: Any) -> n.SpanData:
    return n.SpanData.from_mapping({"name": name, "attributes": attributes})


# ── Event type ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        # Both "this is a model call" signals are honoured, because the event
        # type gates token attribution: mistyping the span discards usage the
        # span explicitly reported. The JS normalizer applies the same union.
        ({"gen_ai.request.model": "gpt-4.1-mini"}, "ai_call"),
        ({"gen_ai.usage.input_tokens": 33, "gen_ai.usage.output_tokens": 4}, "ai_call"),
        # An unlisted operation that still looks like a model call.
        ({"gen_ai.operation.name": "chat.stream", "gen_ai.usage.input_tokens": 10}, "ai_call"),
        # A declared kind always wins.
        ({"openinference.span.kind": "TOOL"}, "tool_call"),
        ({"gen_ai.operation.name": "invoke_agent"}, "span"),
        ({}, "span"),
    ],
)
def test_event_type_inference(attributes: Dict[str, Any], expected: str) -> None:
    assert n.normalize_type(attributes, "span")[0] == expected


def test_tokens_ride_only_on_model_calls() -> None:
    """An agent span repeating its children's usage must not be counted again."""
    agent = n.normalize_span(
        span("agent", **{
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.usage.input_tokens": 5000,
        })
    )
    call = n.normalize_span(
        span("chat", **{
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "m",
            "gen_ai.usage.input_tokens": 1200,
        })
    )
    assert "inputTokens" not in agent.event
    assert call.event["inputTokens"] == 1200


def test_cache_write_is_not_folded_into_cached_tokens() -> None:
    usage = n.extract_usage(
        {
            "gen_ai.usage.input_tokens": 1000,
            "gen_ai.usage.cache_read.input_tokens": 800,
            "gen_ai.usage.cache_creation.input_tokens": 150,
        }
    )
    assert usage["cachedInputTokens"] == 800


# ── Sections ─────────────────────────────────────────────────────────────

def test_message_conventions_use_precedence_not_concatenation() -> None:
    """One span, two namespaces, one conversation — rendered once.

    `OPENINFERENCE_ENABLE_GENAI_SEMCONV=true` makes a span carry both; showing
    both would read as a doubled prompt.
    """
    sections = n.extract_sections(
        {
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "book me a flight",
            "gen_ai.input.messages": '[{"role":"user","parts":[{"type":"text","content":"book me a flight"}]}]',
        }
    )
    messages = [s for s in sections if s["kind"] == "message"]
    assert len(messages) == 1


def test_tool_span_renders_its_input_as_a_tool_call() -> None:
    """A TOOL span's `input.value` is the arguments, not a user message."""
    sections = n.extract_sections(
        {
            "openinference.span.kind": "TOOL",
            "tool.name": "search_flights",
            "input.value": '{"city":"Rome"}',
            "output.value": "3 flights",
        }
    )
    assert [s["kind"] for s in sections] == ["tool_call", "tool_result"]
    assert sections[0]["content"] == {"city": "Rome"}
    # Label parity with the JS normalizer.
    assert sections[1]["label"] == "Tool result"


def test_gen_ai_tool_name_alone_marks_a_tool_span() -> None:
    sections = n.extract_sections({"gen_ai.tool.name": "search", "input.value": "{}"})
    assert sections[0]["kind"] == "tool_call"


@pytest.mark.parametrize(
    "attributes",
    [
        {
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": n.REDACTED_SENTINEL,
        },
        {"gen_ai.operation.name": "chat", "gen_ai.input.messages": n.REDACTED_SENTINEL},
        {"gen_ai.prompt.0.role": "user", "gen_ai.prompt.0.content": n.REDACTED_SENTINEL},
    ],
)
def test_redaction_sentinel_is_dropped_not_rendered(attributes: Dict[str, Any]) -> None:
    """`__REDACTED__` is a marker, not content.

    Dropping matters beyond rendering: a section carrying the literal would be
    harvested into an evaluation dataset as though it were a real prompt. JS and
    the Console's own OTLP ingest drop it too.
    """
    assert n.extract_sections(attributes) == []


def test_truncated_json_keeps_the_event() -> None:
    """OTel truncates long attributes, so the JSON arrives unparseable."""
    normalized = n.normalize_span(
        span("chat", **{
            "gen_ai.operation.name": "chat",
            "gen_ai.usage.input_tokens": 42,
            "gen_ai.input.messages": '[{"role":"user","parts":[{"type":"text","content":"tru',
        })
    )
    assert normalized.event["type"] == "ai_call"
    assert normalized.event["inputTokens"] == 42


def test_indexed_attributes_sort_numerically() -> None:
    """They arrive flat and unordered; naive accumulation scrambles the turns."""
    sections = n.extract_sections(
        {
            "gen_ai.prompt.10.role": "user",
            "gen_ai.prompt.10.content": "eleventh",
            "gen_ai.prompt.2.role": "user",
            "gen_ai.prompt.2.content": "third",
        }
    )
    assert [s["content"] for s in sections] == ["third", "eleventh"]


def test_tool_span_label_falls_back_to_the_tool_name() -> None:
    normalized = n.normalize_span(
        span("tool.run", **{"openinference.span.kind": "TOOL", "tool.name": "flaky"})
    )
    assert normalized.event["label"] == "flaky"


# ── Exporter ─────────────────────────────────────────────────────────────

class _Span:
    """The subset of `ReadableSpan` the exporter reads."""

    def __init__(self, trace_id: int, span_id: int, parent: int | None = None, **attributes: Any):
        self.name = "span"
        self.attributes = attributes
        self.parent = type("P", (), {"span_id": parent})() if parent else None
        self.status = None
        self.events = ()
        self.resource = type("R", (), {"attributes": {"service.name": "svc"}})()
        self.start_time = 1_700_000_000_000_000_000
        self.end_time = 1_700_000_001_000_000_000
        self._ctx = type("C", (), {"trace_id": trace_id, "span_id": span_id})()

    def get_span_context(self) -> Any:
        return self._ctx

    @property
    def context(self) -> Any:
        return self._ctx


def test_concurrent_export_opens_one_session_per_trace(
    captured: List[Tuple[str, Any]],
) -> None:
    """Two threads, one trace, one session.

    The session id is derived from the trace id and streaming sessions POST
    `/start` on construction, so a session built outside the lock would be
    opened twice on the wire — and the loser's `end()` would then close the
    session the winner is still writing to.
    """
    cognipeer.init(mode="stream")
    exporter = CognipeerSpanExporter(session_idle_seconds=0)
    try:
        barrier = threading.Barrier(2)

        def export(span_id: int) -> None:
            barrier.wait()
            exporter.export([_Span(0xABC, span_id, parent=0xF00)])

        threads = [threading.Thread(target=export, args=(i + 1,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        starts = [path for path, _ in captured if path.endswith("/start")]
        ends = [path for path, _ in captured if path.endswith("/end")]
        assert len(starts) == 1
        assert ends == []
    finally:
        exporter.shutdown()


def test_open_traces_are_bounded(captured: List[Tuple[str, Any]]) -> None:
    """A producer that never emits root spans must not grow the map forever."""
    cognipeer.init(mode="stream")
    exporter = CognipeerSpanExporter(session_idle_seconds=0, max_open_traces=3)
    try:
        for trace in range(6):
            exporter.export([_Span(0x1000 + trace, 1, parent=0xF00)])

        # Three still open; the three oldest were closed to make room.
        assert len([path for path, _ in captured if path.endswith("/end")]) == 3
    finally:
        exporter.shutdown()
