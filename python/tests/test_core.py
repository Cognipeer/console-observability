"""Core behaviour: configuration, identifiers, redaction, session delivery.

These tests assert the guarantees the README makes — never raise, never block,
absent-not-zero, secrets do not leave the process — because those are what make
the package safe to put in front of a production agent.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import pytest

import cognipeer_observability as cognipeer
from cognipeer_observability import _transport
from cognipeer_observability._config import resolve_config
from cognipeer_observability._ids import span_id_from, trace_id_from
from cognipeer_observability._redact import redact, sanitize_sections


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, Any]]:
    """Swap the wire for a list. Nothing in these tests touches the network."""
    sent: List[Tuple[str, Any]] = []
    monkeypatch.setattr(
        _transport.Transport,
        "_submit",
        lambda self, path, payload: sent.append((path, payload)),
    )
    monkeypatch.setenv("COGNIPEER_API_KEY", "cpeer_test_key_000000000000")
    cognipeer.reset_client()
    return sent


# ── Configuration ────────────────────────────────────────────────────────

def test_missing_api_key_disables_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COGNIPEER_API_KEY", raising=False)
    config = resolve_config()
    assert config.enabled is False


def test_base_url_accepts_the_legacy_api_path() -> None:
    config = resolve_config(api_key="k", base_url="https://console.acme.com/api/client/v1/")
    assert config.base_url == "https://console.acme.com"


def test_capture_none_disables_the_exporter() -> None:
    assert resolve_config(api_key="k", capture="none").enabled is False


# ── Identifiers ──────────────────────────────────────────────────────────

def test_span_ids_do_not_collide_for_uuid7_run_ids() -> None:
    # UUIDv7's first 16 hex digits are a millisecond timestamp plus 12 bits of
    # entropy, so truncating would collide for runs started in the same
    # millisecond — which is most of them.
    a = span_id_from("01a00571-cf48-7a60-8000-000000000001")
    b = span_id_from("01a00571-cf48-7a60-8000-000000000002")
    assert a != b
    assert re.fullmatch(r"[0-9a-f]{16}", a)


def test_span_id_is_deterministic_and_passes_through_real_span_ids() -> None:
    assert span_id_from("run-1") == span_id_from("run-1")
    assert span_id_from("00f067aa0ba902b7") == "00f067aa0ba902b7"
    assert trace_id_from("4bf92f3577b34da6a3ce929d0e0e4736") == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id_from("conversation-1"))


# ── Redaction ────────────────────────────────────────────────────────────

def test_redacts_credential_shapes() -> None:
    out = redact("use sk-abcdefghijklmnopqrstuvwx and AKIAIOSFODNN7EXAMPLE")
    assert "sk-abcdefghijklmnopqrstuvwx" not in out
    assert "AKIAIOSFODNN7EXAMPLE" not in out


def test_leaves_ordinary_prose_alone() -> None:
    prose = "The flight to Rome costs 240 EUR and departs at 09:15."
    assert redact(prose) == prose


def test_strips_base64_data_urls() -> None:
    payload = "data:image/png;base64," + ("A" * 500)
    out = redact(payload)
    assert "AAAA" not in out
    assert "stripped" in out


def test_metadata_capture_keeps_structure_but_drops_bodies() -> None:
    config = resolve_config(api_key="k", capture="metadata")
    sections = sanitize_sections(
        [{"kind": "message", "role": "user", "content": "secret question"}], config
    )
    assert sections is not None
    assert sections[0]["kind"] == "message"
    assert sections[0].get("content") is None


def test_content_is_capped() -> None:
    config = resolve_config(api_key="k", max_content_chars=100)
    sections = sanitize_sections([{"kind": "message", "content": "x" * 5000}], config)
    assert sections is not None
    assert len(str(sections[0]["content"])) < 300
    assert sections[0]["truncated"] is True


# ── Session delivery ─────────────────────────────────────────────────────

def test_batch_mode_sends_one_request_with_every_event(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="batch", agent={"name": "test-agent"})
    session = client.start_session(thread_id="conv-1")
    session.record({"type": "ai_call", "model": "gpt-4.1-mini", "inputTokens": 10, "outputTokens": 5})
    session.record({"type": "tool_call", "toolName": "search"})
    session.end()

    assert len(captured) == 1
    path, payload = captured[0]
    assert path == "/api/client/v1/tracing/sessions"
    assert payload["threadId"] == "conv-1"
    assert payload["agent"]["name"] == "test-agent"
    assert [event["type"] for event in payload["events"]] == ["ai_call", "tool_call"]
    assert payload["summary"]["totalInputTokens"] == 10
    assert payload["summary"]["totalOutputTokens"] == 5
    assert payload["summary"]["eventCounts"] == {"ai_call": 1, "tool_call": 1}


def test_batch_payload_merges_client_default_and_session_metadata(
    captured: List[Tuple[str, Any]],
) -> None:
    client = cognipeer.init(mode="batch", metadata={"env": "prod"})
    session = client.start_session(metadata={"complexity": "complex"})
    session.record({"type": "ai_call"})
    session.end()

    _, payload = captured[0]
    assert payload["metadata"] == {"env": "prod", "complexity": "complex"}


def test_stream_start_payload_carries_session_metadata(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="stream")
    session = client.start_session(metadata={"complexity": "simple"})
    session.record({"type": "ai_call"})
    session.end()

    start_payload = next(payload for path, payload in captured if path.endswith("/start"))
    assert start_payload["metadata"] == {"complexity": "simple"}


def test_stream_mode_opens_before_it_appends(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="stream")
    session = client.start_session()
    session.record({"type": "ai_call"})
    session.end()

    paths = [path for path, _ in captured]
    assert paths[0].endswith("/start")
    assert paths[1].endswith("/events")
    assert paths[-1].endswith("/end")


def test_stream_mode_delivers_each_event_as_it_happens(
    captured: List[Tuple[str, Any]],
) -> None:
    """Streaming must be live, not merely ordered.

    Buffering until ``end()`` would still deliver everything in the right
    order — and would make a long-running agent invisible in the Console until
    it finished, which is the whole reason to stream.
    """
    client = cognipeer.init(mode="stream")
    session = client.start_session()

    session.record({"type": "ai_call"})
    assert [path for path, _ in captured][-1].endswith("/events")

    session.record({"type": "tool_call"})
    assert len([path for path, _ in captured if path.endswith("/events")]) == 2

    session.end()


def test_spans_pair_start_and_end_into_one_event(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="batch")
    session = client.start_session()
    session.open_span("run-1", type="ai_call", label="gpt-4.1-mini", model="gpt-4.1-mini")
    session.open_span("run-2", type="tool_call", parent_key="run-1", tool_name="search")
    session.close_span("run-2", sections=[{"kind": "tool_result", "content": "ok"}])
    session.close_span("run-1", input_tokens=100, output_tokens=20, cached_input_tokens=64)
    session.end()

    events = captured[0][1]["events"]
    assert len(events) == 2
    tool, model_call = events
    assert tool["type"] == "tool_call"
    assert model_call["inputTokens"] == 100
    assert model_call["cachedInputTokens"] == 64
    # The tool span names the model span as its parent.
    assert tool["parentSpanId"] == model_call["spanId"]


def test_unreported_usage_is_absent_not_zero(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="batch")
    session = client.start_session()
    session.open_span("run-1", type="ai_call")
    session.close_span("run-1")
    session.end()

    event = captured[0][1]["events"][0]
    assert "inputTokens" not in event
    assert "outputTokens" not in event


def test_dangling_spans_are_closed_when_the_session_ends(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="batch")
    session = client.start_session()
    session.open_span("never-closed", type="tool_call")
    session.end()

    assert len(captured[0][1]["events"]) == 1


def test_error_events_mark_the_session_failed(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="batch")
    session = client.start_session()
    session.open_span("run-1", type="tool_call")
    session.close_span("run-1", error=ValueError("boom"))
    session.end()

    payload = captured[0][1]
    assert payload["status"] == "error"
    assert payload["errors"][0]["message"] == "boom"


def test_a_disabled_client_records_nothing_and_raises_nothing(
    monkeypatch: pytest.MonkeyPatch, captured: List[Tuple[str, Any]]
) -> None:
    monkeypatch.delenv("COGNIPEER_API_KEY", raising=False)
    cognipeer.reset_client()
    client = cognipeer.init()
    session = client.start_session()
    assert session.disabled is True
    session.record({"type": "ai_call"})
    session.end()
    assert captured == []


# ── Manual instrumentation ───────────────────────────────────────────────

def test_observe_nests_and_creates_a_root_session(captured: List[Tuple[str, Any]]) -> None:
    cognipeer.init(mode="batch")

    @cognipeer.observe(type="tool_call", tool_name="inner")
    def inner(value: int) -> int:
        return value * 2

    @cognipeer.observe(name="outer")
    def outer() -> int:
        return inner(21)

    assert outer() == 42

    events: List[Dict[str, Any]] = captured[0][1]["events"]
    by_label = {event["label"]: event for event in events}
    assert set(by_label) == {"outer", "inner"}
    assert by_label["inner"]["parentSpanId"] == by_label["outer"]["spanId"]
    assert by_label["inner"]["type"] == "tool_call"
    assert by_label["inner"]["toolName"] == "inner"


@pytest.mark.asyncio
async def test_observe_supports_async_functions(captured: List[Tuple[str, Any]]) -> None:
    cognipeer.init(mode="batch")

    @cognipeer.observe(name="fetch")
    async def fetch() -> str:
        return "done"

    assert await fetch() == "done"
    assert captured[0][1]["events"][0]["label"] == "fetch"


def test_trace_marks_the_session_failed_and_reraises(captured: List[Tuple[str, Any]]) -> None:
    cognipeer.init(mode="batch")

    with pytest.raises(RuntimeError):
        with cognipeer.trace(name="failing-agent"):
            raise RuntimeError("nope")

    payload = captured[0][1]
    assert payload["status"] == "error"
    assert payload["agent"]["name"] == "failing-agent"


# ── Generators ───────────────────────────────────────────────────────────

def test_generator_span_does_not_leak_into_the_caller(
    captured: List[Tuple[str, Any]],
) -> None:
    """A generator shares its caller's context.

    Binding its span across a ``yield`` would make the caller's next observed
    call a child of the generator, which is both wrong and invisible.
    """
    from cognipeer_observability.context import get_current_span_key

    cognipeer.init(mode="batch")

    @cognipeer.observe(name="stream_rows")
    def stream_rows() -> Any:
        yield 1
        yield 2
        yield 3

    @cognipeer.observe(name="sibling")
    def sibling() -> str:
        return "ok"

    with cognipeer.trace(name="gen-agent"):
        for row in stream_rows():
            assert get_current_span_key() is None
            if row == 2:
                break  # GeneratorExit at the yield
        sibling()

    events = {event["label"]: event for _, payload in captured for event in payload["events"]}
    # An early `break` is an ordinary exit, not a failure.
    assert events["stream_rows"]["status"] == "success"
    assert events["stream_rows"]["metadata"]["abandoned"] is True
    assert events["sibling"]["parentSpanId"] != events["stream_rows"]["spanId"]


def test_trace_does_not_inherit_an_enclosing_span(captured: List[Tuple[str, Any]]) -> None:
    """A new session must not parent onto a span from a different session."""
    cognipeer.init(mode="batch")

    @cognipeer.observe(name="inner")
    def inner() -> str:
        return "ok"

    @cognipeer.observe(name="outer")
    def outer() -> None:
        with cognipeer.trace(name="nested-agent"):
            inner()

    outer()

    by_agent = {payload["agent"].get("name"): payload for _, payload in captured}
    nested = by_agent["nested-agent"]
    inner_event = next(e for e in nested["events"] if e["label"] == "inner")
    assert inner_event["parentSpanId"] == nested["rootSpanId"]


# ── Hostile payloads and the metadata channel ────────────────────────────

class _Hostile:
    """A value whose own code raises — the traced application's, not ours."""

    __slots__ = ()

    def __str__(self) -> str:
        raise RuntimeError("nope")

    def __repr__(self) -> str:
        raise RuntimeError("nope")


def test_stringify_never_raises_on_a_hostile_value() -> None:
    from cognipeer_observability._redact import UNSERIALIZABLE, stringify

    assert stringify(_Hostile()) == f'"{UNSERIALIZABLE}"'
    assert UNSERIALIZABLE in stringify([_Hostile()])
    assert UNSERIALIZABLE in stringify({"k": _Hostile()})


def test_recording_a_hostile_payload_does_not_raise(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="batch")
    session = client.start_session()

    session.record(
        {
            "type": "ai_call",
            "sections": [{"kind": "message", "content": _Hostile()}],
            "metadata": {"hostile": _Hostile()},
        }
    )
    session.end()

    assert len(captured) == 1


def test_event_metadata_is_redacted_and_capped(captured: List[Tuple[str, Any]]) -> None:
    """Metadata is the other channel by which caller data reaches the wire."""
    client = cognipeer.init(mode="batch", max_content_chars=200)
    session = client.start_session()
    session.record(
        {
            "type": "ai_call",
            "metadata": {
                "apiKey": "sk-abcdefghijklmnopqrstuvwx",
                "nested": {"token": "cpeer_abcdefghijklmnopqrstuv"},
                "blob": "x" * 50_000,
            },
        }
    )
    session.end()

    wire = json.dumps(captured[0][1])
    assert "sk-abcdefghijklmnopqrstuvwx" not in wire
    assert "cpeer_abcdefghijklmnopqrstuv" not in wire
    assert "[redacted]" in wire
    assert len(wire) < 10_000


def test_metadata_structure_is_preserved(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="batch")
    session = client.start_session()
    session.record(
        {"type": "span", "metadata": {"langgraph_node": "agent", "step": 3, "nested": {"ok": True}}}
    )
    session.end()

    event = captured[0][1]["events"][0]
    assert event["metadata"] == {"langgraph_node": "agent", "step": 3, "nested": {"ok": True}}


def test_session_config_is_redacted(captured: List[Tuple[str, Any]]) -> None:
    client = cognipeer.init(mode="batch")
    session = client.start_session(
        session_config={"systemPrompt": "internal key sk-abcdefghijklmnopqrstuvwx"}
    )
    session.record({"type": "ai_call"})
    session.end()

    wire = json.dumps(captured[0][1])
    assert "sk-abcdefghijklmnopqrstuvwx" not in wire
    assert "[redacted]" in wire


def test_a_raising_on_error_handler_does_not_escape() -> None:
    """`on_error` is arbitrary user code and must not reach the application."""
    from cognipeer_observability._config import resolve_config

    def explode(_error: BaseException) -> None:
        raise RuntimeError("sentry not initialised")

    config = resolve_config(api_key="k", on_error=explode)
    config.report_error(ValueError("original"))  # must not raise
