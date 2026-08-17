"""A tracing session: the unit the Console shows as one agent run.

Sessions buffer events and choose how to deliver them (see ``mode`` in
:class:`~cognipeer_observability._config.Config`). The two delivery shapes are
not interchangeable server-side: the batch endpoint REPLACES a session's whole
event list, while the streaming endpoints append. A session therefore commits
to one shape the first time it delivers anything and never switches back.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from ._config import Config
from ._ids import new_session_id, new_span_id, new_trace_id, span_id_from
from ._redact import sanitize_metadata, sanitize_sections, stringify
from ._transport import Transport
from .types import Agent, Event, Section, Summary, ToolDefinition


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iso(value: Optional[datetime]) -> str:
    if value is None:
        return _now_iso()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_error(error: Any) -> Dict[str, Any]:
    """Turn anything a framework calls an "error" into a stable record."""
    if isinstance(error, BaseException):
        return {"message": str(error) or type(error).__name__, "type": type(error).__name__}
    if isinstance(error, str):
        return {"message": error}
    if isinstance(error, dict):
        message = error.get("message")
        out: Dict[str, Any] = {
            "message": message if isinstance(message, str) else stringify(error)[:2000]
        }
        if isinstance(error.get("type"), str):
            out["type"] = error["type"]
        return out
    return {"message": str(error)}


class _OpenSpan:
    """An open span.

    Duration comes from the monotonic clock when we opened the span ourselves,
    and from the wall clock when the caller supplied an explicit start time
    (a framework replaying a timestamp it recorded earlier).
    """

    __slots__ = ("init", "mono_start", "wall_start")

    def __init__(
        self, init: Dict[str, Any], wall_start: datetime, mono_start: Optional[float]
    ) -> None:
        self.init = init
        self.wall_start = wall_start
        self.mono_start = mono_start

    def elapsed_ms(self) -> int:
        if self.mono_start is not None:
            return int((time.monotonic() - self.mono_start) * 1000)
        return max(
            0, int((datetime.now(timezone.utc) - self.wall_start).total_seconds() * 1000)
        )


class TraceSession:
    """One agent run. Create with :meth:`Cognipeer.start_session`."""

    def __init__(
        self,
        config: Config,
        transport: Transport,
        on_close: Callable[["TraceSession"], None],
        *,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent: Optional[Agent] = None,
        metadata: Optional[Dict[str, str]] = None,
        session_config: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        root_span_id: Optional[str] = None,
        mode: Optional[str] = None,
        started_at: Optional[datetime] = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._on_close = on_close
        self._lock = threading.Lock()

        self.session_id = session_id or new_session_id()
        self.trace_id = trace_id or new_trace_id()
        self.root_span_id = root_span_id or new_span_id()
        self.thread_id = thread_id or config.thread_id

        self._agent: Agent = {**(config.agent or {}), **(agent or {})}  # type: ignore[dict-item]
        self._metadata: Dict[str, str] = {**(config.metadata or {}), **(metadata or {})}
        # Sanitized once: it is written verbatim by both the batch payload and
        # `/start`, and integrations put framework metadata (the Agents SDK's
        # `trace.metadata`, for one) straight into it.
        self._session_config = sanitize_metadata(session_config, config)
        self._mode = mode or config.mode
        self._started_at = started_at or datetime.now(timezone.utc)
        self._started_monotonic = time.monotonic()

        self._sequence = 0
        self._open: Dict[str, _OpenSpan] = {}
        self._all: List[Event] = []
        self._queue: List[Event] = []
        self._errors: List[Dict[str, Any]] = []
        self._summary: Dict[str, Any] = {
            "eventCounts": {},
            "totalInputTokens": 0,
            "totalOutputTokens": 0,
            "totalCachedInputTokens": 0,
            "totalDurationMs": 0,
        }
        self._streaming = False
        self._finished = False

        # `stream` mode announces the session immediately so it appears as
        # in-progress in the Console while the agent is still thinking.
        if self._config.enabled and self._mode == "stream":
            self._begin_stream()

    # ── Recording ────────────────────────────────────────────────────────
    @property
    def disabled(self) -> bool:
        """True when the transport is off — integrations can skip work."""
        return not self._config.enabled

    def record(self, event: Event) -> Optional[Event]:
        """Record a fully-formed event.

        Runs inside the traced application's own call stack, so it must not
        raise: a payload can carry a raising ``__str__`` or property, and an
        escaping exception would take the application down over a trace event.
        Sanitisation failures degrade the event instead.
        """
        if not self._config.enabled or self._finished:
            return None

        with self._lock:
            self._sequence += 1
            sequence = self._sequence

        normalized: Event = dict(event)  # type: ignore[assignment]
        normalized.setdefault("id", f"{self.session_id}-{sequence}")
        normalized["sequence"] = sequence
        normalized.setdefault("timestamp", _now_iso())
        normalized.setdefault("traceId", self.trace_id)
        normalized.setdefault("spanId", new_span_id())
        normalized.setdefault("parentSpanId", self.root_span_id)
        normalized.setdefault("status", "success")

        sections = self._guard(lambda: sanitize_sections(normalized.get("sections"), self._config))
        if sections:
            normalized["sections"] = sections
        else:
            normalized.pop("sections", None)

        # Metadata is the other channel by which caller data reaches the wire,
        # and it used to bypass redaction and the size cap entirely.
        metadata = self._guard(lambda: sanitize_metadata(normalized.get("metadata"), self._config))
        if metadata:
            normalized["metadata"] = metadata
        else:
            normalized.pop("metadata", None)

        self._accumulate(normalized)
        self._all.append(normalized)
        self._queue.append(normalized)
        self._maybe_stream()
        return normalized

    def open_span(
        self,
        key: str,
        *,
        type: str = "span",  # noqa: A002 - mirrors the wire field name
        label: Optional[str] = None,
        parent_key: Optional[str] = None,
        model: Optional[str] = None,
        actor: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        tool_execution_id: Optional[str] = None,
        sections: Optional[Sequence[Section]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tool_definitions: Optional[List[ToolDefinition]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
    ) -> None:
        """Open a span keyed by the framework's run id.

        Nothing is sent yet — the event is emitted on :meth:`close_span`, so
        one framework start/end pair becomes exactly one Console event carrying
        both sides.
        """
        if not self._config.enabled or self._finished:
            return
        wall_start = started_at or datetime.now(timezone.utc)
        if wall_start.tzinfo is None:
            wall_start = wall_start.replace(tzinfo=timezone.utc)
        self._open[key] = _OpenSpan(
            {
                "type": type,
                "label": label,
                "parent_key": parent_key,
                "model": model,
                "actor": actor,
                "tool_name": tool_name,
                "tool_execution_id": tool_execution_id,
                "sections": list(sections or []),
                "metadata": dict(metadata or {}),
                "tool_definitions": tool_definitions,
                "response_format": response_format,
                "wall_start": _iso(wall_start),
            },
            wall_start,
            None if started_at is not None else time.monotonic(),
        )

    def has_span(self, key: str) -> bool:
        """True when ``key`` has an open span — keeps integrations idempotent."""
        return key in self._open

    def update_span(self, key: str, **patch: Any) -> None:
        """Merge extra data into an already-open span."""
        span = self._open.get(key)
        if span is None:
            return
        sections = patch.pop("sections", None)
        metadata = patch.pop("metadata", None)
        if sections:
            span.init["sections"] = [*span.init.get("sections", []), *sections]
        if metadata:
            span.init["metadata"] = {**span.init.get("metadata", {}), **metadata}
        span.init.update({k: v for k, v in patch.items() if v is not None})

    def close_span(
        self,
        key: str,
        *,
        status: Optional[str] = None,
        label: Optional[str] = None,
        model: Optional[str] = None,
        sections: Optional[Sequence[Section]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tool_definitions: Optional[List[ToolDefinition]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cached_input_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        tool_name: Optional[str] = None,
        error: Any = None,
    ) -> Optional[Event]:
        """Close a span and emit the resulting event."""
        if not self._config.enabled or self._finished:
            return None

        span = self._open.pop(key, None)
        init: Dict[str, Any] = span.init if span else {}
        duration_ms = span.elapsed_ms() if span else 0
        error_record = normalize_error(error) if error is not None else None

        event: Event = {
            "id": key,
            "type": init.get("type") or "span",
            "spanId": span_id_from(key),
            "parentSpanId": (
                span_id_from(init["parent_key"]) if init.get("parent_key") else self.root_span_id
            ),
            "timestamp": init.get("wall_start") or _now_iso(),
            "durationMs": duration_ms,
            "status": "error" if error_record else (status or "success"),
        }
        label = label or init.get("label")
        if label:
            event["label"] = label
        model = model or init.get("model")
        if model:
            event["model"] = model
        if init.get("actor"):
            event["actor"] = init["actor"]
        tool_name = tool_name or init.get("tool_name")
        if tool_name:
            event["toolName"] = tool_name
        if init.get("tool_execution_id"):
            event["toolExecutionId"] = init["tool_execution_id"]
        tool_definitions = tool_definitions or init.get("tool_definitions")
        if tool_definitions:
            event["toolDefinitions"] = tool_definitions
        response_format = response_format or init.get("response_format")
        if response_format:
            event["responseFormat"] = response_format
        merged_sections = [*init.get("sections", []), *(sections or [])]
        if merged_sections:
            event["sections"] = merged_sections
        merged_metadata = {**init.get("metadata", {}), **(metadata or {})}
        if merged_metadata:
            event["metadata"] = merged_metadata
        if finish_reason:
            event["finishReason"] = finish_reason
        for field_name, value in (
            ("inputTokens", input_tokens),
            ("outputTokens", output_tokens),
            ("cachedInputTokens", cached_input_tokens),
            ("reasoningTokens", reasoning_tokens),
            ("totalTokens", total_tokens),
        ):
            if value is not None:
                event[field_name] = value  # type: ignore[literal-required]
        if error_record:
            event["error"] = error_record

        return self.record(event)

    # ── Delivery ─────────────────────────────────────────────────────────
    def end(self, *, status: Optional[str] = None, error: Any = None) -> None:
        """Close the session and deliver everything. Safe to call twice.

        Open spans are closed before the enabled check so the session's own
        bookkeeping ends in the same state either way — the JS implementation
        does the same, and a divergence here would show up as different event
        counts between the two SDKs for identical code.
        """
        if self._finished:
            return

        for key in list(self._open.keys()):
            self.close_span(key, status="error" if status == "error" else "success")
        if error is not None:
            self._errors.append(normalize_error(error))
        self._finished = True

        if not self._config.enabled:
            return

        ended_at = datetime.now(timezone.utc)
        final_status = status or ("error" if self._errors else "success")
        duration_ms = int((ended_at - self._started_at).total_seconds() * 1000)

        if self._streaming:
            self._flush_queued()
            self._transport.end_stream(
                self.session_id,
                {
                    "status": final_status,
                    "endedAt": _iso(ended_at),
                    "durationMs": duration_ms,
                    "summary": dict(self._summary),
                    "errors": self._errors,
                },
            )
        else:
            events = list(self._all)
            self._queue = []
            payload: Dict[str, Any] = {
                "sessionId": self.session_id,
                "traceId": self.trace_id,
                "rootSpanId": self.root_span_id,
                "agent": self._agent,
                "status": final_status,
                "startedAt": _iso(self._started_at),
                "endedAt": _iso(ended_at),
                "durationMs": duration_ms,
                "summary": dict(self._summary),
                "errors": self._errors,
                "events": events,
            }
            if self.thread_id:
                payload["threadId"] = self.thread_id
            if self._session_config:
                payload["config"] = self._session_config
            if self._metadata:
                payload["metadata"] = self._metadata
            self._transport.ingest_session(payload)

        self._on_close(self)

    def set_thread_id(self, thread_id: Optional[str]) -> None:
        """Attach a conversation id discovered after the session opened.

        Needed by the OpenTelemetry exporter: a trace's root span ends LAST, so
        the span carrying ``session.id`` routinely arrives after child spans
        have already opened the session. Without this, every OTel-sourced run
        would land unthreaded.

        A no-op once a thread id is set or the session has been delivered.
        When the session is already streaming, ``/start`` is re-issued — the
        server's reopen path merges the new thread id and keeps every running
        total.
        """
        if not thread_id or self._finished or self.thread_id:
            return
        self.thread_id = thread_id
        if self._streaming:
            self._transport.start_stream(
                self.session_id,
                {
                    "threadId": thread_id,
                    "traceId": self.trace_id,
                    "rootSpanId": self.root_span_id,
                    "agent": self._agent,
                    **({"metadata": self._metadata} if self._metadata else {}),
                    "startedAt": _iso(self._started_at),
                },
            )

    def set_agent(self, agent: Optional[Agent]) -> None:
        """Merge agent identity discovered after the session opened.

        Same reason as :meth:`set_thread_id` — an OTel root span names the
        agent, and it ends after its children.
        """
        if not agent or self._finished:
            return
        self._agent = {**self._agent, **{k: v for k, v in agent.items() if v}}  # type: ignore[dict-item]

    def flush(self, timeout: Optional[float] = None) -> None:
        """Deliver whatever is buffered and wait for the network to settle."""
        if self._streaming:
            self._flush_queued()
        self._transport.flush(timeout)

    def _guard(self, fn: Callable[[], Any]) -> Any:
        """Run a sanitisation step that must never escape into the caller."""
        try:
            return fn()
        except Exception as error:  # pragma: no cover - defensive
            self._config.report_error(error)
            return None

    def get_summary(self) -> Summary:
        """Snapshot of the running totals — handy in tests and for logging."""
        return dict(self._summary)  # type: ignore[return-value]

    # ── Internals ────────────────────────────────────────────────────────
    def _accumulate(self, event: Event) -> None:
        event_type = event.get("type") or "span"
        counts = self._summary["eventCounts"]
        counts[event_type] = counts.get(event_type, 0) + 1
        self._summary["totalInputTokens"] += event.get("inputTokens") or 0
        self._summary["totalOutputTokens"] += event.get("outputTokens") or 0
        self._summary["totalCachedInputTokens"] += event.get("cachedInputTokens") or 0
        self._summary["totalDurationMs"] += event.get("durationMs") or 0
        if event.get("status") == "error":
            raw_error = event.get("error")
            message = (
                raw_error.get("message")
                if isinstance(raw_error, dict)
                else (raw_error if isinstance(raw_error, str) else None)
            )
            self._errors.append(
                {
                    "eventId": event.get("id"),
                    "type": event_type,
                    "message": message or event.get("label") or "Event error",
                    "timestamp": event.get("timestamp"),
                }
            )

    def _maybe_stream(self) -> None:
        """Decide whether this session should go live, and do it if so."""
        if self._finished or self._mode == "batch":
            return
        # Already live: hand the event straight to the transport. Without this
        # the session would open a stream and then sit on every subsequent
        # event until ``end()`` — delivery-correct, but it defeats the point of
        # streaming, since the run would not appear in the Console until it
        # finished.
        if self._streaming:
            self._flush_queued()
            return
        if self._mode == "stream":
            self._begin_stream()
            return
        elapsed_ms = (time.monotonic() - self._started_monotonic) * 1000
        if (
            elapsed_ms >= self._config.stream_after_ms
            or len(self._queue) >= self._config.stream_after_events
        ):
            self._begin_stream()

    def _begin_stream(self) -> None:
        """Open the streaming session and backfill everything buffered so far."""
        if self._streaming:
            return
        self._streaming = True
        payload: Dict[str, Any] = {
            "traceId": self.trace_id,
            "rootSpanId": self.root_span_id,
            "agent": self._agent,
            "startedAt": _iso(self._started_at),
        }
        if self.thread_id:
            payload["threadId"] = self.thread_id
        if self._session_config:
            payload["config"] = self._session_config
        if self._metadata:
            payload["metadata"] = self._metadata
        self._transport.start_stream(self.session_id, payload)
        self._flush_queued()

    def _flush_queued(self) -> None:
        pending, self._queue = self._queue, []
        for event in pending:
            self._transport.append_event(self.session_id, event)

    # ── Context manager ──────────────────────────────────────────────────
    def __enter__(self) -> "TraceSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None:
            self.end(status="error", error=exc)
        else:
            self.end()
