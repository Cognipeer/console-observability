"""OpenTelemetry span exporter.

This is the "any framework" path. Point any OTel-instrumented agent at it —
CrewAI, LlamaIndex, Pydantic AI, Google ADK, AWS Strands, smolagents, DSPy,
Haystack, Semantic Kernel — and its spans become Console sessions::

    import cognipeer_observability as cognipeer
    from cognipeer_observability.otel import CognipeerSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    cognipeer.init(api_key="cpeer_…")

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(CognipeerSpanExporter()))

    from openinference.instrumentation.crewai import CrewAIInstrumentor
    CrewAIInstrumentor().instrument(tracer_provider=provider)

Two structural facts about OTel shape everything below.

**There is no session lifecycle.** OTel has no "run started" or "run ended"
signal, so a session is opened on the first span of a trace and closed either
when the trace's root span arrives or after ``session_idle_seconds`` of
silence. Both paths exist because neither is sufficient alone: a sampled or
dropped root never arrives, and a long-running agent would otherwise be closed
mid-run.

**There is no conversation id.** Nothing emits one unless the application opted
in — ``using_session()`` (OpenInference), ``set_conversation_id()`` /
``set_association_properties()`` (OpenLLMetry). Without it every run is its own
thread, which is correct but less useful; pass ``thread_id`` to the exporter or
instrument the application to set one.

Sessions are delivered in ``stream`` mode, so each span is shipped as it
arrives rather than buffered until the trace ends. That is what makes a
long-running agent visible while it is still running, and it is also what makes
the idle-close safe: a late span reopens the session and appends, where a
batched re-post would have replaced everything already sent.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._ids import span_id_from
from ..client import Cognipeer, get_client
from ..session import TraceSession
from .normalize import NormalizedSpan, SpanData, normalize_span

logger = logging.getLogger("cognipeer")

try:  # pragma: no cover - import guard
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
except ImportError as error:  # pragma: no cover - import guard
    raise ImportError(
        "cognipeer_observability.otel requires the OpenTelemetry SDK. "
        "Install it with `pip install cognipeer-observability[otel]`."
    ) from error


class _TraceState:
    """One in-flight OTel trace and the Console session mirroring it."""

    __slots__ = ("session", "last_seen", "seen_spans", "has_error", "root_seen", "thread_id")

    def __init__(self, session: TraceSession, thread_id: Optional[str]) -> None:
        self.session = session
        self.thread_id = thread_id
        self.last_seen = time.monotonic()
        #: Guards against a processor re-delivering a span; a duplicate would
        #: double its tokens in the session totals.
        self.seen_spans: set = set()
        self.has_error = False
        self.root_seen = False


class CognipeerSpanExporter(SpanExporter):
    """Ships OTel spans to Cognipeer Console as tracing sessions.

    Args:
        client: Client to send through. Defaults to the process-wide one.
        agent_name: Session agent name. Otherwise resolved from the FIRST span
            of the trace — ``gen_ai.agent.name``, then ``agent.name``, then
            ``service.name``, then the span name. The first span is usually a
            model call rather than the agent span (the agent span ends last),
            so in practice this is ``service.name``: stable across the trace
            and the same for every run of a service, which is what the Agents
            screen wants. The per-step agent identity is still on each event's
            label and actor. Pass this explicitly to override.
        thread_id: Conversation id for every session this exporter creates.
            Use when the instrumented application cannot set one itself.
        session_idle_seconds: Close a session after this long without a new
            span. Set ``0`` to disable and rely on the root span alone.
        session_id_prefix: Prefix for the derived session id.
        mode: Delivery mode passed to each session. ``stream`` (the default) is
            what makes the idle-close safe — see the module docstring.
        max_open_traces: Ceiling on concurrently open traces. The oldest is
            closed when it is passed, so a producer that never emits root spans
            cannot grow the map without bound.
    """

    def __init__(
        self,
        *,
        client: Optional[Cognipeer] = None,
        agent_name: Optional[str] = None,
        thread_id: Optional[str] = None,
        session_idle_seconds: float = 30.0,
        session_id_prefix: str = "otel",
        mode: str = "stream",
        max_open_traces: int = 1000,
    ) -> None:
        self._client = client or get_client()
        self._agent_name = agent_name
        self._thread_id = thread_id
        self._idle_seconds = max(0.0, session_idle_seconds)
        self._prefix = session_id_prefix
        self._mode = mode
        self._max_open_traces = max(1, max_open_traces)

        self._traces: Dict[str, _TraceState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sweeper: Optional[threading.Thread] = None
        self._shutdown = False

    # ── SpanExporter ─────────────────────────────────────────────────────
    def export(self, spans: Sequence[Any]) -> "SpanExportResult":
        """Record a batch of finished spans. Never raises."""
        if self._shutdown:
            return SpanExportResult.FAILURE
        try:
            self._ensure_sweeper()
            finished: List[str] = []
            # Keyed by identity: two spans of one trace share a session and it
            # must only be flushed once per batch.
            touched: Dict[int, TraceSession] = {}
            for span in spans:
                trace_id, session = self._record(span)
                if session is not None:
                    touched[id(session)] = session
                if trace_id is not None:
                    finished.append(trace_id)

            # A streaming session queues recorded events and only pushes them
            # when it next flushes, so an always-open trace would sit in memory
            # until the idle sweep. `flush(0)` hands the queue to the transport
            # without waiting for the network — the actual send happens on the
            # transport's own thread.
            for session in touched.values():
                session.flush(0)

            # Closing is deferred to the end of the batch: within a batch spans
            # arrive in end order, so the root is last, but deferring costs
            # nothing and removes any dependence on that ordering.
            for trace_id in finished:
                self._close(trace_id)
            return SpanExportResult.SUCCESS
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("cognipeer: OTel span export failed: %s", error)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """End every open session and stop the idle sweeper."""
        self._shutdown = True
        self._stop.set()
        with self._lock:
            trace_ids = list(self._traces)
        for trace_id in trace_ids:
            self._close(trace_id, force=True)
        self._client.flush(5.0)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Wait for everything recorded so far to reach the Console.

        Open sessions are deliberately left open: their recorded events are
        pushed to the transport, but ending a session that is still receiving
        spans would split one run in two.
        """
        with self._lock:
            sessions = [state.session for state in self._traces.values()]
        for session in sessions:
            session.flush(0)
        self._client.flush(timeout_millis / 1000.0)
        return True

    # ── Recording ────────────────────────────────────────────────────────
    def _record(self, span: Any) -> Tuple[Optional[str], Optional[TraceSession]]:
        """Record one span.

        Returns ``(trace_id_if_complete, session_touched)`` — the caller uses
        the first to close finished traces and the second to flush.
        """
        data = _to_span_data(span)
        if not data.trace_id or not data.span_id:
            return None, None

        normalized = normalize_span(data)
        state = self._state_for(data.trace_id, normalized)
        if state is None:
            return None, None

        with self._lock:
            # A processor may re-deliver a span on retry; recording it twice
            # would double its tokens in the session totals.
            if data.span_id in state.seen_spans:
                return None, None
            state.seen_spans.add(data.span_id)
            state.last_seen = time.monotonic()
            if normalized.is_root:
                state.root_seen = True
            if normalized.event.get("status") == "error":
                state.has_error = True

        state.session.record(normalized.event)
        return (data.trace_id if normalized.is_root else None), state.session

    def _state_for(self, trace_id: str, normalized: NormalizedSpan) -> Optional[_TraceState]:
        with self._lock:
            state = self._traces.get(trace_id)
            if state is not None:
                # A trace's ROOT span ends last, so the span carrying
                # `session.id` / `gen_ai.conversation.id` — and often the agent
                # name — routinely arrives after a child already opened the
                # session. Backfill both while the session is still open;
                # otherwise every OTel-sourced run lands unthreaded.
                if state.thread_id is None and normalized.thread_id:
                    state.thread_id = normalized.thread_id
                    state.session.set_thread_id(normalized.thread_id)
                # Only the ROOT may rename the agent: a child span's name is a
                # step, not the agent, and would otherwise overwrite a good name.
                if normalized.is_root and normalized.agent_name and not self._agent_name:
                    state.session.set_agent({"name": normalized.agent_name})
                return state

            # Created UNDER the lock, deliberately. The session id is
            # deterministic and the default mode is `stream`, so constructing a
            # session immediately POSTs `/start` for that id — two threads
            # racing on one trace would both open it, and the loser's `end()`
            # would then close the session the winner is still writing to.
            # Creating outside the lock and discarding the loser is therefore
            # not enough; the wire call has already happened. Safe to hold,
            # because delivery is enqueued onto a background worker rather than
            # performed here.
            thread_id = self._thread_id or normalized.thread_id
            session = self._client.start_session(
                session_id=f"{self._prefix}-{trace_id}",
                thread_id=thread_id,
                agent={"name": self._agent_name or normalized.agent_name or "agent"},
                trace_id=trace_id,
                # Deterministic, so a session reopened after an idle close keeps
                # the same anchor and the tree does not fragment.
                root_span_id=span_id_from(f"otel-root:{trace_id}"),
                mode=self._mode,
            )
            state = _TraceState(session, thread_id)
            self._traces[trace_id] = state
            overflowing = self._overflowing_locked()

        # Evicted outside the lock: closing a session is ordinary work and
        # there is no longer any shared state to protect.
        for evicted in overflowing:
            evicted.session.end(status="error" if evicted.has_error else "success")
        return state

    def _overflowing_locked(self) -> List["_TraceState"]:
        """Drop the oldest traces once the ceiling is passed.

        A producer that never emits root spans — a sampled-away or crashed
        instrumentation — would otherwise grow this map for the life of the
        process. Insertion order is open order, so the front of the dict is the
        oldest.
        """
        evicted: List[_TraceState] = []
        while len(self._traces) > self._max_open_traces:
            oldest = next(iter(self._traces))
            evicted.append(self._traces.pop(oldest))
        return evicted

    def _close(self, trace_id: str, *, force: bool = False) -> None:
        with self._lock:
            state = self._traces.get(trace_id)
            if state is None:
                return
            if not force and not state.root_seen:
                return
            del self._traces[trace_id]
        state.session.end(status="error" if state.has_error else "success")

    # ── Idle sweeping ────────────────────────────────────────────────────
    def _ensure_sweeper(self) -> None:
        """Start the idle sweeper on first use.

        Daemon thread: an exporter must never be the reason a process refuses
        to exit. Started lazily so importing the module costs nothing.
        """
        if self._idle_seconds <= 0 or self._shutdown:
            return
        with self._lock:
            if self._sweeper is not None and self._sweeper.is_alive():
                return
            self._sweeper = threading.Thread(
                target=self._sweep, name="cognipeer-otel-sweeper", daemon=True
            )
            self._sweeper.start()

    def _sweep(self) -> None:
        interval = max(1.0, min(self._idle_seconds / 2.0, 5.0))
        while not self._stop.wait(interval):
            cutoff = time.monotonic() - self._idle_seconds
            with self._lock:
                stale = [
                    trace_id
                    for trace_id, state in self._traces.items()
                    if state.last_seen < cutoff
                ]
            for trace_id in stale:
                self._close(trace_id, force=True)


def _to_span_data(span: Any) -> SpanData:
    """Convert a ``ReadableSpan`` (or a compatible object) to :class:`SpanData`.

    The Python SDK exposes trace and span ids as ints; they are formatted to
    the W3C hex widths here rather than anywhere downstream, so the rest of the
    package only ever sees the string form.
    """
    if isinstance(span, SpanData):
        return span

    context = getattr(span, "context", None) or getattr(span, "get_span_context", lambda: None)()
    parent = getattr(span, "parent", None)
    status = getattr(span, "status", None)
    resource = getattr(span, "resource", None)

    return SpanData(
        trace_id=_hex(getattr(context, "trace_id", None), 32),
        span_id=_hex(getattr(context, "span_id", None), 16),
        name=str(getattr(span, "name", "") or ""),
        start_time_ns=int(getattr(span, "start_time", 0) or 0),
        end_time_ns=int(getattr(span, "end_time", 0) or 0),
        attributes=dict(getattr(span, "attributes", None) or {}),
        parent_span_id=_hex(getattr(parent, "span_id", None), 16) or None,
        status_code=_status_code(status),
        status_message=getattr(status, "description", None),
        resource_attributes=dict(getattr(resource, "attributes", None) or {}),
        events=[
            {
                "name": getattr(event, "name", None),
                "attributes": dict(getattr(event, "attributes", None) or {}),
            }
            for event in (getattr(span, "events", None) or ())
        ],
    )


def _hex(value: Any, width: int) -> str:
    if isinstance(value, int) and value:
        return format(value, f"0{width}x")
    if isinstance(value, str) and value:
        return value.lower()
    return ""


def _status_code(status: Any) -> Optional[int]:
    """OTel StatusCode as its int value: 0 UNSET, 1 OK, 2 ERROR."""
    code = getattr(status, "status_code", None)
    if code is None:
        return None
    value = getattr(code, "value", code)
    return value if isinstance(value, int) else None


__all__ = ["CognipeerSpanExporter"]
