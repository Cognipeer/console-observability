"""HTTP transport for the Console tracing ingest API.

Two guarantees the rest of the package relies on:

* **Never raises at the caller.** A failed export is reported through
  ``Config.report_error`` and dropped. Losing a trace is acceptable; breaking
  the traced application is not.
* **Never blocks the caller.** Requests run on a background daemon thread.
  Work is processed strictly FIFO, which is what keeps a session's
  ``/start`` → ``/events`` → ``/end`` ordering intact (the streaming endpoints
  reject events for a session that has not been opened yet).

Only the standard library is used, so installing this package cannot conflict
with the HTTP stack of the application being traced.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional
from urllib.parse import quote

from ._config import Config

logger = logging.getLogger("cognipeer")

USER_AGENT = "cognipeer-observability-python"

#: Statuses worth retrying: transient server/edge failures and rate limits.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Bound the backlog so a Console outage cannot exhaust the host's memory.
_MAX_QUEUE = 10_000


class TransportError(Exception):
    """A non-retryable or exhausted export attempt."""

    def __init__(self, message: str, status: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class _Dispatcher:
    """Single background worker draining a FIFO of export callables."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Optional[Callable[[], None]]]" = queue.Queue(maxsize=_MAX_QUEUE)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._dropped = 0

    def submit(self, task: Callable[[], None]) -> None:
        self._ensure_thread()
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning(
                    "cognipeer: export backlog full, dropped %s trace requests", self._dropped
                )

    def flush(self, timeout: Optional[float] = None) -> None:
        """Block until the backlog drains, including the request in flight.

        ``unfinished_tasks`` (rather than ``empty()``) is what makes this wait
        for the task the worker already popped — otherwise ``flush()`` returns
        while the last export is still on the wire, and a script that exits
        immediately afterwards loses it.
        """
        if self._thread is None:
            return
        if timeout is None:
            self._queue.join()
            return
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(
                target=self._run, name="cognipeer-observability", daemon=True
            )
            self._thread = thread
            thread.start()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                task()
            except Exception as error:  # pragma: no cover - defensive
                logger.warning("cognipeer: exporter thread error: %s", error)
            finally:
                self._queue.task_done()


_dispatcher = _Dispatcher()
atexit.register(lambda: _dispatcher.flush(timeout=5.0))


class Transport:
    """Typed wrapper over the four ingest endpoints."""

    def __init__(self, config: Config) -> None:
        self._config = config

    # ── Endpoints ────────────────────────────────────────────────────────
    def ingest_session(self, payload: Any) -> None:
        """Batch ingest — one request carrying the whole session."""
        self._submit("/api/client/v1/tracing/sessions", payload)

    def start_stream(self, session_id: str, payload: Any) -> None:
        """Open a streaming session. Must land before events are accepted."""
        self._submit(
            f"/api/client/v1/tracing/sessions/stream/{quote(session_id, safe='')}/start", payload
        )

    def append_event(self, session_id: str, event: Any) -> None:
        """Append one event to an open streaming session."""
        self._submit(
            f"/api/client/v1/tracing/sessions/stream/{quote(session_id, safe='')}/events",
            {"event": event},
        )

    def end_stream(self, session_id: str, payload: Any) -> None:
        """Close a streaming session."""
        self._submit(
            f"/api/client/v1/tracing/sessions/stream/{quote(session_id, safe='')}/end", payload
        )

    def ingest_otlp(self, payload: Any) -> None:
        """Raw OTLP/HTTP JSON ingest — used by the OpenTelemetry exporter."""
        self._submit("/api/client/v1/traces", payload)

    # ── Plumbing ─────────────────────────────────────────────────────────
    def flush(self, timeout: Optional[float] = None) -> None:
        _dispatcher.flush(timeout)

    def _submit(self, path: str, payload: Any) -> None:
        if not self._config.enabled:
            return
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as error:
            self._config.report_error(error)
            return
        _dispatcher.submit(lambda: self._post(path, body))

    def _post(self, path: str, body: bytes) -> None:
        url = f"{self._config.base_url}{path}"
        last_error: Optional[BaseException] = None

        for attempt in range(self._config.max_retries + 1):
            if attempt:
                time.sleep(_backoff_seconds(attempt))
            try:
                request = urllib.request.Request(
                    url,
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._config.api_key}",
                        "User-Agent": USER_AGENT,
                        **self._config.headers,
                    },
                )
                with urllib.request.urlopen(request, timeout=self._config.timeout) as response:
                    if self._config.debug:
                        logger.debug("cognipeer: POST %s → %s", path, response.status)
                    return
            except urllib.error.HTTPError as error:
                text = _read_error_body(error)
                failure = TransportError(
                    f"POST {path} failed: {error.code} {text[:500]}", error.code, text
                )
                if error.code not in _RETRYABLE_STATUS:
                    self._config.report_error(failure)
                    return
                last_error = failure
            except Exception as error:  # URLError, socket.timeout, ssl errors…
                last_error = error

        self._config.report_error(
            last_error or TransportError(f"POST {path} failed after retries")
        )


def _read_error_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        return ""


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped at 8s."""
    base = min(8.0, 0.25 * (2 ** (attempt - 1)))
    return base + random.random() * base * 0.25
