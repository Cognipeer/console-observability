"""The transport, over a real socket.

Every other test swaps the transport for a list, which is the right way to test
a mapping but leaves the wire itself unexercised: headers, JSON encoding,
request ordering, retry, and the background worker's delivery guarantee. These
tests run a real HTTP server on a loopback port and assert on what actually
arrives.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterator, List, Tuple

import pytest

import cognipeer_observability as cognipeer


class _Recorder(HTTPServer):
    """An HTTP server that records requests and can be told to fail."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.received: List[Dict[str, Any]] = []
        #: Status codes to return before succeeding — for retry tests.
        self.fail_with: List[int] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    server: _Recorder  # type: ignore[assignment]

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length) or b"{}"
        self.server.received.append(
            {
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "body": json.loads(raw),
            }
        )
        status = self.server.fail_with.pop(0) if self.server.fail_with else 200
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"success":true}' if status == 200 else b'{"error":"nope"}')

    def log_message(self, *args: Any) -> None:  # silence the default stderr log
        return


@pytest.fixture
def server() -> Iterator[_Recorder]:
    recorder = _Recorder()
    thread = threading.Thread(target=recorder.serve_forever, daemon=True)
    thread.start()
    try:
        yield recorder
    finally:
        recorder.shutdown()
        recorder.server_close()


@pytest.fixture
def configure(server: _Recorder, monkeypatch: pytest.MonkeyPatch) -> Tuple[_Recorder, Any]:
    monkeypatch.setenv("COGNIPEER_API_KEY", "cpeer_wire_token_000000000000")
    monkeypatch.setenv("COGNIPEER_BASE_URL", server.base_url)
    cognipeer.reset_client()
    return server, cognipeer


def test_sends_the_expected_request(configure: Tuple[_Recorder, Any]) -> None:
    server, _ = configure
    client = cognipeer.init(mode="batch", agent={"name": "wire-agent"})
    session = client.start_session(thread_id="conv-wire")
    session.record({"type": "ai_call", "model": "gpt-4.1-mini", "inputTokens": 10})
    session.end()
    cognipeer.flush(10.0)

    assert len(server.received) == 1
    request = server.received[0]
    assert request["path"] == "/api/client/v1/tracing/sessions"
    assert request["headers"]["authorization"] == "Bearer cpeer_wire_token_000000000000"
    assert request["headers"]["content-type"] == "application/json"
    assert request["headers"]["user-agent"] == "cognipeer-observability-python"
    assert request["body"]["threadId"] == "conv-wire"
    assert request["body"]["events"][0]["model"] == "gpt-4.1-mini"


def test_stream_mode_arrives_in_order(configure: Tuple[_Recorder, Any]) -> None:
    """`/events` for a session the server has not seen `/start` for is a 404.

    Ordering is therefore a correctness requirement, not a nicety, and the
    single-worker dispatcher is what guarantees it.
    """
    server, _ = configure
    client = cognipeer.init(mode="stream")
    session = client.start_session()
    for index in range(5):
        session.record({"type": "tool_call", "label": f"tool-{index}"})
    session.end()
    cognipeer.flush(10.0)

    paths = [request["path"] for request in server.received]
    assert paths[0].endswith("/start")
    assert paths[-1].endswith("/end")
    assert sum(1 for path in paths if path.endswith("/events")) == 5
    labels = [
        request["body"]["event"]["label"]
        for request in server.received
        if request["path"].endswith("/events")
    ]
    assert labels == [f"tool-{index}" for index in range(5)]


def test_retries_a_retryable_status(configure: Tuple[_Recorder, Any]) -> None:
    server, _ = configure
    server.fail_with = [503, 503]
    client = cognipeer.init(mode="batch", max_retries=3)
    session = client.start_session()
    session.record({"type": "ai_call"})
    session.end()
    cognipeer.flush(15.0)

    assert len(server.received) == 3  # two failures, then success


def test_does_not_retry_a_client_error_and_never_raises(
    configure: Tuple[_Recorder, Any],
) -> None:
    server, _ = configure
    server.fail_with = [400]
    errors: List[BaseException] = []
    client = cognipeer.init(mode="batch", max_retries=3, on_error=errors.append)
    session = client.start_session()
    session.record({"type": "ai_call"})
    session.end()  # must not raise
    cognipeer.flush(10.0)

    assert len(server.received) == 1
    assert len(errors) == 1
    assert "400" in str(errors[0])


def test_secrets_in_prompts_never_reach_the_wire(configure: Tuple[_Recorder, Any]) -> None:
    server, _ = configure
    client = cognipeer.init(mode="batch")
    session = client.start_session()
    session.record(
        {
            "type": "ai_call",
            "sections": [
                {
                    "kind": "message",
                    "role": "user",
                    "content": "my key is sk-abcdefghijklmnopqrstuvwx, please use it",
                }
            ],
        }
    )
    session.end()
    cognipeer.flush(10.0)

    wire = json.dumps(server.received)
    assert "sk-abcdefghijklmnopqrstuvwx" not in wire
    assert "[redacted]" in wire
