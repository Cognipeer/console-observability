"""Wire types for the Cognipeer Console tracing ingest API.

These mirror ``POST /api/client/v1/tracing/sessions`` (batch) and
``POST /api/client/v1/tracing/sessions/stream/{session_id}/{start,events,end}``
(streaming). The JSON keys are the server's camelCase names; the helpers in
this package do the snake_case → camelCase translation so user code never has
to.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Literal, Protocol, TypedDict

if sys.version_info >= (3, 11):  # pragma: no cover - version branch
    from typing import NotRequired
else:  # pragma: no cover - version branch
    from typing_extensions import NotRequired

# Canonical event types the Console aggregates on. Anything else is accepted
# and shown verbatim, but these get first-class treatment in the UI and in the
# cost pipeline.
EventType = Literal[
    "ai_call",
    "tool_call",
    "retrieval",
    "embedding",
    "summarization",
    "guardrail",
    "span",
]

Status = Literal["success", "error", "in_progress"]

CaptureMode = Literal["all", "metadata", "none"]

Mode = Literal["auto", "stream", "batch"]


class ToolDefinition(TypedDict):
    """One entry of the tool menu offered to the model on a given call."""

    name: str
    description: NotRequired[str]
    parameters: NotRequired[Dict[str, Any]]
    truncated: NotRequired[bool]


class ResponseFormat(TypedDict, total=False):
    """The structured-output contract enforced on ONE model call.

    The other half of the request shape next to the tool menu, and captured for
    the same reason: messages alone cannot tell a model that CHOSE prose from
    one that was never asked for JSON, and a replay of the call that drops the
    schema measures a looser system than production runs under. Per event,
    never per session — an agent may enforce a schema on its final turn only.
    """

    #: The wire ``response_format.type``: json_schema | json_object | text.
    type: str
    #: Named schema (``json_schema.name``) when the provider takes one.
    name: str
    #: Whether the provider enforces the schema (``json_schema.strict``).
    strict: bool
    #: The JSON Schema as sent.
    schema: Dict[str, Any]
    #: How it was enforced: a native ``response_format`` or a forced tool call.
    strategy: str


class Section(TypedDict, total=False):
    """A renderable block inside an event.

    ``kind`` drives the badge colour in the tracing detail UI; every other key
    is rendered generically as a labelled field, so extra keys are safe.
    """

    kind: str  # message | tool_call | tool_result | tool_definitions | response_format | metadata
    label: str
    role: str  # system | user | assistant | tool
    content: Any
    tool: str
    tools: List[ToolDefinition]
    truncated: bool


class Actor(TypedDict, total=False):
    name: str
    role: str
    scope: str  # agent | tool | model | user


class Event(TypedDict, total=False):
    id: str
    type: str
    label: str
    sequence: int
    timestamp: str
    status: str
    traceId: str
    spanId: str
    parentSpanId: str

    actor: Actor
    sections: List[Section]
    metadata: Dict[str, Any]

    model: str
    modelNames: List[str]

    inputTokens: int
    outputTokens: int
    cachedInputTokens: int
    #: Reasoning tokens billed INSIDE ``outputTokens`` (a subset of it). On a
    #: reasoning model they are routinely most of the output bill while being
    #: invisible in the response text. Never re-billed — already counted.
    reasoningTokens: int
    #: Why the model stopped: stop | tool_calls | length | content_filter.
    #: ``length`` is the most common cause of a truncated structured response.
    finishReason: str
    totalTokens: int
    requestBytes: int
    responseBytes: int
    durationMs: int

    toolName: str
    toolExecutionId: str
    toolDefinitions: List[ToolDefinition]
    #: Normalized server-side into a ``response_format`` section. The OpenAI
    #: ``response_format`` body is accepted directly as well.
    responseFormat: ResponseFormat

    error: Any


class Agent(TypedDict, total=False):
    name: str
    version: str
    model: str
    provider: str


class Summary(TypedDict, total=False):
    eventCounts: Dict[str, int]
    totalInputTokens: int
    totalOutputTokens: int
    totalCachedInputTokens: int
    totalDurationMs: int
    totalBytesIn: int
    totalBytesOut: int


class SessionPayload(TypedDict, total=False):
    sessionId: str
    threadId: str
    traceId: str
    rootSpanId: str
    agent: Agent
    config: Dict[str, Any]
    #: Free-form attribution tags (e.g. ``{"complexity": "complex"}``), sibling
    #: of ``agent``. The Console groups/reports on these as a dynamic
    #: ``group_by``/``group_by_entity=metadata.<key>`` dimension — string
    #: values only, and unlike event ``metadata`` this is never redacted or
    #: size-capped client-side.
    metadata: Dict[str, str]
    status: str
    startedAt: str
    endedAt: str
    durationMs: int
    summary: Summary
    errors: List[Any]
    events: List[Event]


class Logger(Protocol):
    def debug(self, msg: str, *args: Any) -> None: ...
    def warning(self, msg: str, *args: Any) -> None: ...
    def error(self, msg: str, *args: Any) -> None: ...


ErrorHandler = Callable[[BaseException], None]

__all__ = [
    "Actor",
    "Agent",
    "CaptureMode",
    "ErrorHandler",
    "Event",
    "EventType",
    "Logger",
    "Mode",
    "Section",
    "SessionPayload",
    "Status",
    "Summary",
    "ToolDefinition",
]
