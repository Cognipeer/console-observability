"""Claude Agent SDK tracing.

The SDK's message stream is the richest seam it has: it carries the model, the
full assistant content (text, thinking and tool_use blocks), per-call token
usage with the cache breakdown, tool results, subagent nesting via
``parent_tool_use_id`` and the final cost/duration summary. So the integration
consumes that stream rather than installing hooks::

    import cognipeer_observability as cognipeer
    from cognipeer_observability.claude_agent_sdk import trace_query

    cognipeer.init(api_key="cpeer_…")

    async for message in trace_query(
        prompt="refactor the billing module",
        options=ClaudeAgentOptions(model="claude-opus-5"),
        thread_id="conv-42",
    ):
        print(message)

``trace_query`` yields the SDK's messages untouched, so it is a drop-in
replacement for ``query``. If you drive the stream yourself (``ClaudeSDKClient``,
or a wrapper of your own) use :class:`ClaudeMessageTracer` and feed it each
message.

**Not available through this seam:** tool JSON schemas. The SDK reports tool
*names* on the ``system/init`` message but never their input schemas — only
Claude Code's OTel `api_request_body` event carries those, and it is off by
default. Tool names are recorded; schemas are simply absent.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from .client import Cognipeer, get_client
from .session import TraceSession
from .types import Section

logger = logging.getLogger("cognipeer")


class ClaudeMessageTracer:
    """Turns a Claude Agent SDK message stream into a Cognipeer session.

    Feed it every message the SDK yields — in order — and call :meth:`close`
    when the stream ends. A ``result`` message closes the session on its own,
    so ``close`` is only needed when the stream is abandoned early.

    Args:
        client: Client to send through. Defaults to the process-wide one.
        agent: Agent identity. Defaults to ``{"name": "claude-agent"}``.
        thread_id: Conversation id, so several turns group into one thread.
        session_id: Fixed session id; otherwise one is generated per stream.
    """

    def __init__(
        self,
        *,
        client: Optional[Cognipeer] = None,
        agent: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._client = client or get_client()
        self._agent = agent or {"name": "claude-agent"}
        self._thread_id = thread_id
        self._session_id = session_id
        self._session: Optional[TraceSession] = None
        #: tool_use blocks awaiting their tool_result, keyed by tool_use id.
        self._pending_tools: Dict[str, Dict[str, Any]] = {}
        self._tool_names: List[str] = []
        #: API turns whose usage has already been counted. The CLI can split
        #: one turn across several assistant messages that share a
        #: `message.id`, each repeating the SAME cumulative usage — counting
        #: every one of them multiplies the run's token totals.
        self._counted_message_ids: Set[str] = set()

    @property
    def session(self) -> Optional[TraceSession]:
        return self._session

    # ── Stream consumption ───────────────────────────────────────────────
    def handle(self, message: Any) -> None:
        """Record one SDK message. Never raises."""
        try:
            self._handle(message)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("cognipeer: failed to record a Claude SDK message: %s", error)

    def close(self, *, status: str = "success", error: Any = None) -> None:
        """End the session if the stream stopped without a ``result``."""
        self._flush_pending_tools()
        if self._session is not None:
            self._session.end(status=status, error=error)
            self._session = None

    # ── Internals ────────────────────────────────────────────────────────
    def _ensure_session(self, message: Any) -> TraceSession:
        if self._session is None:
            self._session = self._client.start_session(
                session_id=self._session_id,
                thread_id=self._thread_id or _get(message, "session_id"),
                agent=self._agent,
            )
        return self._session

    def _handle(self, message: Any) -> None:
        kind = _message_type(message)

        if kind == "system":
            self._handle_system(message)
        elif kind == "assistant":
            self._handle_assistant(message)
        elif kind == "user":
            self._handle_user(message)
        elif kind == "result":
            self._handle_result(message)

    def _handle_system(self, message: Any) -> None:
        # Python's SystemMessage keeps init data in an untyped `.data` dict;
        # TypeScript spreads the same fields onto the message itself.
        data = _get(message, "data") or {}
        payload = data if isinstance(data, dict) else {}
        model = payload.get("model") or _get(message, "model")
        tools = payload.get("tools") or _get(message, "tools") or []
        if isinstance(tools, list):
            self._tool_names = [str(name) for name in tools]

        if model:
            self._agent = {**self._agent, "model": model}
        session = self._ensure_session(message)
        session.record(
            {
                "type": "span",
                "label": "Session init",
                "actor": {"scope": "agent", "name": self._agent.get("name", "claude-agent")},
                "sections": [
                    {
                        "kind": "metadata",
                        "label": "Init",
                        "content": {
                            "model": model,
                            "tools": self._tool_names,
                            "mcp_servers": payload.get("mcp_servers")
                            or _get(message, "mcp_servers"),
                            "permission_mode": payload.get("permissionMode")
                            or _get(message, "permissionMode"),
                            "cwd": payload.get("cwd") or _get(message, "cwd"),
                        },
                    }
                ],
            }
        )

    def _handle_assistant(self, message: Any) -> None:
        session = self._ensure_session(message)
        inner = _get(message, "message")
        model = _get(message, "model") or _get(inner, "model")
        # One API turn can arrive as several assistant messages sharing a
        # `message.id`, each repeating the SAME cumulative usage. Count the
        # first and treat the rest as text-only, or the run's token totals
        # multiply by the number of messages in the turn.
        message_id = _get(inner, "id") or _get(message, "message_id")
        usage_counted = bool(message_id) and str(message_id) in self._counted_message_ids
        usage = (
            {}
            if usage_counted
            else _usage_of(_get(message, "usage") or _get(inner, "usage"))
        )
        if message_id and usage:
            self._counted_message_ids.add(str(message_id))
        blocks = _content_blocks(inner if inner is not None else message)
        parent = _get(message, "parent_tool_use_id")

        sections: List[Section] = []
        for block in blocks:
            block_type = _block_type(block)
            if block_type == "text":
                sections.append(
                    {
                        "kind": "message",
                        "label": "Assistant message",
                        "role": "assistant",
                        "content": _get(block, "text"),
                    }
                )
            elif block_type == "thinking":
                sections.append(
                    {
                        "kind": "message",
                        "label": "Thinking",
                        "role": "assistant",
                        "content": _get(block, "thinking") or _get(block, "text"),
                    }
                )
            elif block_type == "tool_use":
                tool_id = str(_get(block, "id") or "")
                name = _get(block, "name") or "tool"
                sections.append(
                    {
                        "kind": "tool_call",
                        "label": f"Tool call: {name}",
                        "tool": str(name),
                        "content": _get(block, "input"),
                    }
                )
                # Held until the matching tool_result arrives so one call and
                # its result become a single event in the Console timeline.
                if tool_id:
                    self._pending_tools[tool_id] = {
                        "name": str(name),
                        "input": _get(block, "input"),
                        "parent": parent,
                    }

        event: Dict[str, Any] = {
            "id": _get(message, "uuid") or _get(inner, "id"),
            "type": "ai_call",
            "label": str(model or "assistant"),
            "actor": {"scope": "model", "name": str(model or "model")},
            "sections": sections,
            "metadata": _assistant_metadata(message, inner),
        }
        if model:
            event["model"] = str(model)
        if parent:
            event["parentSpanId"] = _span_id(str(parent))
        event.update(usage)
        session.record(event)  # type: ignore[arg-type]

    def _handle_user(self, message: Any) -> None:
        """A user message carries tool results — pair them with their calls."""
        session = self._ensure_session(message)
        inner = _get(message, "message")
        structured = _get(message, "tool_use_result")

        for block in _content_blocks(inner if inner is not None else message):
            if _block_type(block) != "tool_result":
                continue
            tool_id = str(_get(block, "tool_use_id") or "")
            pending = self._pending_tools.pop(tool_id, None)
            is_error = bool(_get(block, "is_error"))
            name = (pending or {}).get("name") or "tool"
            sections: List[Section] = []
            if pending is not None:
                sections.append(
                    {
                        "kind": "tool_call",
                        "label": "Arguments",
                        "tool": name,
                        "content": pending.get("input"),
                    }
                )
            sections.append(
                {
                    "kind": "tool_result",
                    "label": "Result",
                    "tool": name,
                    # `tool_use_result` is the structured value; the block's
                    # own content is the stringified version the model saw.
                    "content": structured if structured is not None else _get(block, "content"),
                }
            )

            event: Dict[str, Any] = {
                "id": tool_id or None,
                "type": "tool_call",
                "label": name,
                "toolName": name,
                "toolExecutionId": tool_id or None,
                "actor": {"scope": "tool", "name": name},
                "sections": sections,
                "status": "error" if is_error else "success",
            }
            if tool_id:
                event["spanId"] = _span_id(tool_id)
            parent = (pending or {}).get("parent") or _get(message, "parent_tool_use_id")
            if parent:
                event["parentSpanId"] = _span_id(str(parent))
            if is_error:
                event["error"] = {"message": _stringify(_get(block, "content"))}
            session.record({k: v for k, v in event.items() if v is not None})  # type: ignore[arg-type]

    def _handle_result(self, message: Any) -> None:
        session = self._ensure_session(message)
        self._flush_pending_tools()
        is_error = bool(_get(message, "is_error")) or _get(message, "subtype") != "success"

        session.record(
            {
                "type": "span",
                "label": "Result",
                "status": "error" if is_error else "success",
                "durationMs": _get(message, "duration_ms"),
                "sections": [
                    {
                        "kind": "metadata",
                        "label": "Run summary",
                        "content": {
                            "num_turns": _get(message, "num_turns"),
                            "total_cost_usd": _get(message, "total_cost_usd"),
                            "duration_api_ms": _get(message, "duration_api_ms"),
                            "stop_reason": _get(message, "stop_reason"),
                            "permission_denials": _get(message, "permission_denials"),
                        },
                    },
                    *(
                        [
                            {
                                "kind": "message",
                                "label": "Final answer",
                                "role": "assistant",
                                "content": _get(message, "result"),
                            }
                        ]
                        if _get(message, "result")
                        else []
                    ),
                ],
                # Totals already came from the per-message usage; repeating the
                # run-level `usage` here would double every token count.
                "metadata": {"costUsd": _get(message, "total_cost_usd")},
            }
        )
        session.end(status="error" if is_error else "success")
        self._session = None

    def _flush_pending_tools(self) -> None:
        """Emit calls whose result never arrived (interrupt, denial, crash)."""
        if self._session is None:
            return
        for tool_id, pending in list(self._pending_tools.items()):
            self._session.record(
                {
                    "id": tool_id,
                    "type": "tool_call",
                    "label": pending["name"],
                    "toolName": pending["name"],
                    "toolExecutionId": tool_id,
                    "spanId": _span_id(tool_id),
                    "status": "error",
                    "error": {"message": "Tool call did not complete"},
                    "sections": [
                        {
                            "kind": "tool_call",
                            "label": "Arguments",
                            "tool": pending["name"],
                            "content": pending.get("input"),
                        }
                    ],
                }
            )
        self._pending_tools.clear()


async def trace_query(
    *,
    prompt: Any,
    options: Any = None,
    client: Optional[Cognipeer] = None,
    agent: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **query_kwargs: Any,
) -> AsyncIterator[Any]:
    """Drop-in replacement for ``claude_agent_sdk.query`` that also traces.

    Yields the SDK's messages unchanged, so existing consumer code keeps
    working; the session is closed when the stream ends, including on an
    exception or an early ``break``.
    """
    try:
        from claude_agent_sdk import query as sdk_query
    except ImportError as error:  # pragma: no cover - import guard
        raise ImportError(
            "cognipeer_observability.claude_agent_sdk requires claude-agent-sdk. "
            "Install it with `pip install cognipeer-observability[claude-agent-sdk]`."
        ) from error

    tracer = ClaudeMessageTracer(
        client=client, agent=agent, thread_id=thread_id, session_id=session_id
    )
    try:
        async for message in sdk_query(prompt=prompt, options=options, **query_kwargs):
            tracer.handle(message)
            yield message
    except BaseException as error:
        # Covers GeneratorExit from an early `break` as well as real failures.
        tracer.close(status="error", error=error if isinstance(error, Exception) else None)
        raise
    else:
        tracer.close()


# ── Helpers ──────────────────────────────────────────────────────────────

def _message_type(message: Any) -> str:
    explicit = _get(message, "type")
    if isinstance(explicit, str) and explicit:
        return explicit
    # Python's SDK uses classes rather than a discriminant field.
    name = type(message).__name__
    if name.startswith("Assistant"):
        return "assistant"
    if name.startswith("User"):
        return "user"
    if name.startswith("System"):
        return "system"
    if name.startswith("Result"):
        return "result"
    if name.startswith("Stream"):
        return "stream_event"
    return "unknown"


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _content_blocks(message: Any) -> List[Any]:
    content = _get(message, "content")
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [content]


def _block_type(block: Any) -> str:
    explicit = _get(block, "type")
    if isinstance(explicit, str) and explicit:
        return explicit
    name = type(block).__name__
    if name.startswith("Text"):
        return "text"
    if name.startswith("Thinking"):
        return "thinking"
    if name.startswith("ToolUse"):
        return "tool_use"
    if name.startswith("ToolResult"):
        return "tool_result"
    return "unknown"


def _usage_of(usage: Any) -> Dict[str, int]:
    """Translate Anthropic usage into the Console's token model.

    Anthropic reports ``input_tokens`` EXCLUSIVE of cached tokens, while the
    Console treats ``cachedInputTokens`` as a SUBSET of ``inputTokens`` when
    pricing a call. Adding the cache buckets back into the input total is what
    makes the cost come out right; leaving them out would under-report a
    cache-heavy agent's prompt volume by an order of magnitude.
    """
    if usage is None:
        return {}
    fresh = _int(_get(usage, "input_tokens"))
    cache_read = _int(_get(usage, "cache_read_input_tokens"))
    cache_write = _int(_get(usage, "cache_creation_input_tokens"))
    output = _int(_get(usage, "output_tokens"))

    total_input = fresh + cache_read + cache_write
    out: Dict[str, int] = {}
    if total_input:
        out["inputTokens"] = total_input
    if output:
        out["outputTokens"] = output
    if cache_read:
        out["cachedInputTokens"] = cache_read
    if total_input or output:
        out["totalTokens"] = total_input + output
    return out


def _assistant_metadata(message: Any, inner: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in ("request_id", "subagent_type", "task_description", "session_id"):
        value = _get(message, key)
        if value is not None:
            metadata[key] = value
    stop_reason = _get(inner, "stop_reason")
    if stop_reason is not None:
        metadata["stop_reason"] = stop_reason
    usage = _get(message, "usage") or _get(inner, "usage")
    cache_write = _int(_get(usage, "cache_creation_input_tokens"))
    if cache_write:
        # Not a Console field, but worth surfacing: cache writes are billed
        # above the normal input rate.
        metadata["cacheCreationInputTokens"] = cache_write
    return metadata


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _span_id(value: str) -> str:
    from ._ids import span_id_from

    return span_id_from(value)


def _stringify(value: Any) -> str:
    from ._redact import stringify

    return stringify(value)[:2000]


__all__ = ["ClaudeMessageTracer", "trace_query"]
