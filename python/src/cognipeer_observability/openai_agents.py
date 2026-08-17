"""OpenAI Agents SDK tracing processor.

    import cognipeer_observability as cognipeer
    from cognipeer_observability.openai_agents import install_openai_agents_tracing

    cognipeer.init(api_key="cpeer_…")
    install_openai_agents_tracing()          # replaces OpenAI's own exporter

    from agents import Agent, Runner, RunConfig
    await Runner.run(agent, "…", run_config=RunConfig(group_id="conv-42"))

``group_id`` is the SDK's only conversation identifier and nothing sets it for
you — not even an SDK ``Session``. Pass it if you want runs to group into one
thread in the Console.

**What the SDK can and cannot tell us.** Prompts, completions, tool arguments,
tool results, per-call token usage (with the cached-token breakdown) and the
agent/handoff/guardrail structure all come through. Full tool JSON schemas only
exist on the Responses API path (``ResponseSpanData.response.tools``); on the
Chat Completions path the SDK's traceable-settings allowlist excludes ``tools``,
so only tool *names* are recoverable — those are still emitted, without schemas.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ._ids import span_id_from, trace_id_from
from .client import Cognipeer, get_client
from .session import TraceSession
from .types import Section, ToolDefinition

logger = logging.getLogger("cognipeer")

try:  # pragma: no cover - import guard
    from agents.tracing import TracingProcessor
except ImportError as error:  # pragma: no cover - import guard
    raise ImportError(
        "cognipeer_observability.openai_agents requires the OpenAI Agents SDK. "
        "Install it with `pip install cognipeer-observability[openai-agents]`."
    ) from error


#: span_data.type → Console event type. Types absent here fall back to 'span'.
EVENT_TYPE_BY_SPAN: Dict[str, str] = {
    "generation": "ai_call",
    "response": "ai_call",
    "function": "tool_call",
    "guardrail": "guardrail",
    "transcription": "span",
    "speech": "span",
    "speech_group": "span",
    "mcp_tools": "span",
    "agent": "span",
    "handoff": "span",
    "custom": "span",
    "task": "span",
    "turn": "span",
}

#: Only leaf model calls carry tokens we may count. `task` and `turn` spans
#: repeat the same usage at run and turn granularity — summing all three
#: inflates a session's token totals threefold.
TOKEN_BEARING_SPANS = frozenset({"generation", "response"})


class CognipeerTracingProcessor(TracingProcessor):
    """Bridges the Agents SDK tracing bus onto Cognipeer sessions.

    One SDK trace becomes one Console session; every span becomes one event,
    linked by ``parent_id`` so the Console renders the agent/handoff tree.

    Args:
        client: Client to send through. Defaults to the process-wide one.
        agent_name: Session agent name. Defaults to the trace's workflow name.
        thread_id: Conversation id. Defaults to ``RunConfig.group_id``.
        capture_agent_tools: Emit the agent's tool NAMES as a tool-definitions
            section on agent spans, even when schemas are unavailable.
    """

    def __init__(
        self,
        *,
        client: Optional[Cognipeer] = None,
        agent_name: Optional[str] = None,
        thread_id: Optional[str] = None,
        capture_agent_tools: bool = True,
    ) -> None:
        self._client = client or get_client()
        self._agent_name = agent_name
        self._thread_id = thread_id
        self._capture_agent_tools = capture_agent_tools
        self._sessions: Dict[str, TraceSession] = {}
        #: First model seen per trace — the SDK never puts a model on the
        #: agent span, so the session's model has to be hoisted from a child.
        self._models: Dict[str, str] = {}

    # ── TracingProcessor ─────────────────────────────────────────────────
    def on_trace_start(self, trace: Any) -> None:
        # Deliberately lazy: a resumed run (`ReattachedTrace`) never fires
        # this, so creating the session here alone would drop the whole run.
        self._session_for(trace)

    def on_trace_end(self, trace: Any) -> None:
        session = self._sessions.pop(_trace_id(trace), None)
        if session is None:
            return
        self._models.pop(_trace_id(trace), None)
        session.end()

    def on_span_start(self, span: Any) -> None:
        """No-op.

        The SDK's span data is only fully populated once the span finishes, so
        everything is emitted from :meth:`on_span_end` — which also keeps this
        callback (invoked inline on the agent's own thread) free of work.
        """

    def on_span_end(self, span: Any) -> None:
        try:
            self._emit(span)
        except Exception as error:  # pragma: no cover - defensive
            # The SDK swallows processor exceptions, so a silent failure here
            # would look like "tracing just doesn't work". Log it loudly.
            logger.error("cognipeer: failed to export an Agents SDK span: %s", error)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        for session in list(self._sessions.values()):
            session.end()
        self._sessions.clear()
        self._client.flush(timeout or 5.0)

    def force_flush(self) -> None:
        self._client.flush(5.0)

    # ── Mapping ──────────────────────────────────────────────────────────
    def _session_for(self, trace: Any) -> Optional[TraceSession]:
        trace_id = _trace_id(trace)
        if not trace_id:
            return None
        session = self._sessions.get(trace_id)
        if session is not None:
            return session

        # `group_id` and `metadata` live on TraceImpl, not on the abstract
        # Trace, so a custom Trace implementation would blow up on attribute
        # access — the SDK itself uses getattr here.
        group_id = getattr(trace, "group_id", None)
        metadata = getattr(trace, "metadata", None) or {}
        session = self._client.start_session(
            thread_id=self._thread_id or (str(group_id) if group_id else None),
            agent={"name": self._agent_name or getattr(trace, "name", None) or "agent"},
            trace_id=trace_id_from(trace_id),
            session_config=dict(metadata) if metadata else None,
        )
        self._sessions[trace_id] = session
        return session

    def _emit(self, span: Any) -> None:
        data = getattr(span, "span_data", None)
        if data is None:
            return
        # `span.export()['type']` masquerades as 'custom' for task/turn spans;
        # the span_data attribute is the honest discriminator.
        span_type = getattr(data, "type", None) or "custom"
        trace_id = _trace_id(span)
        session = self._sessions.get(trace_id)
        if session is None:
            session = self._session_for(_SyntheticTrace(trace_id))
        if session is None:
            return

        model = _model_of(data)
        if model and trace_id not in self._models:
            self._models[trace_id] = model

        usage = _usage_of(data) if span_type in TOKEN_BEARING_SPANS else {}
        error = getattr(span, "error", None)
        parent_id = getattr(span, "parent_id", None)

        event: Dict[str, Any] = {
            "id": _span_id(span),
            "type": EVENT_TYPE_BY_SPAN.get(span_type, "span"),
            "label": _label_of(data, span_type),
            "spanId": span_id_from(_span_id(span)),
            "parentSpanId": span_id_from(parent_id) if parent_id else session.root_span_id,
            "traceId": session.trace_id,
            "status": "error" if error else "success",
            "sections": self._sections_of(data, span_type),
            "metadata": {"spanType": span_type},
        }
        if model:
            event["model"] = model
        started_at = getattr(span, "started_at", None)
        ended_at = getattr(span, "ended_at", None)
        if started_at:
            event["timestamp"] = started_at
        duration = _duration_ms(started_at, ended_at)
        if duration is not None:
            event["durationMs"] = duration
        if error:
            event["error"] = {
                "message": _error_message(error),
                **({"data": error.get("data")} if isinstance(error, dict) and error.get("data") else {}),
            }
        if span_type == "function":
            event["toolName"] = getattr(data, "name", None)
            event["actor"] = {"scope": "tool", "name": getattr(data, "name", None) or "tool"}
        elif span_type in TOKEN_BEARING_SPANS:
            event["actor"] = {"scope": "model", "name": model or "model"}
        elif span_type == "agent":
            event["actor"] = {"scope": "agent", "name": getattr(data, "name", None) or "agent"}

        for wire_key, usage_key in (
            ("inputTokens", "input_tokens"),
            ("outputTokens", "output_tokens"),
            ("totalTokens", "total_tokens"),
            ("cachedInputTokens", "cached_tokens"),
        ):
            if usage.get(usage_key) is not None:
                event[wire_key] = usage[usage_key]

        tools = self._tool_definitions_of(data, span_type)
        if tools:
            event["toolDefinitions"] = tools

        response_format = self._response_format_of(data, span_type)
        if response_format:
            event["responseFormat"] = response_format

        session.record(event)  # type: ignore[arg-type]

    def _sections_of(self, data: Any, span_type: str) -> List[Section]:
        sections: List[Section] = []

        if span_type == "function":
            raw_input = getattr(data, "input", None)
            sections.append(
                {
                    "kind": "tool_call",
                    "label": "Arguments",
                    "tool": getattr(data, "name", None) or "tool",
                    # FunctionSpanData.input is the raw JSON argument STRING.
                    "content": _maybe_json(raw_input),
                }
            )
            output = getattr(data, "output", None)
            if output is not None:
                sections.append(
                    {"kind": "tool_result", "label": "Result", "content": _plain(output)}
                )
            mcp_data = getattr(data, "mcp_data", None)
            if mcp_data:
                sections.append({"kind": "metadata", "label": "MCP", "content": _plain(mcp_data)})
            return sections

        if span_type == "generation":
            sections.extend(_messages_to_sections(getattr(data, "input", None), "user"))
            sections.extend(_messages_to_sections(getattr(data, "output", None), "assistant"))
            config = getattr(data, "model_config", None)
            if config:
                sections.append(
                    {"kind": "metadata", "label": "Model settings", "content": _plain(config)}
                )
            return sections

        if span_type == "response":
            # `export()` deliberately drops the payload — read the attributes.
            sections.extend(_messages_to_sections(getattr(data, "input", None), "user"))
            response = _plain(getattr(data, "response", None))
            if isinstance(response, dict):
                instructions = response.get("instructions")
                if instructions:
                    sections.insert(
                        0,
                        {
                            "kind": "message",
                            "label": "Instructions",
                            "role": "system",
                            "content": instructions,
                        },
                    )
                output = response.get("output")
                if output is not None:
                    sections.append(
                        {
                            "kind": "message",
                            "label": "Assistant message",
                            "role": "assistant",
                            "content": output,
                        }
                    )
            elif response is not None:
                sections.append({"kind": "metadata", "label": "Response", "content": response})
            return sections

        if span_type == "agent":
            detail = {
                "handoffs": getattr(data, "handoffs", None),
                "tools": getattr(data, "tools", None),
                "output_type": getattr(data, "output_type", None),
            }
            sections.append({"kind": "metadata", "label": "Agent", "content": _plain(detail)})
            return sections

        if span_type == "handoff":
            sections.append(
                {
                    "kind": "metadata",
                    "label": "Handoff",
                    "content": {
                        "from": getattr(data, "from_agent", None),
                        "to": getattr(data, "to_agent", None),
                    },
                }
            )
            return sections

        if span_type == "guardrail":
            sections.append(
                {
                    "kind": "metadata",
                    "label": "Guardrail",
                    "content": {
                        "name": getattr(data, "name", None),
                        "triggered": getattr(data, "triggered", None),
                    },
                }
            )
            return sections

        if span_type == "mcp_tools":
            sections.append(
                {
                    "kind": "metadata",
                    "label": f"MCP server: {getattr(data, 'server', None)}",
                    "content": _plain(getattr(data, "result", None)),
                }
            )
            return sections

        payload = getattr(data, "data", None)
        if payload is not None:
            sections.append({"kind": "metadata", "label": "Data", "content": _plain(payload)})
        return sections

    def _tool_definitions_of(self, data: Any, span_type: str) -> Optional[List[ToolDefinition]]:
        """The tool menu offered to the model on this call.

        Full schemas exist only on the Responses path, where the API echoes the
        tool array back. Elsewhere the agent span's ``tools`` list of names is
        the best available and is emitted schema-less rather than not at all.
        """
        if span_type == "response":
            response = _plain(getattr(data, "response", None))
            tools = response.get("tools") if isinstance(response, dict) else None
            return _normalize_tools(tools)

        if span_type == "agent" and self._capture_agent_tools:
            names = getattr(data, "tools", None)
            if isinstance(names, (list, tuple)) and names:
                return [{"name": str(name)} for name in names]
        return None


    def _response_format_of(self, data: Any, span_type: str) -> Optional[Dict[str, Any]]:
        """The structured-output contract enforced on this call.

        The Responses path echoes it back as ``text.format``
        (``{type, name, strict, schema}``); the Chat Completions path uses
        ``response_format`` in the model config. An agent span carries only the
        SDK-side ``output_type`` name, emitted schema-less rather than not at
        all — knowing a contract existed is what separates "the model chose
        prose" from "nothing asked for JSON".
        """
        if span_type == "response":
            response = _plain(getattr(data, "response", None))
            if not isinstance(response, dict):
                return None
            text = response.get("text")
            candidate = text.get("format") if isinstance(text, dict) else None
            if candidate is None:
                candidate = response.get("response_format") or response.get("responseFormat")
            return _normalize_response_format(candidate)

        if span_type == "generation":
            config = _plain(getattr(data, "model_config", None))
            if isinstance(config, dict):
                return _normalize_response_format(
                    config.get("response_format") or config.get("responseFormat")
                )
            return None

        if span_type == "agent":
            output_type = getattr(data, "output_type", None)
            # 'str' is the SDK's default — an unconstrained agent, not a contract.
            if isinstance(output_type, str) and output_type and output_type != "str":
                return {"type": "json_schema", "name": output_type}
        return None


def _normalize_response_format(value: Any) -> Optional[Dict[str, Any]]:
    """Coerce an OpenAI structured-output block into our flat shape.

    Accepts both the Responses ``text.format`` object and the Chat Completions
    ``response_format`` envelope.
    """
    if not isinstance(value, dict):
        return None
    raw_type = value.get("type")
    type_ = raw_type if isinstance(raw_type, str) and raw_type else None
    # Plain text mode is a contract in name only; recording it would put a
    # section on every single call.
    if type_ == "text":
        return None
    nested = value.get("json_schema") or value.get("jsonSchema")
    source = nested if isinstance(nested, dict) else value
    schema = source.get("schema")
    schema = schema if isinstance(schema, dict) else None
    if not type_ and schema is None:
        return None

    out: Dict[str, Any] = {"type": type_ or "json_schema"}
    name = source.get("name")
    if isinstance(name, str) and name:
        out["name"] = name
    strict = source.get("strict")
    if isinstance(strict, bool):
        out["strict"] = strict
    if schema is not None:
        out["schema"] = schema
    return out


class _SyntheticTrace:
    """Stand-in when a span arrives for a trace we never saw start."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.name = "agent"
        self.group_id = None
        self.metadata: Dict[str, Any] = {}


# ── Registration ─────────────────────────────────────────────────────────

def install_openai_agents_tracing(
    *,
    keep_openai_exporter: bool = False,
    **processor_options: Any,
) -> CognipeerTracingProcessor:
    """Route Agents SDK traces to Cognipeer.

    By default this REPLACES the SDK's processor list, so traces stop going to
    OpenAI's own dashboard — which also means the SDK no longer needs
    ``OPENAI_API_KEY`` for tracing, and no prompt data leaves for a second
    destination. Pass ``keep_openai_exporter=True`` to export to both.
    """
    from agents.tracing import add_trace_processor, set_trace_processors

    processor = CognipeerTracingProcessor(**processor_options)
    if keep_openai_exporter:
        add_trace_processor(processor)
    else:
        set_trace_processors([processor])
    return processor


# ── Helpers ──────────────────────────────────────────────────────────────

def _trace_id(obj: Any) -> str:
    return str(getattr(obj, "trace_id", "") or "")


def _span_id(span: Any) -> str:
    return str(getattr(span, "span_id", "") or "")


def _label_of(data: Any, span_type: str) -> str:
    name = getattr(data, "name", None)
    if isinstance(name, str) and name:
        return name
    if span_type == "handoff":
        return f"{getattr(data, 'from_agent', '?')} → {getattr(data, 'to_agent', '?')}"
    if span_type == "mcp_tools":
        return f"mcp: {getattr(data, 'server', '?')}"
    model = _model_of(data)
    return model or span_type


def _model_of(data: Any) -> Optional[str]:
    model = getattr(data, "model", None)
    if isinstance(model, str) and model:
        return model
    response = _plain(getattr(data, "response", None))
    if isinstance(response, dict):
        value = response.get("model")
        if isinstance(value, str) and value:
            return value
    return None


def _usage_of(data: Any) -> Dict[str, Optional[int]]:
    """Normalize the SDK's several usage shapes into one.

    Python's ``ResponseSpanData.usage`` is pre-serialized; JS leaves it on the
    raw response. ``GenerationSpanData.usage`` may still use the older
    ``prompt_tokens``/``completion_tokens`` names.
    """
    usage = getattr(data, "usage", None)
    if not isinstance(usage, dict):
        response = _plain(getattr(data, "response", None))
        usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return {}

    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached = None
    if isinstance(input_details, dict):
        cached = input_details.get("cached_tokens")

    return {
        "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
        "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached,
    }


def _messages_to_sections(payload: Any, default_role: str) -> List[Section]:
    """Render an OpenAI-shaped message list as message sections."""
    if payload is None:
        return []
    if isinstance(payload, str):
        return [
            {
                "kind": "message",
                "label": f"{default_role.capitalize()} message",
                "role": default_role,
                "content": payload,
            }
        ]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, (list, tuple)):
        return [{"kind": "metadata", "label": "Payload", "content": _plain(payload)}]

    sections: List[Section] = []
    for message in payload:
        item = _plain(message)
        if not isinstance(item, dict):
            sections.append({"kind": "metadata", "label": "Item", "content": item})
            continue
        role = item.get("role") or default_role
        sections.append(
            {
                "kind": "message",
                "label": f"{str(role).capitalize()} message",
                "role": str(role),
                "content": item.get("content", item),
            }
        )
        for call in item.get("tool_calls") or []:
            call_data = _plain(call)
            function = call_data.get("function", {}) if isinstance(call_data, dict) else {}
            sections.append(
                {
                    "kind": "tool_call",
                    "label": f"Tool call: {function.get('name', 'tool')}",
                    "tool": function.get("name", "tool"),
                    "content": _maybe_json(function.get("arguments")),
                }
            )
    return sections


def _normalize_tools(tools: Any) -> Optional[List[ToolDefinition]]:
    if not isinstance(tools, (list, tuple)):
        return None
    out: List[ToolDefinition] = []
    for entry in tools:
        item = _plain(entry)
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            # Hosted tools (web_search, file_search, …) have no name, only a type.
            tool_type = item.get("type")
            if isinstance(tool_type, str) and tool_type:
                out.append({"name": tool_type})
            continue
        definition: ToolDefinition = {"name": name}
        description = fn.get("description")
        if isinstance(description, str) and description:
            definition["description"] = description
        parameters = fn.get("parameters") or fn.get("input_schema")
        if isinstance(parameters, dict):
            definition["parameters"] = parameters
        out.append(definition)
    return out or None


def _plain(value: Any) -> Any:
    """Best-effort conversion of SDK objects (pydantic, dataclasses) to JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    for attr in ("model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:  # pragma: no cover - defensive
                pass
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _maybe_json(value: Any) -> Any:
    """Tool arguments arrive as a JSON string — parse so the UI can tree them."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return _plain(value)


def _error_message(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or "Span error")
    return str(error)


def _duration_ms(started_at: Any, ended_at: Any) -> Optional[int]:
    if not started_at or not ended_at:
        return None
    from datetime import datetime

    try:
        start = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(ended_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0, int((end - start).total_seconds() * 1000))


__all__ = [
    "CognipeerTracingProcessor",
    "EVENT_TYPE_BY_SPAN",
    "TOKEN_BEARING_SPANS",
    "install_openai_agents_tracing",
]
