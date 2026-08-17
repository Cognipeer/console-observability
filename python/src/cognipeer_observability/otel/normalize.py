"""Semantic-convention normalizer: OTel spans → the Console event model.

Three conventions are in the wild and a single span can carry two of them, so
everything here detects by ATTRIBUTE PRESENCE and never by a version marker.

**OpenInference** (Arize/Phoenix — 37 framework instrumentors)::

    type                openinference.span.kind          (LLM|TOOL|AGENT|CHAIN|RETRIEVER|…)
    model               llm.model_name
    inputTokens         llm.token_count.prompt
    outputTokens        llm.token_count.completion
    cachedInputTokens   llm.token_count.prompt_details.cache_read
    messages            llm.input_messages.N.message.{role,content}
                        llm.output_messages.N.message.{role,content}
    tool calls          …message.tool_calls.M.tool_call.function.{name,arguments}
    tool result         output.value  (on a TOOL span)
    responseFormat      llm.invocation_parameters.response_format
    toolDefinitions     llm.tools.N.tool.json_schema   (whole tool object, JSON-encoded)
    toolName            tool.name
    thread id           session.id                     user id: user.id

**OTel GenAI / OpenLLMetry >= 0.55** (the current Traceloop line)::

    type                gen_ai.operation.name            (chat|execute_tool|embeddings|…)
    model               gen_ai.response.model  →  gen_ai.request.model
    inputTokens         gen_ai.usage.input_tokens
    outputTokens        gen_ai.usage.output_tokens
    cachedInputTokens   gen_ai.usage.cache_read.input_tokens
    messages            gen_ai.input.messages / gen_ai.output.messages
                        (JSON string: [{role, parts:[…]}])
    system prompt       gen_ai.system_instructions       SPLIT OUT of input.messages
    tool calls          parts[type=tool_call]{id,name,arguments}
    tool result         parts[type=tool_call_response]  /  gen_ai.tool.call.result
    responseFormat      gen_ai.output.type + gen_ai.request.structured_output_schema
    toolDefinitions     gen_ai.tool.definitions          (JSON array, already unwrapped)
    toolName            gen_ai.tool.name
    thread id           gen_ai.conversation.id

**OpenLLMetry <= 0.54** (legacy; v0.55 cut over with no dual-emit)::

    type                traceloop.span.kind  /  llm.request.type
    model               gen_ai.request.model
    inputTokens         gen_ai.usage.prompt_tokens
    outputTokens        gen_ai.usage.completion_tokens
    cachedInputTokens   gen_ai.usage.cache_read_input_tokens
    messages            gen_ai.prompt.N.{role,content} / gen_ai.completion.N.{role,content}
    tool calls          gen_ai.{prompt,completion}.N.tool_calls.M.{name,arguments}
    tool result         traceloop.entity.output
    toolDefinitions     llm.request.functions.N.{name,description,parameters}
    toolName            traceloop.entity.name
    thread id           traceloop.association.properties.session_id
    user id             traceloop.association.properties.user_id

Deliberate behaviours worth knowing, each of which exists because of a real
failure mode:

* **Message extraction uses precedence, not concatenation.** With
  ``OPENINFERENCE_ENABLE_GENAI_SEMCONV=true`` one span carries both namespaces
  describing the same conversation; concatenating would show every message
  twice.
* **A JSON parse failure never drops the event.** OTel truncates attribute
  values at ``OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT``, and since OpenLLMetry 0.55
  the whole conversation lives in ONE attribute — so a breach yields invalid
  JSON. The raw text is captured as a marked section instead.
* **Indexed attributes are grouped and sorted numerically.** They arrive flat
  and in arbitrary order, so ``gen_ai.prompt.7.content`` can precede
  ``gen_ai.prompt.0.role``; naive accumulation scrambles the conversation.
* **Only ``cache_read`` becomes ``cachedInputTokens``.** ``cache_write`` /
  ``cache_creation`` is a different number at a different price; folding it in
  would mis-price every cached call.
* **``__REDACTED__`` is a sentinel, not content.** A redaction switch
  (``OPENINFERENCE_HIDE_*``, ``TRACELOOP_TRACE_CONTENT=false``) replaces the
  value with that literal string, so the section is DROPPED rather than
  rendered — and, more importantly, rather than harvested into an eval dataset
  as if it were a real prompt. The JS normalizer and the Console's own OTLP
  ingest drop it too; all three have to agree or the same run looks different
  depending on how it arrived.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..types import Event, Section, ToolDefinition

# ── Attribute names ──────────────────────────────────────────────────────
# Verified against @arizeai/openinference-semantic-conventions and the
# OpenLLMetry span_utils source rather than transcribed from documentation.

OI_SPAN_KIND = "openinference.span.kind"
OI_MODEL_NAME = "llm.model_name"
OI_PROVIDER = "llm.provider"
OI_SYSTEM = "llm.system"
OI_INPUT_MESSAGES = "llm.input_messages"
OI_OUTPUT_MESSAGES = "llm.output_messages"
OI_TOKEN_PROMPT = "llm.token_count.prompt"
OI_TOKEN_COMPLETION = "llm.token_count.completion"
OI_TOKEN_TOTAL = "llm.token_count.total"
OI_TOKEN_CACHE_READ = "llm.token_count.prompt_details.cache_read"
OI_TOKEN_REASONING = "llm.token_count.completion_details.reasoning"
OI_TOOLS = "llm.tools"
OI_TOOL_JSON_SCHEMA = "tool.json_schema"
OI_TOOL_NAME = "tool.name"
OI_TOOL_DESCRIPTION = "tool.description"
OI_TOOL_PARAMETERS = "tool.parameters"
OI_INPUT_VALUE = "input.value"
OI_OUTPUT_VALUE = "output.value"
OI_SESSION_ID = "session.id"
OI_USER_ID = "user.id"
OI_METADATA = "metadata"
OI_TAGS = "tag.tags"
OI_AGENT_NAME = "agent.name"
OI_RETRIEVAL_DOCUMENTS = "retrieval.documents"
OI_EMBEDDING_MODEL = "embedding.model_name"
OI_INVOCATION_PARAMETERS = "llm.invocation_parameters"
OI_COST_TOTAL = "llm.cost.total"

GENAI_OPERATION = "gen_ai.operation.name"
GENAI_REQUEST_MODEL = "gen_ai.request.model"
GENAI_RESPONSE_MODEL = "gen_ai.response.model"
GENAI_PROVIDER = "gen_ai.provider.name"
GENAI_SYSTEM = "gen_ai.system"
GENAI_INPUT_MESSAGES = "gen_ai.input.messages"
GENAI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GENAI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
GENAI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
GENAI_TOOL_NAME = "gen_ai.tool.name"
GENAI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GENAI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
GENAI_CONVERSATION_ID = "gen_ai.conversation.id"
GENAI_AGENT_NAME = "gen_ai.agent.name"
GENAI_AGENT_ID = "gen_ai.agent.id"
GENAI_WORKFLOW_NAME = "gen_ai.workflow.name"
GENAI_USAGE_INPUT = "gen_ai.usage.input_tokens"
GENAI_USAGE_OUTPUT = "gen_ai.usage.output_tokens"
GENAI_USAGE_TOTAL = "gen_ai.usage.total_tokens"
GENAI_USAGE_CACHE_READ = "gen_ai.usage.cache_read.input_tokens"

LEGACY_PROMPT = "gen_ai.prompt"
LEGACY_COMPLETION = "gen_ai.completion"
LEGACY_FUNCTIONS = "llm.request.functions"
LEGACY_USAGE_PROMPT = "gen_ai.usage.prompt_tokens"
LEGACY_USAGE_COMPLETION = "gen_ai.usage.completion_tokens"
LEGACY_USAGE_CACHE_READ = "gen_ai.usage.cache_read_input_tokens"
LEGACY_REQUEST_TYPE = "llm.request.type"

TRACELOOP_SPAN_KIND = "traceloop.span.kind"
TRACELOOP_ENTITY_NAME = "traceloop.entity.name"
TRACELOOP_ENTITY_INPUT = "traceloop.entity.input"
TRACELOOP_ENTITY_OUTPUT = "traceloop.entity.output"
TRACELOOP_WORKFLOW_NAME = "traceloop.workflow.name"
TRACELOOP_ASSOCIATION = "traceloop.association.properties"

#: Literal written in place of a value hidden by a redaction switch
#: (``OPENINFERENCE_HIDE_*``). Not an absence and not an empty string.
REDACTED_SENTINEL = "__REDACTED__"

#: OpenInference span kind → Console event type. The Console aggregates on its
#: own small vocabulary, so kinds it has no slot for become ``span`` and keep
#: their original name in ``metadata.spanKind``.
EVENT_TYPE_BY_OI_KIND: Dict[str, str] = {
    "LLM": "ai_call",
    "TOOL": "tool_call",
    "RETRIEVER": "retrieval",
    "RERANKER": "retrieval",
    "EMBEDDING": "embedding",
    "GUARDRAIL": "guardrail",
    "AGENT": "span",
    "CHAIN": "span",
    "EVALUATOR": "span",
    "PROMPT": "span",
}

#: ``gen_ai.operation.name`` → Console event type.
EVENT_TYPE_BY_GENAI_OPERATION: Dict[str, str] = {
    "chat": "ai_call",
    "text_completion": "ai_call",
    "generate_content": "ai_call",
    "execute_tool": "tool_call",
    "embeddings": "embedding",
    "invoke_agent": "span",
    "create_agent": "span",
}

#: Event types whose tokens may be summed into the session total. A framework
#: instrumentor and a client instrumentor both wrap the same call, producing a
#: parent chain span and a child model span that each report usage — summing
#: everything double-counts.
TOKEN_BEARING_TYPES = frozenset({"ai_call", "embedding"})


# ── Span input ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpanData:
    """One finished span, in the shape this module normalizes.

    Decoupled from ``opentelemetry.sdk`` on purpose: the normalizer is pure and
    testable without the OTel SDK installed, and the exporter is the only place
    that knows about ``ReadableSpan``.

    ``trace_id`` and ``span_id`` are lower-hex strings of the correct W3C width
    (32 and 16). The Python OTel SDK exposes them as ints — the exporter
    formats them before constructing this.
    """

    trace_id: str
    span_id: str
    name: str
    start_time_ns: int
    end_time_ns: int
    attributes: Mapping[str, Any] = field(default_factory=dict)
    parent_span_id: Optional[str] = None
    #: OTel StatusCode: 0 UNSET, 1 OK, 2 ERROR.
    status_code: Optional[int] = None
    status_message: Optional[str] = None
    resource_attributes: Mapping[str, Any] = field(default_factory=dict)
    #: Span events, as ``{"name": str, "attributes": {...}}`` — carries the
    #: ``exception`` record when an instrumentation records one.
    events: Sequence[Mapping[str, Any]] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SpanData":
        """Build from a plain dict, accepting snake_case or OTLP/JSON keys."""
        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in raw and raw[key] is not None:
                    return raw[key]
            return default

        return cls(
            trace_id=str(pick("trace_id", "traceId", default="")),
            span_id=str(pick("span_id", "spanId", default="")),
            name=str(pick("name", default="")),
            start_time_ns=int(pick("start_time_ns", "startTimeUnixNano", default=0)),
            end_time_ns=int(pick("end_time_ns", "endTimeUnixNano", default=0)),
            attributes=pick("attributes", default={}) or {},
            # OTLP/JSON writes "" rather than omitting the field on a root.
            parent_span_id=(pick("parent_span_id", "parentSpanId") or None),
            status_code=pick("status_code", "statusCode"),
            status_message=pick("status_message", "statusMessage"),
            resource_attributes=pick("resource_attributes", "resourceAttributes", default={}) or {},
            events=pick("events", default=()) or (),
        )


@dataclass
class NormalizedSpan:
    """A span mapped onto the Console model, plus the session-level hints.

    The hints ride along because the values that identify a *session*
    (conversation id, agent name, user) are span attributes — the exporter
    would otherwise have to re-parse the attribute bag to find them.
    """

    event: Event
    conventions: Tuple[str, ...]
    thread_id: Optional[str] = None
    agent_name: Optional[str] = None
    user_id: Optional[str] = None
    is_root: bool = False
    counts_tokens: bool = False


# ── Primitive readers ────────────────────────────────────────────────────

def _str(attributes: Mapping[str, Any], *keys: str) -> Optional[str]:
    """First key present with a non-empty, non-redacted string value."""
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, str) and value and value != REDACTED_SENTINEL:
            return value
    return None


def _num(attributes: Mapping[str, Any], *keys: str) -> Optional[int]:
    """First key present with a numeric value.

    Absent stays absent: a missing token count must not become ``0``, or a
    streaming call with no usage silently under-reports spend.
    """
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(float(value))
            except ValueError:
                continue
    return None


def _json_or(raw: Any, fallback: Any) -> Any:
    """Parse a JSON attribute, falling back rather than raising."""
    if isinstance(raw, (list, dict)):
        return raw
    if not isinstance(raw, str) or not raw:
        return fallback
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return fallback


def group_indexed(attributes: Mapping[str, Any], prefix: str) -> List[Dict[str, Any]]:
    """Group flat ``<prefix>.<N>.<field>`` attributes into an ordered list.

    Both legacy conventions emit indexed keys as separate flat attributes in
    arbitrary order, so the index has to be parsed and sorted numerically —
    reading them in dict order scrambles the conversation.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.(.+)$")
    grouped: Dict[int, Dict[str, Any]] = {}
    for key, value in attributes.items():
        if not isinstance(key, str):
            continue
        match = pattern.match(key)
        if not match:
            continue
        index = int(match.group(1))
        grouped.setdefault(index, {})[match.group(2)] = value
    return [grouped[index] for index in sorted(grouped)]


# ── Convention detection ─────────────────────────────────────────────────

def detect_conventions(attributes: Mapping[str, Any]) -> Tuple[str, ...]:
    """Which conventions this span speaks, by attribute presence.

    Returns one or more of ``openinference``, ``genai``, ``openllmetry-legacy``
    — a span really can carry two at once — or ``("unknown",)``.
    """
    found: List[str] = []

    if attributes.get(OI_SPAN_KIND) is not None or attributes.get(OI_MODEL_NAME) is not None:
        found.append("openinference")

    if any(
        attributes.get(key) is not None
        for key in (
            GENAI_INPUT_MESSAGES,
            GENAI_OUTPUT_MESSAGES,
            GENAI_SYSTEM_INSTRUCTIONS,
            GENAI_TOOL_DEFINITIONS,
            GENAI_USAGE_INPUT,
            GENAI_OPERATION,
        )
    ):
        found.append("genai")

    legacy_prompt = re.compile(r"^gen_ai\.(prompt|completion)\.\d+\.")
    legacy_functions = re.compile(r"^llm\.request\.functions\.\d+\.")
    if any(
        isinstance(key, str) and (legacy_prompt.match(key) or legacy_functions.match(key))
        for key in attributes
    ):
        found.append("openllmetry-legacy")

    if not found and attributes.get(TRACELOOP_SPAN_KIND) is not None:
        found.append("openllmetry-legacy")

    return tuple(found) if found else ("unknown",)


# ── Tool definitions ─────────────────────────────────────────────────────

def extract_tool_definitions(attributes: Mapping[str, Any]) -> Optional[List[ToolDefinition]]:
    """The tool menu offered to the model on this call.

    Precedence, because the three conventions nest the same data differently:
    OpenInference stores the WHOLE tool object JSON-encoded per index,
    OpenLLMetry >=0.55 one JSON array of already-unwrapped dicts, and the
    legacy line three flat attributes per tool.
    """
    indexed = group_indexed(attributes, OI_TOOLS)
    if indexed:
        entries = [_json_or(entry.get(OI_TOOL_JSON_SCHEMA), None) for entry in indexed]
        definitions = _normalize_tool_entries(entries)
        if definitions:
            return definitions

    parsed = _json_or(attributes.get(GENAI_TOOL_DEFINITIONS), None)
    if isinstance(parsed, list):
        definitions = _normalize_tool_entries(parsed)
        if definitions:
            return definitions

    legacy = group_indexed(attributes, LEGACY_FUNCTIONS)
    if legacy:
        definitions = _normalize_tool_entries(legacy)
        if definitions:
            return definitions

    # A TOOL span describes exactly one tool; its own schema is still a menu
    # entry worth keeping.
    name = _str(attributes, OI_TOOL_NAME, GENAI_TOOL_NAME)
    if name and (attributes.get(OI_TOOL_PARAMETERS) or attributes.get(OI_TOOL_DESCRIPTION)):
        return _normalize_tool_entries(
            [
                {
                    "name": name,
                    "description": attributes.get(OI_TOOL_DESCRIPTION),
                    "parameters": attributes.get(OI_TOOL_PARAMETERS),
                }
            ]
        )
    return None


# ── Response format ──────────────────────────────────────────────────────

def extract_response_format(attributes: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The structured-output contract enforced on this model call.

    Three places it hides, in order of fidelity: OpenInference stashes the
    whole request body in ``llm.invocation_parameters`` (``response_format``
    included verbatim); some emitters attach the body directly; OTel GenAI
    splits it into the output MODE (``gen_ai.output.type``) and the schema
    (``gen_ai.request.structured_output_schema``).

    Returns ``None`` when the span says nothing about the contract — which is
    NOT the same as "the call was unconstrained", and the Console shows the
    difference by simply having no section.
    """
    invocation = _json_or(attributes.get(OI_INVOCATION_PARAMETERS), None)
    direct: Any = None
    if isinstance(invocation, dict):
        direct = invocation.get("response_format") or invocation.get("responseFormat")
    if direct is None:
        for key in ("llm.request.response_format", "gen_ai.request.response_format"):
            direct = _json_or(attributes.get(key), None)
            if direct is not None:
                break

    normalized = _normalize_response_format(direct)
    if normalized:
        return normalized

    schema = _json_or(attributes.get("gen_ai.request.structured_output_schema"), None)
    if isinstance(schema, dict):
        out: Dict[str, Any] = {"type": "json_schema", "schema": schema}
        name = attributes.get("gen_ai.request.structured_output_name")
        if isinstance(name, str) and name:
            out["name"] = name
        return out

    # ``json`` with no schema is still a contract — the model was told to emit
    # JSON, which is why an unparseable reply is a defect rather than a choice.
    if attributes.get("gen_ai.output.type") == "json":
        return {"type": "json_object"}
    return None


def _normalize_response_format(value: Any) -> Optional[Dict[str, Any]]:
    """Coerce an OpenAI-shaped ``response_format`` body into our flat form."""
    record = _json_or(value, value)
    if not isinstance(record, dict):
        return None
    raw_type = record.get("type")
    type_ = raw_type if isinstance(raw_type, str) and raw_type else None
    nested = record.get("json_schema") or record.get("jsonSchema")
    source = nested if isinstance(nested, dict) else record
    schema = _json_or(source.get("schema"), None)
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


def _normalize_tool_entries(entries: Sequence[Any]) -> Optional[List[ToolDefinition]]:
    """Flatten any of the accepted tool shapes to ``{name, description, parameters}``."""
    out: List[ToolDefinition] = []
    for entry in entries:
        record = _json_or(entry, entry)
        if not isinstance(record, dict):
            continue
        # OpenInference wraps in the OpenAI envelope; the others do not.
        inner = record.get("function") if isinstance(record.get("function"), dict) else record
        name = inner.get("name")
        if not isinstance(name, str) or not name:
            continue
        definition: ToolDefinition = {"name": name}
        description = inner.get("description")
        if isinstance(description, str) and description and description != REDACTED_SENTINEL:
            definition["description"] = description
        parameters = _json_or(
            inner.get("parameters") or inner.get("input_schema") or inner.get("inputSchema"),
            None,
        )
        if isinstance(parameters, dict):
            definition["parameters"] = parameters
        out.append(definition)
    return out or None


# ── Sections ─────────────────────────────────────────────────────────────

def extract_sections(attributes: Mapping[str, Any]) -> List[Section]:
    """Render the span's conversation as Console sections.

    Message extraction is by PRECEDENCE, not concatenation: a dual-emitting
    span (``OPENINFERENCE_ENABLE_GENAI_SEMCONV=true``) describes one
    conversation in two namespaces, and concatenating shows every message
    twice.
    """
    sections = (
        _sections_openinference(attributes)
        or _sections_genai(attributes)
        or _sections_legacy(attributes)
        or _sections_generic(attributes)
    )

    # Tool spans carry their result in a single value rather than a message.
    if not any(section.get("kind") == "tool_result" for section in sections):
        result = _str(attributes, GENAI_TOOL_CALL_RESULT, TRACELOOP_ENTITY_OUTPUT)
        if result is None and _is_tool_span(attributes):
            result = _str(attributes, OI_OUTPUT_VALUE)
        if result is not None:
            sections.append(
                {
                    "kind": "tool_result",
                    "label": "Tool result",
                    "role": "tool",
                    "content": _maybe_json(result),
                }
            )

    documents = _json_or(attributes.get(OI_RETRIEVAL_DOCUMENTS), None)
    if isinstance(documents, list) and documents:
        sections.append(
            {
                "kind": "metadata",
                "label": f"Documents ({len(documents)})",
                "content": documents,
            }
        )

    metadata = _json_or(attributes.get(OI_METADATA), None)
    if isinstance(metadata, dict) and metadata:
        sections.append({"kind": "metadata", "label": "Metadata", "content": metadata})

    return sections


def _sections_openinference(attributes: Mapping[str, Any]) -> List[Section]:
    sections: List[Section] = []
    for prefix, default_role in ((OI_INPUT_MESSAGES, "user"), (OI_OUTPUT_MESSAGES, "assistant")):
        for message in group_indexed(attributes, prefix):
            role = message.get("message.role")
            role = role if isinstance(role, str) and role else default_role
            content = message.get("message.content")
            if content == REDACTED_SENTINEL:
                # Dropped, not shaped: see the sentinel note in the module
                # docstring.
                continue
            if isinstance(content, str) and content:
                sections.append(
                    {
                        "kind": "message",
                        "label": _message_label(role),
                        "role": role,
                        "content": content,
                    }
                )
            else:
                # Multimodal messages put their parts in message.contents.N.*
                parts = group_indexed(message, "message.contents")
                text = "\n".join(
                    str(part.get("message_content.text"))
                    for part in parts
                    if isinstance(part.get("message_content.text"), str)
                )
                if text:
                    sections.append(
                        {
                            "kind": "message",
                            "label": _message_label(role),
                            "role": role,
                            "content": text,
                        }
                    )

            for call in group_indexed(message, "message.tool_calls"):
                name = call.get("tool_call.function.name")
                sections.append(
                    {
                        "kind": "tool_call",
                        "label": f"Tool call: {name or 'tool'}",
                        "role": role,
                        "tool": str(name) if name else "tool",
                        # Arguments are a JSON *string* in this convention.
                        "content": _maybe_json(call.get("tool_call.function.arguments")),
                    }
                )
    return sections


def _sections_genai(attributes: Mapping[str, Any]) -> List[Section]:
    sections: List[Section] = []

    # The system prompt was split out of input.messages in OpenLLMetry 0.55.
    # Reading only input.messages loses the largest token block on the call.
    sections.extend(
        _genai_parts(
            attributes,
            GENAI_SYSTEM_INSTRUCTIONS,
            forced_role="system",
        )
    )
    sections.extend(_genai_parts(attributes, GENAI_INPUT_MESSAGES, default_role="user"))
    sections.extend(_genai_parts(attributes, GENAI_OUTPUT_MESSAGES, default_role="assistant"))
    return sections


def _genai_parts(
    attributes: Mapping[str, Any],
    key: str,
    *,
    default_role: str = "user",
    forced_role: Optional[str] = None,
) -> List[Section]:
    raw = attributes.get(key)
    if raw is None:
        return []
    if raw == REDACTED_SENTINEL:
        return []

    parsed = _json_or(raw, None)
    if parsed is None:
        # Truncated mid-JSON. Keep the text rather than losing the turn.
        return [
            {
                "kind": "metadata",
                "label": f"{key} (unparseable)",
                "content": raw,
                "truncated": True,
            }
        ]

    # system_instructions is a bare parts[] list; the message keys wrap it in
    # [{role, parts[]}].
    messages = parsed if isinstance(parsed, list) else [parsed]
    sections: List[Section] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if "parts" not in message and ("content" in message or "type" in message):
            # A bare part, as system_instructions emits.
            message = {"role": forced_role or default_role, "parts": [message]}
        role = forced_role or message.get("role") or default_role
        for part in message.get("parts") or []:
            section = _genai_part_section(part, str(role))
            if section is not None:
                sections.append(section)
    return sections


def _genai_part_section(part: Any, role: str) -> Optional[Section]:
    if not isinstance(part, dict):
        return None
    part_type = part.get("type")

    if part_type in (None, "text", "reasoning", "refusal"):
        content = part.get("content")
        if content is None:
            return None
        label = {"reasoning": "Reasoning", "refusal": "Refusal"}.get(
            str(part_type), _message_label(role)
        )
        return {"kind": "message", "label": label, "role": role, "content": content}

    if part_type == "tool_call":
        name = part.get("name") or "tool"
        return {
            "kind": "tool_call",
            "label": f"Tool call: {name}",
            "role": role,
            "tool": str(name),
            # Arguments are already parsed here, unlike OpenInference's string.
            "content": part.get("arguments"),
        }

    if part_type == "tool_call_response":
        return {
            "kind": "tool_result",
            "label": "Result",
            "role": "tool",
            "tool": str(part.get("id") or "tool"),
            "content": _maybe_json(part.get("response")),
        }

    if part_type == "blob":
        # Binary payloads are summarised, never inlined.
        return {
            "kind": "metadata",
            "label": f"Blob ({part.get('modality') or part.get('mime_type') or 'binary'})",
            "content": {"mime_type": part.get("mime_type"), "modality": part.get("modality")},
        }

    return {"kind": "metadata", "label": str(part_type), "content": part}


def _sections_legacy(attributes: Mapping[str, Any]) -> List[Section]:
    sections: List[Section] = []
    for prefix, default_role in ((LEGACY_PROMPT, "user"), (LEGACY_COMPLETION, "assistant")):
        for message in group_indexed(attributes, prefix):
            role = message.get("role")
            role = role if isinstance(role, str) and role else default_role
            content = message.get("content")
            if content == REDACTED_SENTINEL:
                continue
            if content is not None and content != "":
                sections.append(
                    {
                        "kind": "message",
                        "label": _message_label(role),
                        "role": role,
                        "content": content,
                    }
                )
            for call in group_indexed(message, "tool_calls"):
                name = call.get("name")
                sections.append(
                    {
                        "kind": "tool_call",
                        "label": f"Tool call: {name or 'tool'}",
                        "role": role,
                        "tool": str(name) if name else "tool",
                        "content": _maybe_json(call.get("arguments")),
                    }
                )
    return sections


def _sections_generic(attributes: Mapping[str, Any]) -> List[Section]:
    """Last resort: the single-value input/output attributes.

    ``TRACELOOP_TRACE_CONTENT=false`` and the ``OPENINFERENCE_HIDE_*`` switches
    leave spans with perfect token and latency data and no content at all —
    that is a valid span, not a malformed one, so this may legitimately return
    nothing.
    """
    sections: List[Section] = []
    is_tool = _is_tool_span(attributes)
    value = _str(attributes, OI_INPUT_VALUE, TRACELOOP_ENTITY_INPUT)
    if value is not None:
        # On a TOOL span these single values ARE the call: rendering the
        # arguments as a user message loses the tool framing the Console
        # renders from `kind`, and diverges from the JS normalizer.
        sections.append(
            {"kind": "tool_call", "label": "Input", "content": _maybe_json(value)}
            if is_tool
            else {
                "kind": "message",
                "label": "Input",
                "role": "user",
                "content": _maybe_json(value),
            }
        )
    output = _str(attributes, OI_OUTPUT_VALUE, TRACELOOP_ENTITY_OUTPUT)
    if output is not None and not is_tool:
        sections.append(
            {
                "kind": "message",
                "label": "Output",
                "role": "assistant",
                "content": _maybe_json(output),
            }
        )
    return sections


def _message_label(role: str) -> str:
    return f"{role[:1].upper()}{role[1:]} message" if role else "Message"


def _maybe_json(value: Any) -> Any:
    """Parse a JSON string so the Console can render it as a tree."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            return _json_or(stripped, value)
    return value


def _is_tool_span(attributes: Mapping[str, Any]) -> bool:
    return (
        attributes.get(OI_SPAN_KIND) == "TOOL"
        or attributes.get(GENAI_OPERATION) == "execute_tool"
        or attributes.get(TRACELOOP_SPAN_KIND) == "tool"
        or attributes.get(GENAI_TOOL_NAME) is not None
    )


# ── Type, identity, usage ────────────────────────────────────────────────

def normalize_type(attributes: Mapping[str, Any], span_name: str) -> Tuple[str, Optional[str]]:
    """Console event type plus the original span kind, if any."""
    kind = _str(attributes, OI_SPAN_KIND)
    if kind:
        return EVENT_TYPE_BY_OI_KIND.get(kind.upper(), "span"), kind

    operation = _str(attributes, GENAI_OPERATION)
    if operation:
        mapped = EVENT_TYPE_BY_GENAI_OPERATION.get(operation)
        if mapped:
            return mapped, operation
        # Vendors ship unlisted variants — `chat.stream`, `generate_content_v2`.
        if operation.startswith("chat") or operation.startswith("generate"):
            return "ai_call", operation
        # Fall through rather than settling for 'span': the signals below may
        # still identify the call, and the event type gates token attribution.

    traceloop_kind = _str(attributes, TRACELOOP_SPAN_KIND)
    if traceloop_kind and traceloop_kind != "unknown":
        return {"tool": "tool_call", "agent": "span", "workflow": "span", "task": "span"}.get(
            traceloop_kind, "span"
        ), traceloop_kind

    request_type = _str(attributes, LEGACY_REQUEST_TYPE)
    if request_type:
        return {
            "chat": "ai_call",
            "completion": "ai_call",
            "embedding": "embedding",
            "rerank": "retrieval",
        }.get(request_type, "span"), request_type

    # Nothing declared a kind. Two independent signals say "model call", and
    # BOTH are honoured — they must, because usage is only attached to a
    # model-call event, so a mistype here silently discards tokens the span
    # explicitly reported. Kept identical to the JS normalizer.
    if (
        attributes.get(OI_MODEL_NAME)
        or attributes.get(GENAI_REQUEST_MODEL)
        or attributes.get(GENAI_RESPONSE_MODEL)
    ):
        return "ai_call", operation
    if _num(attributes, GENAI_USAGE_INPUT, LEGACY_USAGE_PROMPT) is not None:
        return "ai_call", operation
    if "tool" in span_name.lower():
        return "tool_call", operation
    return "span", operation


def extract_thread_id(attributes: Mapping[str, Any]) -> Optional[str]:
    """Conversation id.

    No instrumentation emits this on its own — every branch requires the
    application to have opted in (``using_session()``, ``set_conversation_id()``,
    ``set_association_properties()``). Absent is the normal case.
    """
    return _str(
        attributes,
        OI_SESSION_ID,
        GENAI_CONVERSATION_ID,
        f"{TRACELOOP_ASSOCIATION}.session_id",
        f"{TRACELOOP_ASSOCIATION}.thread_id",
        f"{TRACELOOP_ASSOCIATION}.conversation_id",
    )


def extract_agent_name(
    attributes: Mapping[str, Any],
    resource_attributes: Mapping[str, Any],
    span_name: str,
) -> Optional[str]:
    """Agent identity, best source first."""
    return _str(
        attributes,
        GENAI_AGENT_NAME,
        OI_AGENT_NAME,
        GENAI_WORKFLOW_NAME,
        TRACELOOP_WORKFLOW_NAME,
    ) or _str(resource_attributes, "service.name") or (span_name or None)


def extract_usage(attributes: Mapping[str, Any]) -> Dict[str, int]:
    """Token counts, absent when unreported.

    ``cache_write`` / ``cache_creation`` is deliberately not read: it is a
    different quantity at a different price, and the Console treats
    ``cachedInputTokens`` as the cache-READ subset of ``inputTokens``.
    """
    usage: Dict[str, int] = {}
    for wire_key, keys in (
        ("inputTokens", (OI_TOKEN_PROMPT, GENAI_USAGE_INPUT, LEGACY_USAGE_PROMPT)),
        ("outputTokens", (OI_TOKEN_COMPLETION, GENAI_USAGE_OUTPUT, LEGACY_USAGE_COMPLETION)),
        ("totalTokens", (OI_TOKEN_TOTAL, GENAI_USAGE_TOTAL)),
        (
            "cachedInputTokens",
            (OI_TOKEN_CACHE_READ, GENAI_USAGE_CACHE_READ, LEGACY_USAGE_CACHE_READ),
        ),
    ):
        value = _num(attributes, *keys)
        if value is not None:
            usage[wire_key] = value
    return usage


def _span_error(span: SpanData) -> Optional[Dict[str, Any]]:
    """Error record from the span status or a recorded exception event."""
    for event in span.events or ():
        if (event.get("name") if isinstance(event, Mapping) else None) != "exception":
            continue
        attributes = event.get("attributes") or {}
        return {
            "message": attributes.get("exception.message") or "Unknown error",
            "type": attributes.get("exception.type"),
            "stack": attributes.get("exception.stacktrace"),
        }
    if span.status_code == 2:
        return {"message": span.status_message or "Span failed"}
    return None


def _iso(timestamp_ns: int) -> Optional[str]:
    if not timestamp_ns:
        return None
    seconds = timestamp_ns / 1_000_000_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


# ── The event ────────────────────────────────────────────────────────────

def normalize_span(span: Any) -> NormalizedSpan:
    """Map one span onto a Console event plus its session-level hints.

    ``span`` is a :class:`SpanData` or any mapping :meth:`SpanData.from_mapping`
    accepts.
    """
    if not isinstance(span, SpanData):
        span = SpanData.from_mapping(span)

    attributes = span.attributes or {}
    conventions = detect_conventions(attributes)
    event_type, span_kind = normalize_type(attributes, span.name)
    error = _span_error(span)

    event: Event = {
        "id": f"{span.trace_id}:{span.span_id}",
        "type": event_type,
        # `tool.name` is last: on a TOOL span it is the only meaningful row
        # label, and the JS normalizer has always ended its chain there.
        "label": _str(
            attributes, GENAI_AGENT_NAME, OI_AGENT_NAME, TRACELOOP_ENTITY_NAME, OI_TOOL_NAME
        )
        or span.name
        or event_type,
        "spanId": span.span_id,
        "status": "error" if error else "success",
    }
    if span.parent_span_id:
        event["parentSpanId"] = span.parent_span_id

    timestamp = _iso(span.start_time_ns)
    if timestamp:
        event["timestamp"] = timestamp
    if span.end_time_ns and span.start_time_ns:
        event["durationMs"] = max(0, (span.end_time_ns - span.start_time_ns) // 1_000_000)

    model = _str(
        attributes, OI_MODEL_NAME, GENAI_RESPONSE_MODEL, GENAI_REQUEST_MODEL, OI_EMBEDDING_MODEL
    )
    if model:
        event["model"] = model

    # Tokens ride ONLY on leaf model calls. An agent/workflow span frequently
    # repeats its children's usage as a run-level aggregate, and attaching it
    # there too would count the same tokens twice in the session totals.
    usage = extract_usage(attributes) if event_type in TOKEN_BEARING_TYPES else {}
    event.update(usage)  # type: ignore[typeddict-item]

    tool_name = _str(attributes, OI_TOOL_NAME, GENAI_TOOL_NAME)
    if tool_name is None and event_type == "tool_call":
        tool_name = _str(attributes, TRACELOOP_ENTITY_NAME) or span.name
    if tool_name:
        event["toolName"] = tool_name
    tool_call_id = _str(attributes, GENAI_TOOL_CALL_ID, "tool_call.id")
    if tool_call_id:
        event["toolExecutionId"] = tool_call_id

    if event_type == "ai_call":
        event["actor"] = {"scope": "model", "name": model or "model"}
    elif event_type == "tool_call":
        event["actor"] = {"scope": "tool", "name": tool_name or "tool"}
    else:
        event["actor"] = {"scope": "agent", "name": event["label"]}

    sections = extract_sections(attributes)
    if sections:
        event["sections"] = sections

    tools = extract_tool_definitions(attributes)
    if tools:
        event["toolDefinitions"] = tools

    if event_type == "ai_call":
        response_format = extract_response_format(attributes)
        if response_format:
            event["responseFormat"] = response_format

    metadata = _event_metadata(attributes, span, conventions, span_kind)
    if metadata:
        event["metadata"] = metadata

    if error:
        event["error"] = error

    return NormalizedSpan(
        event=event,
        conventions=conventions,
        thread_id=extract_thread_id(attributes),
        agent_name=extract_agent_name(attributes, span.resource_attributes or {}, span.name),
        user_id=_str(attributes, OI_USER_ID, f"{TRACELOOP_ASSOCIATION}.user_id"),
        is_root=not span.parent_span_id,
        counts_tokens=event_type in TOKEN_BEARING_TYPES and bool(usage),
    )


def _event_metadata(
    attributes: Mapping[str, Any],
    span: SpanData,
    conventions: Tuple[str, ...],
    span_kind: Optional[str],
) -> Dict[str, Any]:
    """Everything worth keeping that has no first-class field."""
    metadata: Dict[str, Any] = {"conventions": list(conventions)}
    if span_kind:
        metadata["spanKind"] = span_kind

    provider = _str(attributes, OI_PROVIDER, GENAI_PROVIDER, OI_SYSTEM, GENAI_SYSTEM)
    if provider:
        metadata["provider"] = provider

    tags = attributes.get(OI_TAGS)
    if isinstance(tags, (list, tuple)) and tags:
        metadata["tags"] = list(tags)

    invocation = _json_or(attributes.get(OI_INVOCATION_PARAMETERS), None)
    if isinstance(invocation, dict) and invocation:
        # Tool schemas and the full message list can also live here; keep the
        # scalar knobs only so the event does not carry the prompt twice.
        metadata["invocationParameters"] = {
            key: value
            for key, value in invocation.items()
            if isinstance(value, (str, int, float, bool))
        }

    cost = _num(attributes, OI_COST_TOTAL)
    if cost is not None:
        metadata["reportedCostUsd"] = cost

    reasoning = _num(attributes, OI_TOKEN_REASONING)
    if reasoning is not None:
        metadata["reasoningTokens"] = reasoning

    association = {
        key[len(TRACELOOP_ASSOCIATION) + 1 :]: value
        for key, value in attributes.items()
        if isinstance(key, str) and key.startswith(f"{TRACELOOP_ASSOCIATION}.")
    }
    if association:
        metadata["associationProperties"] = association

    # A logical agent-graph hierarchy that deliberately need not agree with the
    # OTel parent/child tree — kept for display, never used to build the tree.
    node_id = _str(attributes, "graph.node.id")
    if node_id:
        metadata["graphNodeId"] = node_id
        parent_node = _str(attributes, "graph.node.parent_id")
        if parent_node:
            metadata["graphNodeParentId"] = parent_node

    scope = _str(span.resource_attributes or {}, "service.name")
    if scope:
        metadata["serviceName"] = scope

    return metadata


__all__ = [
    "EVENT_TYPE_BY_GENAI_OPERATION",
    "EVENT_TYPE_BY_OI_KIND",
    "NormalizedSpan",
    "REDACTED_SENTINEL",
    "SpanData",
    "TOKEN_BEARING_TYPES",
    "detect_conventions",
    "extract_agent_name",
    "extract_response_format",
    "extract_sections",
    "extract_thread_id",
    "extract_tool_definitions",
    "extract_usage",
    "group_indexed",
    "normalize_span",
    "normalize_type",
]
