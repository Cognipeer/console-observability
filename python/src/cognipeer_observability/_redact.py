"""Content sanitisation applied to every section before it leaves the process.

Three jobs, in order:

1. capture policy — ``metadata`` drops content entirely, ``all`` keeps it;
2. secret redaction — built-in patterns for the credential shapes that
   routinely end up inside prompts, plus caller-supplied regexes;
3. size capping — a single oversized message must not blow the ingest body
   limit (``TRACING_MAX_BODY_SIZE_MB``, 10 MB by default) for the whole session.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Pattern, Sequence

from ._config import Config
from .types import Section

REDACTED = "[redacted]"

# Deliberately narrow: a false positive silently destroys the prompt the user
# came to read, so these only match tokens unambiguous by construction.
_BUILTIN_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),  # OpenAI-style
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),  # Anthropic
    re.compile(r"\bcpeer_[A-Za-z0-9_-]{16,}\b"),  # Cognipeer API token
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),  # JWT
]


#: Inline binary payloads. Multimodal messages embed images, audio and PDFs as
#: base64 data URLs; one of them can be tens of megabytes, which would blow the
#: ingest body limit for the whole session and render as a wall of noise.
_DATA_URL = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=]{100,}")


def _strip_data_urls(value: str) -> str:
    return _DATA_URL.sub(
        lambda match: f"data:{match.group(0)[5:match.group(0).find(';')]};base64,"
        f"[stripped {len(match.group(0))} chars]",
        value,
    )


def redact(value: str, extra: Sequence[Pattern[str]] = ()) -> str:
    """Apply the redaction patterns to a string."""
    out = _strip_data_urls(value)
    for pattern in list(_BUILTIN_PATTERNS) + list(extra):
        out = pattern.sub(REDACTED, out)
    return out


class _SafeEncoder(json.JSONEncoder):
    """Survives the values real agent frameworks put in payloads."""

    def default(self, o: Any) -> Any:  # noqa: D102 - inherited
        if isinstance(o, BaseException):
            return {"type": type(o).__name__, "message": str(o)}
        if isinstance(o, (set, frozenset)):
            return list(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        if hasattr(o, "model_dump"):  # pydantic v2
            try:
                return o.model_dump()
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(o, "dict") and callable(o.dict):  # pydantic v1
            try:
                return o.dict()
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(o, "__dict__"):
            return {k: v for k, v in vars(o).items() if not k.startswith("_")}
        return _safe_str(o)


UNSERIALIZABLE = "[unserializable]"


def _safe_str(value: Any) -> str:
    """``str()`` that cannot raise.

    A ``__str__`` or ``__repr__`` written by the traced application is
    arbitrary code, and this runs inside that application's own call stack.
    """
    try:
        return str(value)
    except Exception:
        return UNSERIALIZABLE


def stringify(value: Any) -> str:
    """Serialize any value to the string the Console will render.

    Never raises. The encoder's last resort is ``str(o)``, which is the traced
    application's own code and may raise anything at all — narrowing this to
    ``(TypeError, ValueError)`` let a raising ``__str__`` escape into that
    application over a log line.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, cls=_SafeEncoder, ensure_ascii=False, indent=2)
    except Exception:
        return _safe_str(value)


def cap(value: str, max_chars: int) -> "tuple[str, bool]":
    """Truncate to ``max_chars``, appending a visible marker when cut."""
    if len(value) <= max_chars:
        return value, False
    dropped = len(value) - max_chars
    return f"{value[:max_chars]}\n…[truncated {dropped} chars]", True


def sanitize_section(section: Section, config: Config) -> Optional[Section]:
    """Sanitize one section, or drop it when the capture policy removes it."""
    if config.capture == "none":
        return None

    # `metadata` capture keeps the shape of the run — which tools ran, in which
    # order, with what schema — but never the message bodies.
    if config.capture == "metadata":
        if section.get("kind") == "tool_definitions":
            return section
        stripped: Dict[str, Any] = {k: v for k, v in section.items() if k != "content"}
        stripped["redacted"] = True
        return stripped  # type: ignore[return-value]

    if "content" not in section:
        return section

    raw = stringify(section.get("content"))
    redacted = redact(raw, config.redact_patterns)
    capped, truncated = cap(redacted, config.max_content_chars)

    out: Dict[str, Any] = dict(section)
    out["content"] = capped
    if truncated:
        out["truncated"] = True
    return out  # type: ignore[return-value]


def sanitize_sections(
    sections: Optional[Sequence[Section]], config: Config
) -> Optional[List[Section]]:
    """Sanitize a section list, dropping anything the capture policy removes."""
    if not sections:
        return None
    out = [s for s in (sanitize_section(section, config) for section in sections) if s]
    return out or None


#: How deep a metadata object is walked before the rest is summarised.
METADATA_MAX_DEPTH = 6
#: Entries kept per object/list level in metadata.
METADATA_MAX_ENTRIES = 100
#: Per-value cap inside metadata — a metadata field is a label, not a transcript.
METADATA_VALUE_MAX_CHARS = 4_000


def sanitize_metadata(
    metadata: Optional[Dict[str, Any]], config: Config
) -> Optional[Dict[str, Any]]:
    """Sanitize an event's ``metadata`` or a session's ``config``.

    These are the other channel by which caller data reaches the wire — the AI
    SDK copies ``telemetry.metadata.*`` verbatim, the Agents SDK puts
    ``trace.metadata`` into the session config, LangChain passes user tags
    through — and until this existed they bypassed secret redaction, base64
    stripping and the size cap under EVERY capture mode, including the default
    one whose documented promise is "redacted by the configured patterns".

    Structure is preserved rather than flattened, because the Console renders
    these as key/value blocks: strings are redacted and capped in place, and
    depth/breadth are bounded so a payload accidentally routed through metadata
    cannot blow the ingest body limit for the whole session.
    """
    if not metadata:
        return metadata
    if config.capture == "none":
        return None
    result = _sanitize_metadata_value(metadata, config, 0)
    return result if isinstance(result, dict) else {"value": result}


def _sanitize_metadata_value(value: Any, config: Config, depth: int) -> Any:
    try:
        return _sanitize_metadata_value_unsafe(value, config, depth)
    except Exception:
        # A raising property or __iter__ anywhere in the tree degrades that
        # subtree to a placeholder rather than failing the whole event.
        return UNSERIALIZABLE


def _sanitize_metadata_value_unsafe(value: Any, config: Config, depth: int) -> Any:
    if value is None or isinstance(value, bool) or isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        capped, _ = cap(
            redact(value, config.redact_patterns),
            min(config.max_content_chars, METADATA_VALUE_MAX_CHARS),
        )
        return capped

    if depth >= METADATA_MAX_DEPTH:
        return _sanitize_metadata_value(stringify(value), config, depth)

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        keys = list(value.keys())
        for key in keys[:METADATA_MAX_ENTRIES]:
            out[str(key)] = _sanitize_metadata_value(value[key], config, depth + 1)
        if len(keys) > METADATA_MAX_ENTRIES:
            out["…"] = f"[{len(keys) - METADATA_MAX_ENTRIES} more keys]"
        return out

    if isinstance(value, (list, tuple)):
        items = list(value)
        kept: List[Any] = [
            _sanitize_metadata_value(item, config, depth + 1)
            for item in items[:METADATA_MAX_ENTRIES]
        ]
        if len(items) > METADATA_MAX_ENTRIES:
            kept.append(f"…[{len(items) - METADATA_MAX_ENTRIES} more]")
        return kept

    # Anything with its own rendering (Exception, pydantic model, class
    # instance) goes through the serializer the content path uses, then gets
    # redacted as a string.
    return _sanitize_metadata_value(stringify(value), config, depth)
