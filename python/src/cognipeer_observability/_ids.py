"""W3C-shaped identifier helpers.

The Console stores ``traceId`` / ``spanId`` / ``parentSpanId`` as opaque
strings, but keeping them W3C-valid (32 / 16 lower-hex chars) means a trace can
be correlated with whatever other OpenTelemetry backend the customer already
runs. Framework run ids are usually UUIDs, so :func:`span_id_from` folds an
arbitrary string into a stable 16-hex id instead of inventing a new one.
"""

from __future__ import annotations

import re
import secrets

_HEX_ONLY = re.compile(r"[^0-9a-f]")


def new_trace_id() -> str:
    """New random 32-hex trace id."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    """New random 16-hex span id."""
    return secrets.token_hex(8)


def new_session_id(prefix: str = "sess") -> str:
    """Readable session id: ``sess_<24 hex>``."""
    return f"{prefix}_{secrets.token_hex(12)}"


def _fnv1a(value: str) -> str:
    """32-bit FNV-1a rendered as 8 lower-hex chars."""
    hash_ = 0x811C9DC5
    for char in value:
        hash_ ^= ord(char) & 0xFFFFFFFF
        hash_ = (hash_ * 0x01000193) & 0xFFFFFFFF
    return f"{hash_:08x}"


def span_id_from(value: str) -> str:
    """Deterministically fold any string into a 16-hex span id.

    The same input always yields the same span id, which is what lets
    ``parentSpanId`` be resolved from a parent run id we may never have seen as
    an event ourselves.

    Only an input that is *already* a span id passes through. Truncating a
    longer identifier is not safe: LangChain run ids are UUIDv7, whose first
    16 hex digits are a millisecond timestamp plus 12 bits of entropy, so
    truncation collides between runs started in the same millisecond — and
    colliding span ids silently corrupt the trace tree.
    """
    lowered = value.lower()
    if len(lowered) == 16 and _HEX_ONLY.sub("", lowered) == lowered:
        return lowered
    return _fnv1a(value) + _fnv1a(value + "#2")


def trace_id_from(value: str) -> str:
    """Deterministically fold any string into a 32-hex trace id.

    Pass-through only for a value that already is one — see
    :func:`span_id_from` for why truncation is unsafe.
    """
    lowered = value.lower()
    if len(lowered) == 32 and _HEX_ONLY.sub("", lowered) == lowered:
        return lowered
    return "".join(_fnv1a(f"{value}#{index}") for index in range(1, 5))
