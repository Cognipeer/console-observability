"""OpenTelemetry support: the "any framework" path.

    from cognipeer_observability.otel import CognipeerSpanExporter

:class:`CognipeerSpanExporter` needs ``opentelemetry-sdk``
(``pip install cognipeer-observability[otel]``) and says so clearly if it is
missing — but it is resolved LAZILY, on first attribute access, so that
:mod:`cognipeer_observability.otel.normalize` stays reachable without it.

That matters because the normalizer is pure attribute mapping over plain
dictionaries: a receiving backend, or a unit test, can use it with no
OpenTelemetry installed at all.
"""

from typing import TYPE_CHECKING, Any

from .normalize import (
    NormalizedSpan,
    SpanData,
    detect_conventions,
    extract_agent_name,
    extract_sections,
    extract_thread_id,
    extract_tool_definitions,
    extract_usage,
    group_indexed,
    normalize_span,
    normalize_type,
)

if TYPE_CHECKING:  # pragma: no cover - type checkers resolve it eagerly
    from .exporter import CognipeerSpanExporter


def __getattr__(name: str) -> Any:
    """Resolve the exporter on first use (PEP 562)."""
    if name == "CognipeerSpanExporter":
        from .exporter import CognipeerSpanExporter as exporter

        return exporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CognipeerSpanExporter",
    "NormalizedSpan",
    "SpanData",
    "detect_conventions",
    "extract_agent_name",
    "extract_sections",
    "extract_thread_id",
    "extract_tool_definitions",
    "extract_usage",
    "group_indexed",
    "normalize_span",
    "normalize_type",
]
