"""``responseFormat`` — the structured-output contract carried per model call.

Every producer has to land on the same flat ``{type, name?, strict?, schema?}``
shape, because the Console normalizes exactly that into its ``response_format``
section. As with the rest of the normalizer, **the two SDKs must agree**: the
assertions below mirror the TypeScript tests one for one, so the same run traced
from either language produces the same event.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

import cognipeer_observability as cognipeer
from cognipeer_observability import _transport
from cognipeer_observability.langchain import _response_format
from cognipeer_observability.otel import normalize as n

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"total": {"type": "number"}},
    "required": ["total"],
}


class TestOtelAttributeExtraction:
    def test_reads_response_format_out_of_openinference_invocation_parameters(self) -> None:
        assert n.extract_response_format(
            {
                "llm.invocation_parameters": json.dumps(
                    {
                        "temperature": 0,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"name": "invoice", "strict": True, "schema": SCHEMA},
                        },
                    }
                )
            }
        ) == {"type": "json_schema", "name": "invoice", "strict": True, "schema": SCHEMA}

    def test_reads_a_directly_attached_response_format_body(self) -> None:
        assert n.extract_response_format(
            {"llm.request.response_format": json.dumps({"type": "json_object"})}
        ) == {"type": "json_object"}

    def test_reassembles_the_genai_split_form(self) -> None:
        assert n.extract_response_format(
            {
                "gen_ai.output.type": "json",
                "gen_ai.request.structured_output_schema": json.dumps(SCHEMA),
                "gen_ai.request.structured_output_name": "invoice",
            }
        ) == {"type": "json_schema", "name": "invoice", "schema": SCHEMA}

    def test_json_mode_without_a_schema_is_a_contract_of_its_own(self) -> None:
        assert n.extract_response_format({"gen_ai.output.type": "json"}) == {"type": "json_object"}

    def test_says_nothing_when_the_span_says_nothing(self) -> None:
        assert n.extract_response_format({}) is None
        assert n.extract_response_format({"gen_ai.output.type": "text"}) is None
        assert n.extract_response_format({"llm.invocation_parameters": "not json"}) is None


def _span(name: str = "span", **attributes: Any) -> n.SpanData:
    return n.SpanData.from_mapping({"name": name, "attributes": attributes})


class TestNormalizeSpan:
    def test_attaches_the_contract_to_an_ai_call_event(self) -> None:
        result = n.normalize_span(
            _span(
                "chat gpt-4.1-mini",
                **{
                    "openinference.span.kind": "LLM",
                    "llm.model_name": "gpt-4.1-mini",
                    "llm.invocation_parameters": json.dumps(
                        {"response_format": {"type": "json_object"}}
                    ),
                },
            )
        )
        assert result.event["type"] == "ai_call"
        assert result.event["responseFormat"] == {"type": "json_object"}

    def test_leaves_a_tool_span_alone(self) -> None:
        # The contract belongs to the model call, not to the tool it decided to
        # invoke — attaching it to both would double-count it in the UI.
        result = n.normalize_span(
            _span(
                "search",
                **{
                    "openinference.span.kind": "TOOL",
                    "tool.name": "search",
                    "llm.invocation_parameters": json.dumps(
                        {"response_format": {"type": "json_object"}}
                    ),
                },
            )
        )
        assert "responseFormat" not in result.event


class TestLangChainInvocationParams:
    def test_records_the_bound_response_format(self) -> None:
        assert _response_format(
            {
                "model": "gpt-4.1-mini",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "invoice", "strict": True, "schema": SCHEMA},
                },
            }
        ) == {"type": "json_schema", "name": "invoice", "strict": True, "schema": SCHEMA}

    def test_accepts_the_bare_json_object_mode(self) -> None:
        assert _response_format({"response_format": {"type": "json_object"}}) == {
            "type": "json_object"
        }

    def test_stays_silent_for_an_unconstrained_call(self) -> None:
        assert _response_format({"model": "gpt-4.1-mini"}) is None
        assert _response_format(None) is None
        assert _response_format({"response_format": "nonsense"}) is None


class TestSessionThreading:
    def test_the_contract_survives_open_to_close(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: List[Any] = []
        monkeypatch.setattr(
            _transport.Transport,
            "_submit",
            lambda self, path, payload: sent.append((path, payload)),
        )
        monkeypatch.setenv("COGNIPEER_API_KEY", "cpeer_test_key_000000000000")
        cognipeer.reset_client()

        client = cognipeer.init(mode="batch", agent={"name": "test"})
        session = client.start_session()
        session.open_span(
            "run-1",
            type="ai_call",
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
        )
        session.close_span("run-1", status="success")
        session.end()

        events = sent[0][1]["events"]
        assert events[0]["responseFormat"] == {"type": "json_object"}
