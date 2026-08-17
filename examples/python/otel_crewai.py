"""CrewAI traced over OpenTelemetry — the "any framework" path.

Nothing here is CrewAI-specific except the instrumentor on line ~60. Swap it
for `openinference-instrumentation-llama-index`, `…-smolagents`,
`…-autogen`, or drop it entirely for a framework that is already OTel-native
(Pydantic AI, Google ADK, AWS Strands, Semantic Kernel) and the rest is
unchanged. That is the point of this route.

Demonstrates:

  * `CognipeerSpanExporter` in an ordinary `TracerProvider` — the exporter
    normalises OpenInference, current OTel GenAI and legacy OpenLLMetry
    attributes into one model, so it does not matter which convention the
    instrumentor emits;
  * `thread_id` passed to the exporter, because no instrumentation emits a
    conversation id on its own. Without it every run arrives as its own
    unrelated session.

⚠️ Calls a model API — this one costs money.

    pip install "cognipeer-observability[otel]" crewai openinference-instrumentation-crewai
    export OPENAI_API_KEY=sk-…
    python examples/python/otel_crewai.py

In Console: Tracing → Sessions → the service name below. A session is closed
when its root span ends, or after `session_idle_seconds` of silence — OTel has
no end-of-run signal, so that timeout is the backstop.

VERSION NOTE, verified against crewAIInc/crewAI 1.15.16: CrewAI's own event
bus moved from `crewai.utilities.events` to `crewai.events` in 0.177.0 and the
old path was REMOVED in 1.0.0. This example does not touch either — it goes
through the OpenInference instrumentor, which is the stable seam. A first-class
CrewAI integration using that event bus is on the roadmap; it reports token
usage more directly than the OTel path does.
"""

from __future__ import annotations

import os
import sys

import cognipeer_observability as cognipeer
from crewai import Agent, Crew, Task
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ─────────────────────────────────────────────────────────────────────
#  Wire tracing in — this is the whole integration
# ─────────────────────────────────────────────────────────────────────
from cognipeer_observability.otel import CognipeerSpanExporter

for required in ("COGNIPEER_API_KEY", "OPENAI_API_KEY"):
    if not os.environ.get(required):
        sys.exit(f"Set {required} before running this example.")

cognipeer.init()

# `service.name` is what the exporter falls back to for the agent name, so set
# it deliberately — it is what the Agents screen groups runs by.
provider = TracerProvider(resource=Resource.create({"service.name": "research-crew"}))
provider.add_span_processor(
    BatchSpanProcessor(CognipeerSpanExporter(thread_id="conv-42"))
)

from openinference.instrumentation.crewai import CrewAIInstrumentor  # noqa: E402

CrewAIInstrumentor().instrument(tracer_provider=provider)
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    researcher = Agent(
        role="Researcher",
        goal="Find one concrete fact about the topic",
        backstory="You are terse and cite nothing you cannot verify.",
        verbose=False,
    )
    task = Task(
        description="State one verifiable fact about the Danube.",
        expected_output="A single sentence.",
        agent=researcher,
    )

    result = Crew(agents=[researcher], tasks=[task], verbose=False).kickoff()
    print(result)

    # Flush the OTel pipeline first — spans are batched, and the exporter can
    # only send what it has been given.
    provider.force_flush()
    cognipeer.flush()
    print("\nLook in Console → Tracing → Sessions → agent 'research-crew'")


if __name__ == "__main__":
    main()
