# Changelog

All notable changes to this project are documented here. The two packages —
`@cognipeer/observability` (npm) and `cognipeer-observability` (PyPI) — share
one version number and one changelog.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0]

### Added

- **Session-level `metadata`** (`init({ metadata })` / client default, and a
  per-`startSession`/`trace` override merged on top) — free-form attribution
  tags reported alongside `agent` on every session payload (batch, stream
  `/start`, and the streaming reopen path). The Console now supports grouping
  and reporting on these as a dynamic `group_by`/`group_by_entity=metadata.<key>`
  dimension, so callers can slice spend by anything they tag a session with
  (e.g. `{ complexity: "complex" }`) without a schema change on either side.
  Distinct from the existing per-event `metadata` — that one is free-text
  content (redacted, size-capped); this one is short, structured attribution
  tags and is never redacted or capped client-side.

## [0.1.0]

First release.

### Added

- **Core**, in Python and TypeScript, with no required dependencies in either:
  session/event/section model, background delivery with retry, secret
  redaction, base64 stripping, content capping, and three delivery modes
  (`auto`, `stream`, `batch`).
- **LangChain** callback handler, working from `langchain-core` 0.1 through
  1.x and `@langchain/core` 0.1 through 1.x. Captures prompts, completions,
  per-call token usage with the cache-read breakdown, tool calls with
  arguments and results, and the tool definitions bound to each model call.
- **LangGraph** support on the same handler, plus `graph_config` /
  `langgraphConfig` and `trace_graph` / `withCognipeerTracing`. Interrupts are
  recorded as control flow rather than failures, and a conversation's runs
  group by thread id.
- **OpenAI Agents SDK** tracing processor for both SDKs, mapping every span
  kind and counting tokens only from leaf model spans.
- **Claude Agent SDK** message-stream tracer and a `trace_query` /
  `traceQuery` drop-in for `query()`, pairing each `tool_use` with its
  `tool_result` and translating Anthropic's cache-exclusive token accounting.
- **Vercel AI SDK** integration with three routes — the native telemetry
  integration (ai 6+), a language-model middleware (every version), and
  `experimental_telemetry` with a bundled tracer (ai 3–6).
- **n8n** integration: a polling bridge with a `cognipeer-n8n` CLI that works
  on n8n Cloud and Community, and a `workflow.postExecute` external hook for
  self-hosted installs.
- **OpenTelemetry** span exporter in both languages, normalising OpenInference,
  current OTel GenAI and legacy OpenLLMetry attributes — which is what makes
  CrewAI, LlamaIndex, Pydantic AI, Google ADK, AWS Strands, Semantic Kernel,
  smolagents, Haystack and DSPy work without a bespoke integration.
- **Manual instrumentation**: `@observe` / `observe()` and `trace()` for agents
  no framework covers, with sync, async, generator and async-generator support
  in Python.
- Runnable examples per framework and a full guide per integration in the
  [Console documentation](https://cognipeer.github.io/console/guide/observability/overview).

[Unreleased]: https://github.com/Cognipeer/cognipeer-observability/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Cognipeer/cognipeer-observability/releases/tag/v0.1.0
