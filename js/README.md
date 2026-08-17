<div align="center">

# @cognipeer/observability

**Ship agent traces from LangChain, LangGraph, the OpenAI Agents SDK, the Claude Agent SDK, Vercel AI SDK, n8n or any OpenTelemetry-instrumented agent into [Cognipeer Console](https://github.com/Cognipeer/console).**

[![npm](https://img.shields.io/npm/v/@cognipeer/observability?label=%40cognipeer%2Fobservability)](https://www.npmjs.com/package/@cognipeer/observability)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

</div>

You already built the agent. This makes it observable — prompts, completions,
tool calls, token usage, cost and latency — without rewriting it.

```ts
import { init } from '@cognipeer/observability';
import { CognipeerCallbackHandler } from '@cognipeer/observability/langchain';

init({ apiKey: 'cpeer_…', agent: { name: 'support-bot' } });
await agent.invoke(state, { callbacks: [new CognipeerCallbackHandler()] });
```

That is the whole integration. Two lines, no proxy, no base-URL swap, no
change to how your agent calls its models.

This package is the TypeScript half of [`cognipeer-observability`](https://github.com/Cognipeer/cognipeer-observability),
an MIT-licensed monorepo also publishing [`cognipeer-observability`](https://pypi.org/project/cognipeer-observability/)
for Python. See the [full documentation and per-framework guides](https://cognipeer.github.io/console/guide/observability/overview).

## Install

```bash
npm install @cognipeer/observability
```

The core has **no required dependencies**. Framework integrations are
optional peers, so installing this package cannot drag your dependency tree
around — nothing loads until you import its subpath.

## Subpath exports

| Subpath | Framework | Guide |
|---|---|---|
| `@cognipeer/observability` | core client, `observe`/`trace`, session API | [Manual instrumentation](https://cognipeer.github.io/console/guide/observability/manual) |
| `@cognipeer/observability/langchain` | LangChain callback handler (0.1 → 1.x) | [LangChain](https://cognipeer.github.io/console/guide/observability/langchain) |
| `@cognipeer/observability/langgraph` | LangGraph on top of the same handler | [LangGraph](https://cognipeer.github.io/console/guide/observability/langgraph) |
| `@cognipeer/observability/openai-agents` | OpenAI Agents SDK `TracingProcessor` | [OpenAI Agents SDK](https://cognipeer.github.io/console/guide/observability/openai-agents) |
| `@cognipeer/observability/claude-agent-sdk` | Claude Agent SDK message-stream tracer | [Claude Agent SDK](https://cognipeer.github.io/console/guide/observability/claude-agent-sdk) |
| `@cognipeer/observability/vercel-ai` | Model middleware / `experimental_telemetry` | [Vercel AI SDK](https://cognipeer.github.io/console/guide/observability/vercel-ai) |
| `@cognipeer/observability/otel` | OpenTelemetry span exporter | [Anything OpenTelemetry](https://cognipeer.github.io/console/guide/observability/opentelemetry) |
| `cognipeer-n8n` CLI (see below) | n8n execution bridge | [n8n](https://cognipeer.github.io/console/guide/observability/n8n) |

Not on the list? If it emits OpenTelemetry spans — natively or through an
OpenInference/OpenLLMetry instrumentor — the OTel route already covers it.

## Configure

Everything reads from the environment, so the code above works unchanged
across dev, staging and production:

| Variable | Default | Meaning |
|---|---|---|
| `COGNIPEER_API_KEY` | — | Console API token. Without it, tracing disables itself and warns once — it never throws. |
| `COGNIPEER_BASE_URL` | `https://console.cognipeer.com` | Your Console, for self-hosted installs |
| `COGNIPEER_AGENT_NAME` | — | Default agent name for every session |
| `COGNIPEER_AGENT_VERSION` | — | Default agent version for every session |
| `COGNIPEER_CAPTURE_CONTENT` | `all` | `all`, `metadata` (structure and tokens, no message bodies), or `none` |
| `COGNIPEER_TRACING_ENABLED` | `true` | Master switch |
| `COGNIPEER_TRACING_MODE` | `auto` | `auto`, `stream` (live updates) or `batch` (one request per run) |
| `COGNIPEER_DEBUG` | `false` | Log what the exporter is doing |

Any of these can also be passed as explicit options to `init()`, which take
priority over the environment.

## API

```ts
import {
  init,          // configure the default client (idempotent, safe to call once at boot)
  getClient,     // the resolved CognipeerObservability instance
  resetClient,   // mostly for tests
  observe,       // wrap an async function as a traced span
  trace,         // start/end a span imperatively
  TraceSession,  // the session primitive every integration is built on
  flush,         // await before a short-lived process (script, Lambda, CI job) exits
  shutdown,      // end every open session and flush; safe to call more than once
} from '@cognipeer/observability';
```

Long-running services never need `flush`/`shutdown` — exports run on a
background promise chain, off the request's hot path.

### `cognipeer-n8n` CLI

Mirrors n8n workflow executions into Console without touching the n8n
instance itself:

```bash
npx @cognipeer/observability cognipeer-n8n \
  --n8n-url https://n8n.acme.com \
  --n8n-api-key $N8N_API_KEY \
  --api-key $COGNIPEER_API_KEY \
  --once            # mirror the current page and exit — good for cron/CI
```

Every flag also reads an environment variable (`N8N_URL`, `N8N_API_KEY`,
`COGNIPEER_API_KEY`, `COGNIPEER_BASE_URL`, …), so a container needs no
arguments at all. See the [n8n guide](https://cognipeer.github.io/console/guide/observability/n8n)
for the execution-bridge vs. webhook-hook tradeoffs.

## Design rules

- **Tracing never breaks the traced application.** Every export path swallows
  its own failures and reports them through `onError`. A missing API key
  disables the exporter; it does not throw.
- **Tracing never blocks the traced application.** Exports run on a
  background promise chain — no integration awaits network I/O on a
  framework's hot path.
- **No hidden dependencies.** The core is dependency-free; every framework
  import is behind its own subpath.
- **Secrets and blobs do not leave your process.** API keys in prompts are
  redacted by pattern, base64 data URLs are stripped, and oversized content is
  capped before it is sent.

## Links

- [Cognipeer Console](https://github.com/Cognipeer/console) — the product this feeds
- [Full documentation](https://cognipeer.github.io/console/guide/observability/overview)
- [Data model](https://github.com/Cognipeer/cognipeer-observability/blob/main/docs/data-model.md) every integration maps onto
- [Examples](https://github.com/Cognipeer/cognipeer-observability/tree/main/examples/js)
- [Source repository](https://github.com/Cognipeer/cognipeer-observability) (monorepo, `js/` directory)
- [Issues](https://github.com/Cognipeer/cognipeer-observability/issues)

## Licence

MIT — see [LICENSE](./LICENSE). Cognipeer Console itself is licensed separately.
