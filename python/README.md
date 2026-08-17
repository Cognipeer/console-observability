<div align="center">

# cognipeer-observability

**Ship agent traces from LangChain, LangGraph, the OpenAI Agents SDK, the Claude Agent SDK or any OpenTelemetry-instrumented agent into [Cognipeer Console](https://github.com/Cognipeer/console).**

[![PyPI](https://img.shields.io/pypi/v/cognipeer-observability?label=cognipeer-observability)](https://pypi.org/project/cognipeer-observability/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

</div>

You already built the agent. This makes it observable — prompts, completions,
tool calls, token usage, cost and latency — without rewriting it.

```python
import cognipeer_observability as cognipeer
from cognipeer_observability.langchain import CognipeerCallbackHandler

cognipeer.init(api_key="cpeer_…", agent={"name": "support-bot"})
agent.invoke(state, config={"callbacks": [CognipeerCallbackHandler()]})
```

That is the whole integration. Two lines, no proxy, no base-URL swap, no
change to how your agent calls its models.

This package is the Python half of [`cognipeer-observability`](https://github.com/Cognipeer/cognipeer-observability),
an MIT-licensed monorepo also publishing [`@cognipeer/observability`](https://www.npmjs.com/package/@cognipeer/observability)
for TypeScript. See the [full documentation and per-framework guides](https://cognipeer.github.io/console/guide/observability/overview).

## Install

Extras pull in only what you use — the core has **no required dependencies**
(`typing_extensions` on Python < 3.11 only):

```bash
pip install cognipeer-observability                       # core only
pip install "cognipeer-observability[langchain]"
pip install "cognipeer-observability[langgraph]"
pip install "cognipeer-observability[openai-agents]"
pip install "cognipeer-observability[claude-agent-sdk]"
pip install "cognipeer-observability[otel]"
pip install "cognipeer-observability[all]"                 # every integration
```

Requires Python 3.9+.

## Modules

| Module | Framework | Guide |
|---|---|---|
| `cognipeer_observability` | core client, `observe`/`trace`, session API | [Manual instrumentation](https://cognipeer.github.io/console/guide/observability/manual) |
| `cognipeer_observability.langchain` | LangChain callback handler (0.1 → 1.x) | [LangChain](https://cognipeer.github.io/console/guide/observability/langchain) |
| `cognipeer_observability.langgraph` | LangGraph helpers on top of it | [LangGraph](https://cognipeer.github.io/console/guide/observability/langgraph) |
| `cognipeer_observability.openai_agents` | OpenAI Agents SDK tracing processor | [OpenAI Agents SDK](https://cognipeer.github.io/console/guide/observability/openai-agents) |
| `cognipeer_observability.claude_agent_sdk` | Claude Agent SDK hooks + message tracer | [Claude Agent SDK](https://cognipeer.github.io/console/guide/observability/claude-agent-sdk) |
| `cognipeer_observability.otel` | OpenTelemetry span exporter | [Anything OpenTelemetry](https://cognipeer.github.io/console/guide/observability/opentelemetry) |

Importing a framework module without its dependency installed raises a clear
`ImportError` naming the extra to install — never a bare traceback.

## Configure

Everything reads from the environment, so the code above works unchanged
across dev, staging and production:

| Variable | Default | Meaning |
|---|---|---|
| `COGNIPEER_API_KEY` | — | Console API token. Without it, tracing disables itself and warns once — it never raises. |
| `COGNIPEER_BASE_URL` | `https://console.cognipeer.com` | Your Console, for self-hosted installs |
| `COGNIPEER_AGENT_NAME` | — | Default agent name for every session |
| `COGNIPEER_AGENT_VERSION` | — | Default agent version for every session |
| `COGNIPEER_CAPTURE_CONTENT` | `all` | `all`, `metadata` (structure and tokens, no message bodies), or `none` |
| `COGNIPEER_TRACING_ENABLED` | `true` | Master switch |
| `COGNIPEER_TRACING_MODE` | `auto` | `auto`, `stream` (live updates) or `batch` (one request per run) |
| `COGNIPEER_DEBUG` | `false` | Log what the exporter is doing |

Any of these can also be passed as explicit keyword arguments to `init()`,
which take priority over the environment.

## API

```python
from cognipeer_observability import (
    init,               # configure the default client (idempotent, safe to call once at boot)
    get_client,          # the resolved Cognipeer instance
    reset_client,        # mostly for tests
    observe,             # decorator: wrap a function/coroutine as a traced span
    trace,               # context manager: start/end a span imperatively
    TraceSession,        # the session primitive every integration is built on
    use_session,         # context manager: bind the active session
    use_span,            # context manager: bind the active span
    flush,               # block until every queued trace is delivered
    shutdown,            # end every open session and flush; safe to call more than once
)
```

Long-running services rarely need `flush`/`shutdown` — exports run on a
background thread, off the traced code's hot path. Call `flush()` before a
short-lived process exits (a script, a Lambda handler, a CI job).

## Design rules

- **Tracing never breaks the traced application.** Every export path swallows
  its own failures and reports them through `on_error`. A missing API key
  disables the exporter; it does not raise.
- **Tracing never blocks the traced application.** Exports run on a
  background thread — no integration awaits network I/O on a framework's hot
  path.
- **No hidden dependencies.** The core is standard-library only; every
  framework import is lazy and optional.
- **Secrets and blobs do not leave your process.** API keys in prompts are
  redacted by pattern, base64 data URLs are stripped, and oversized content is
  capped before it is sent.

## Links

- [Cognipeer Console](https://github.com/Cognipeer/console) — the product this feeds
- [Full documentation](https://cognipeer.github.io/console/guide/observability/overview)
- [Data model](https://github.com/Cognipeer/cognipeer-observability/blob/main/docs/data-model.md) every integration maps onto
- [Examples](https://github.com/Cognipeer/cognipeer-observability/tree/main/examples/python)
- [Source repository](https://github.com/Cognipeer/cognipeer-observability) (monorepo, `python/` directory)
- [Issues](https://github.com/Cognipeer/cognipeer-observability/issues)

## Licence

MIT — see [LICENSE](./LICENSE). Cognipeer Console itself is licensed separately.
