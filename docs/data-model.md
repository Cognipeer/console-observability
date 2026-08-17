# The data model

Everything this package does is a mapping: from a framework's own telemetry
onto the model below. Adding a framework means writing that mapping and
nothing else, so this page is the reference an integration is written against.

## Session → events

One **session** is one agent run — what the Console shows on
`/dashboard/tracing/sessions/:id`. It contains an ordered list of **events**,
each of which is one step of the run.

```
session  { sessionId, threadId, agent, status, startedAt, endedAt, summary }
   └── event  { type, label, spanId, parentSpanId, model, tokens, sections }
        └── section  { kind, role, content }
```

**Threads** group sessions. A multi-turn conversation is several sessions
sharing one `threadId`; that is also how a LangGraph conversation that pauses
for human input and resumes is reassembled, because every resume is a fresh
run with a fresh trace.

## Session fields

| Field | Type | Notes |
|---|---|---|
| `sessionId` | string | Re-posting the same id updates the session. Generated per run unless you pass one. |
| `threadId` | string | Conversation key. This is what makes the Threads view useful — set it whenever your framework has one. |
| `traceId` | 32 hex | W3C trace id, so a run can be correlated with another OTel backend. |
| `rootSpanId` | 16 hex | Parent of every top-level event. |
| `agent` | `{name, version, model, provider}` | `name` is what the Agents screen groups by. |
| `status` | `success` \| `error` \| `in_progress` | |
| `startedAt` / `endedAt` | ISO 8601 | |
| `summary` | totals | `totalInputTokens`, `totalOutputTokens`, `totalCachedInputTokens`, `totalDurationMs`, `eventCounts` |
| `config` | object | Free-form run configuration, shown on the session header. |
| `errors` | array | Collected failures; a non-empty list marks the session failed. |

## Event types

The Console aggregates per type, so use these rather than inventing names:

| `type` | Use for |
|---|---|
| `ai_call` | A model call. The only type that should carry `model` and token counts. |
| `tool_call` | A tool / function invocation. Set `toolName`. |
| `retrieval` | RAG or vector search. |
| `embedding` | An embedding call. |
| `summarization` | History compaction. |
| `guardrail` | A policy or safety check. |
| `span` | Anything else — a graph node, a chain step, a custom block of work. |

## Event fields

| Field | Type | Notes |
|---|---|---|
| `id` | string | Stable per event; used to de-duplicate on re-ingest. |
| `label` | string | What the timeline row says. Node name, tool name or model name. |
| `spanId` / `parentSpanId` | 16 hex | Builds the tree. See *Identifiers* below. |
| `sequence` | number | Assigned by the SDK; the timeline orders by it. |
| `timestamp` | ISO 8601 | Start of the step. |
| `durationMs` | number | |
| `status` | `success` \| `error` | |
| `error` | string or object | Attached to a failed step. |
| `model` | string | **Must be the provider's model id** (`gpt-4.1-mini`), not a nickname — cost resolution matches it against Model Hub keys, provider model ids and the external pricing catalogue. |
| `inputTokens` / `outputTokens` / `cachedInputTokens` | number | See *Tokens*. |
| `reasoningTokens` | number | Subset of `outputTokens`. See *Tokens*. |
| `finishReason` | string | `stop` \| `tool_calls` \| `length` \| `content_filter`. |
| `toolName` / `toolExecutionId` | string | |
| `toolDefinitions` | array | The tool menu offered on this call. See below. |
| `responseFormat` | object | The structured-output contract enforced on this call. See below. |
| `actor` | `{scope, name}` | `scope` is `agent`, `model`, `tool`, `retriever` or `user`. |
| `sections` | array | The renderable body. See below. |
| `metadata` | object | Anything else worth keeping; shown as a key/value block. |

## Sections

A section is one renderable block. `kind` drives the badge colour; every other
key is rendered as a labelled field, so extra keys are safe.

| `kind` | Shape |
|---|---|
| `message` | `{ role: 'system'\|'user'\|'assistant'\|'tool', content }` |
| `tool_call` | `{ tool, content }` — the arguments |
| `tool_result` | `{ tool, content }` — the result |
| `tool_definitions` | `{ tools: [{name, description?, parameters?}] }` |
| `response_format` | `{ type, schemaName?, strict?, schema? }` |
| `metadata` | `{ content }` — anything structural |

```json
{
  "kind": "message",
  "label": "User message",
  "role": "user",
  "content": "book me a flight to Rome"
}
```

## Tool definitions

`toolDefinitions` records **the menu the model was offered on that call**, not
the tool set the agent was configured with. The menu changes between turns,
and a large one is frequently the single biggest line item in an agent's
prompt bill — which is why it is captured per event rather than per session.

```json
{
  "type": "ai_call",
  "toolDefinitions": [
    { "name": "search_flights",
      "description": "Search flights by city",
      "parameters": { "type": "object", "properties": { "city": { "type": "string" } } } }
  ]
}
```

The server normalises this into a `tool_definitions` section, accepts the
OpenAI `{type: 'function', function: {…}}` envelope and Anthropic's
`input_schema` alias, and caps oversized schemas (entries keep `name` and
`description` and drop `parameters` with `"truncated": true`).

Not every framework can tell us this. Where it cannot, the integration says so
in its guide rather than inventing a menu — except n8n, where a menu
reconstructed from the workflow definition is emitted with
`metadata.toolDefinitionsSource: 'n8n-workflow-json'` so you know what you are
looking at.

## Response format

`responseFormat` is the other half of the request's shape: **the
structured-output contract enforced on that call**. It is captured per event
for the same reason the menu is — an agent may enforce a schema on its final
turn only — and it answers a question the messages cannot: a reply that is not
valid JSON is a *defect* when a schema was enforced and a *choice* when none
was.

```json
{
  "type": "ai_call",
  "responseFormat": {
    "type": "json_schema",
    "name": "invoice_v2",
    "strict": true,
    "schema": { "type": "object", "properties": { "total": { "type": "number" } } }
  }
}
```

`type` is the wire `response_format.type` (`json_schema`, `json_object`,
`text`). JSON mode with no schema is still a contract, and is recorded as
`{"type": "json_object"}`. Omit the field entirely for an unconstrained call —
an absent contract and a `text` one are different facts.

The server normalises this into a `response_format` section, accepts the
OpenAI envelope (`{type, json_schema: {name, strict, schema}}`) and the
agent-sdk's `{response_format: …}` wrapper, and caps oversized schemas (the
entry keeps `type`/`schemaName`/`strict` and drops `schema` with
`"truncated": true`).

Where the integrations read it from: LangChain's `invocation_params`, the AI
SDK's `responseFormat` call option (and `generateObject`'s `ai.schema`
telemetry), the OpenAI Agents Responses `text.format` and agent `output_type`,
OpenInference's `llm.invocation_parameters`, and the OTel GenAI pair
`gen_ai.output.type` + `gen_ai.request.structured_output_schema`.

Beyond the trace UI this is what lets a captured run be **replayed faithfully**:
Traffic Snapshots copy the recorded contract onto each dataset item, and
evaluation suites and prompt-optimizer runs send it back on the wire. A replay
that drops the schema is measuring a looser system than production runs under.

## Tokens

`cachedInputTokens` is a **subset** of `inputTokens`, matching OpenAI's
`prompt_tokens_details.cached_tokens` and LangChain's standardised
`input_token_details.cache_read`. Cost is computed as
`(inputTokens - cachedInputTokens)` at the input rate plus `cachedInputTokens`
at the cached rate.

Anthropic reports it the other way round — its `input_tokens` **excludes**
cache reads — so the Claude integrations add the cache buckets back in:

```
inputTokens       = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
cachedInputTokens = cache_read_input_tokens
```

`reasoningTokens` is likewise a **subset** of `outputTokens`, matching OpenAI's
`completion_tokens_details.reasoning_tokens`. It is recorded for attribution
only and never added to the bill — the tokens are already inside the output
count, and adding them would double-charge. On a reasoning model it is
routinely most of the output spend while being invisible in the response text,
which is why a cost investigation that only sees `outputTokens` cannot explain
where the money went.

When a framework reports no usage at all — a streaming call without usage
opt-in, a cancelled run — the fields are **omitted**, never set to zero. A
zero would silently under-report spend; an absent value shows up as unknown.

## Finish reason

`finishReason` is why the model stopped. Record it whenever the framework
exposes it: `length` is the single most common explanation for a truncated or
unparseable structured response, and without it that failure is
indistinguishable from a model that simply answered badly.

### Double counting

A trace event describing a model call that was **served by this Console's
gateway** is already billed at serving time. Mark such events with
`metadata.gateway = true` (or a `gatewayRequestId`) and the cost pipeline will
skip them. Direct-to-provider calls — the normal case for these integrations —
need no marker.

## Identifiers

`traceId` is 32 lower-hex characters and `spanId` is 16, per W3C. Framework run
ids are usually UUIDs, so the SDK folds them deterministically:

- an input that is already exactly 16 (or 32) hex characters passes through;
- anything else is hashed.

Truncation is deliberately *not* used. LangChain run ids are UUIDv7, whose
first 16 hex digits are a millisecond timestamp plus 12 bits of entropy — two
runs started in the same millisecond would collide, and colliding span ids
silently corrupt the tree.

Because the fold is deterministic, a child can compute its parent's span id
from a parent run id it never saw as an event of its own.

## Delivery

Two wire shapes, chosen by `mode`:

| Mode | Requests | When |
|---|---|---|
| `batch` | one, at the end | Short runs, serverless, cron. The endpoint **replaces** the session's whole event list, so it is idempotent. |
| `stream` | `/start`, one per event, `/end` | Long runs you want to watch live. Events **append**. |
| `auto` (default) | either | Buffers, then switches to streaming once the run passes `streamAfterMs` (2 s) or `streamAfterEvents` (25). One request for a quick run, live updates for a slow one. |

A session commits to one shape the first time it delivers anything and never
switches back — mixing them would let a batch post wipe streamed events.

## Endpoints

All under `POST /api/client/v1`, authenticated with `Authorization: Bearer <token>`:

| Path | Purpose |
|---|---|
| `/tracing/sessions` | Batch: a whole session in one request |
| `/tracing/sessions/stream/:id/start` | Open a streaming session |
| `/tracing/sessions/stream/:id/events` | Append one event |
| `/tracing/sessions/stream/:id/end` | Close, with final totals |
| `/traces` | OTLP/HTTP **JSON** `ExportTraceServiceRequest` |

The request body limit is `TRACING_MAX_BODY_SIZE_MB` (10 MB by default), which
is why the SDK caps section content and strips base64 blobs before sending.
