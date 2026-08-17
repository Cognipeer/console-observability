/**
 * Wire types for the Cognipeer Console tracing ingest API.
 *
 * These mirror `POST /api/client/v1/tracing/sessions` (batch) and
 * `POST /api/client/v1/tracing/sessions/stream/:sessionId/{start,events,end}`
 * (streaming). Field names are the server's — do not rename them here.
 */

/** Canonical event types the Console understands and aggregates on. */
export type TraceEventType =
  /** A model call (chat/completion). Carries tokens, model and tool menu. */
  | 'ai_call'
  /** A tool / function invocation. */
  | 'tool_call'
  /** RAG / vector retrieval. */
  | 'retrieval'
  /** Embedding call. */
  | 'embedding'
  /** History summarization / compaction step. */
  | 'summarization'
  /** Guardrail / policy check. */
  | 'guardrail'
  /** Anything else — shown verbatim, still counted per type. */
  | 'span'
  | (string & {});

/** Status values the Console colours and aggregates on. */
export type TraceStatus = 'success' | 'error' | 'in_progress' | (string & {});

/**
 * A renderable block inside an event. `kind` drives the badge colour in the
 * tracing detail UI; every other key is rendered generically as a labelled
 * field, so extra keys are safe.
 */
export interface TraceSection extends Record<string, unknown> {
  kind: 'message' | 'tool_call' | 'tool_result' | 'tool_definitions' | 'response_format' | 'metadata' | (string & {});
  label?: string;
  role?: 'system' | 'user' | 'assistant' | 'tool' | (string & {});
  content?: unknown;
  /** Tool this section belongs to (tool_call / tool_result). */
  tool?: string;
  /** Only on `kind: 'tool_definitions'`. */
  tools?: TraceToolDefinition[];
  /** Set when the producer dropped data to fit a size budget. */
  truncated?: true;
}

/**
 * The structured-output contract enforced on ONE model call — the other half
 * of the request shape next to the tool menu.
 *
 * Worth capturing for the same reason the menu is: messages alone cannot tell
 * a model that CHOSE prose from one that was never asked for JSON, and a
 * replay of the call (evaluation, prompt optimization) that drops the schema
 * measures a looser system than production runs under. Per event, never per
 * session: an agent may enforce a schema on its final turn only.
 */
export interface TraceResponseFormat {
  /** The wire `response_format.type`: `json_schema` | `json_object` | `text`. */
  type: string;
  /** Named schema (`json_schema.name`) when the provider takes one. */
  name?: string;
  /** Whether the provider enforces the schema (`json_schema.strict`). */
  strict?: boolean;
  /** The JSON Schema as sent. */
  schema?: Record<string, unknown>;
  /** How it was enforced: a native `response_format`, or a forced tool call. */
  strategy?: 'native' | 'tool_based';
}

/** One entry of the tool menu offered to the model on a given call. */
export interface TraceToolDefinition {
  name: string;
  description?: string;
  /** JSON Schema object describing the tool input. */
  parameters?: Record<string, unknown>;
  truncated?: true;
}

/** Who produced the event — drives the actor column in the UI. */
export interface TraceActor {
  name?: string;
  role?: string;
  /** `agent` | `tool` | `model` | `user` | … — `tool` also sets `toolName`. */
  scope?: string;
}

export interface TraceUsage {
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
}

export interface TraceEvent {
  /** Stable id. Also used for de-duplication on the OTLP path. */
  id?: string;
  type?: TraceEventType;
  label?: string;
  sequence?: number;
  timestamp?: string;
  status?: TraceStatus;
  /** W3C trace id (32 hex chars) — correlates with other OTel backends. */
  traceId?: string;
  /** W3C span id (16 hex chars). */
  spanId?: string;
  /** Parent span id — this is what builds the tree in the UI. */
  parentSpanId?: string;

  actor?: TraceActor;
  sections?: TraceSection[];
  metadata?: Record<string, unknown>;

  /**
   * Model name for `ai_call` events. Cost accounting resolves this against
   * Model Hub keys / provider model ids / the external pricing catalog, so it
   * should be the provider's model id (e.g. `gpt-4.1-mini`), not a nickname.
   */
  model?: string;
  modelNames?: string[];

  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  /**
   * Reasoning tokens billed inside `outputTokens` (a SUBSET of it, matching
   * OpenAI's `completion_tokens_details.reasoning_tokens`). Recorded because on
   * a reasoning model they are routinely most of the output bill while being
   * invisible in the response text. NOT re-billed — they are already counted.
   */
  reasoningTokens?: number;
  /**
   * Why the model stopped: `stop` | `tool_calls` | `length` | `content_filter`.
   * `length` is the single most common explanation for a truncated or
   * unparseable structured response; without it that failure is
   * indistinguishable from a model that simply answered badly.
   */
  finishReason?: string;
  totalTokens?: number;
  requestBytes?: number;
  responseBytes?: number;
  durationMs?: number;

  toolName?: string;
  toolExecutionId?: string;
  /** Normalized server-side into a `tool_definitions` section. */
  toolDefinitions?: TraceToolDefinition[];
  /** Normalized server-side into a `response_format` section. Accepts the
   *  OpenAI `response_format` body directly. */
  responseFormat?: TraceResponseFormat | Record<string, unknown>;

  error?: string | Record<string, unknown>;
  usage?: TraceUsage;
}

export interface TraceAgent {
  name?: string;
  version?: string;
  /** Default model of the agent; shown on the session header. */
  model?: string;
  provider?: string;
  [key: string]: unknown;
}

export interface TraceSummary {
  eventCounts?: Record<string, number>;
  totalInputTokens?: number;
  totalOutputTokens?: number;
  totalCachedInputTokens?: number;
  totalDurationMs?: number;
  totalBytesIn?: number;
  totalBytesOut?: number;
}

export interface TraceSessionPayload {
  sessionId: string;
  /** Groups several sessions into one conversation in the Threads view. */
  threadId?: string;
  traceId?: string;
  rootSpanId?: string;
  agent?: TraceAgent;
  config?: Record<string, unknown>;
  /** Free-form attribution tags (e.g. `{ complexity: 'complex' }`), sibling of
   *  `agent`. The Console groups/reports on these as a dynamic
   *  `group_by`/`group_by_entity=metadata.<key>` dimension — a schema-free
   *  alternative to `agent.name` for slicing spend by whatever the caller
   *  cares about. String values only; unlike event `metadata` this is never
   *  redacted or size-capped client-side (it rides in the Console's own
   *  bounded attribution index, not free-text content). */
  metadata?: Record<string, string>;
  status?: TraceStatus;
  startedAt?: string;
  endedAt?: string;
  durationMs?: number;
  summary?: TraceSummary;
  errors?: Array<Record<string, unknown> | string>;
  events?: TraceEvent[];
}

export interface StreamStartPayload {
  threadId?: string;
  traceId?: string;
  rootSpanId?: string;
  agent?: TraceAgent;
  config?: Record<string, unknown>;
  /** See `TraceSessionPayload.metadata`. */
  metadata?: Record<string, string>;
  startedAt?: string;
}

export interface StreamEndPayload {
  status?: TraceStatus;
  endedAt?: string;
  durationMs?: number;
  summary?: TraceSummary;
  errors?: Array<Record<string, unknown> | string>;
}

/** How much prompt/completion content leaves the process. */
export type CaptureMode =
  /** Everything, redacted by the configured patterns (default). */
  | 'all'
  /** Structure, token counts and tool names only — no message content. */
  | 'metadata'
  /** Nothing at all — the transport is disabled. */
  | 'none';

export interface CognipeerConfig {
  /** Console API token. Defaults to `COGNIPEER_API_KEY`. */
  apiKey?: string;
  /**
   * Console host root, e.g. `https://console.cognipeer.com`. A legacy value
   * ending in `/api/client/v1` is accepted and trimmed. Defaults to
   * `COGNIPEER_BASE_URL`.
   */
  baseUrl?: string;
  /** Default agent identity stamped on every session. */
  agent?: TraceAgent;
  /** Default attribution tags stamped on every session; see
   *  `TraceSessionPayload.metadata`. Per-session values (`startSession`/
   *  `trace` `metadata` option) are merged on top, same as `agent`. */
  metadata?: Record<string, string>;
  /** Default thread id — usually set per conversation instead. */
  threadId?: string;
  /** Master switch. Defaults to `COGNIPEER_TRACING_ENABLED` (true). */
  enabled?: boolean;
  /** Content capture policy. Defaults to `COGNIPEER_CAPTURE_CONTENT` (`all`). */
  capture?: CaptureMode;
  /** Extra regexes whose matches are replaced with `[redacted]`. */
  redactPatterns?: RegExp[];
  /** Per-section content cap in characters (default 50_000). */
  maxContentChars?: number;
  /**
   * Transport strategy:
   *  - `auto` (default): buffer, then switch to streaming once the run looks
   *    long-lived, otherwise post one batch at the end — 1 request for short
   *    runs, live updates for long ones.
   *  - `stream`: always `/start` → `/events` → `/end`.
   *  - `batch`: always one `/sessions` post when the session ends.
   */
  mode?: 'auto' | 'stream' | 'batch';
  /** `auto` mode: switch to streaming after this long (default 2000 ms). */
  streamAfterMs?: number;
  /** `auto` mode: switch to streaming after this many events (default 25). */
  streamAfterEvents?: number;
  /** HTTP timeout per request in ms (default 30_000). */
  timeout?: number;
  /** Retry attempts for retryable failures (default 3). */
  maxRetries?: number;
  /** Extra headers on every request. */
  headers?: Record<string, string>;
  /** Custom fetch (tests, proxies, non-Node runtimes). */
  fetch?: typeof fetch;
  /** Log transport activity. Defaults to `COGNIPEER_DEBUG`. */
  debug?: boolean;
  /** Sink for the SDK's own diagnostics. Defaults to `console`. */
  logger?: Logger;
  /**
   * Called instead of throwing when the transport fails. Observability must
   * never break the traced application.
   */
  onError?: (error: Error) => void;
}

export interface Logger {
  debug: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
}
