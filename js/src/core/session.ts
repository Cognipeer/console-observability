/**
 * A tracing session: the unit the Console shows as one agent run.
 *
 * Sessions buffer events and choose how to deliver them (see `mode` in
 * `CognipeerConfig`). The two delivery shapes are not interchangeable
 * server-side: the batch endpoint REPLACES a session's whole event list,
 * while the streaming endpoints append. A session therefore commits to one
 * shape the first time it delivers anything and never switches back.
 */

import { reportSafely, type ResolvedConfig } from './config';
import { newSessionId, newSpanId, newTraceId, spanIdFrom } from './ids';
import { sanitizeMetadata, sanitizeSections, stringifyContent } from './redact';
import type { Transport } from './transport';
import type {
  TraceAgent,
  TraceEvent,
  TraceSection,
  TraceStatus,
  TraceSummary,
} from './types';

export interface SessionOptions {
  /** Reuse an id to append to an existing session; generated otherwise. */
  sessionId?: string;
  /** Groups sessions into one conversation in the Threads view. */
  threadId?: string;
  agent?: TraceAgent;
  config?: Record<string, unknown>;
  /** Merged on top of `CognipeerConfig.metadata` for this session; see
   *  `TraceSessionPayload.metadata`. */
  metadata?: Record<string, string>;
  traceId?: string;
  rootSpanId?: string;
  /** Overrides the client-level delivery mode for this session. */
  mode?: 'auto' | 'stream' | 'batch';
  startedAt?: Date;
}

/** Everything an integration knows when a framework's "start" callback fires. */
export interface SpanInit {
  type?: TraceEvent['type'];
  label?: string;
  /** Key of the parent span; resolves to `rootSpanId` when unknown. */
  parentKey?: string;
  model?: string;
  actor?: TraceEvent['actor'];
  toolName?: string;
  /** Correlates the span with the model's tool-call id that requested it. */
  toolExecutionId?: string;
  sections?: TraceSection[];
  metadata?: Record<string, unknown>;
  toolDefinitions?: TraceEvent['toolDefinitions'];
  responseFormat?: TraceEvent['responseFormat'];
  startedAt?: Date;
}

/** Everything the matching "end"/"error" callback adds. */
export interface SpanClose {
  status?: TraceStatus;
  label?: string;
  model?: string;
  sections?: TraceSection[];
  metadata?: Record<string, unknown>;
  toolDefinitions?: TraceEvent['toolDefinitions'];
  responseFormat?: TraceEvent['responseFormat'];
  reasoningTokens?: number;
  finishReason?: string;
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  totalTokens?: number;
  toolName?: string;
  toolExecutionId?: string;
  error?: unknown;
  endedAt?: Date;
}

interface OpenSpan {
  init: SpanInit;
  startedAt: number;
}

export class TraceSession {
  readonly sessionId: string;
  readonly traceId: string;
  readonly rootSpanId: string;
  threadId?: string;

  private readonly startedAt: Date;
  private readonly agent: TraceAgent;
  private readonly metadata: Record<string, string>;
  private readonly sessionConfig?: Record<string, unknown>;
  private readonly mode: 'auto' | 'stream' | 'batch';

  private sequence = 0;
  private readonly open = new Map<string, OpenSpan>();
  /** Every event, kept for the batch payload. */
  private readonly all: TraceEvent[] = [];
  /** Events not yet delivered over the streaming endpoint. */
  private queue: TraceEvent[] = [];
  private readonly errors: Array<Record<string, unknown>> = [];
  private readonly summary: Required<Pick<
    TraceSummary,
    'totalInputTokens' | 'totalOutputTokens' | 'totalCachedInputTokens' | 'totalDurationMs'
  >> & { eventCounts: Record<string, number> } = {
    totalInputTokens: 0,
    totalOutputTokens: 0,
    totalCachedInputTokens: 0,
    totalDurationMs: 0,
    eventCounts: {},
  };

  private streaming = false;
  private finished = false;
  /** Serializes network calls: `/start` must land before any `/events`. */
  private chain: Promise<void> = Promise.resolve();

  constructor(
    private readonly config: ResolvedConfig,
    private readonly transport: Transport,
    private readonly onClose: (session: TraceSession) => void,
    options: SessionOptions = {},
  ) {
    this.sessionId = options.sessionId ?? newSessionId();
    this.traceId = options.traceId ?? newTraceId();
    this.rootSpanId = options.rootSpanId ?? newSpanId();
    this.threadId = options.threadId ?? config.threadId;
    this.agent = { ...config.agent, ...options.agent };
    this.metadata = { ...config.metadata, ...options.metadata };
    // Sanitized once at construction: it is written verbatim by both the
    // batch payload and `/start`, and integrations put framework metadata
    // (Agents SDK `trace.metadata`, for one) straight into it.
    this.sessionConfig = sanitizeMetadata(options.config, config);
    this.mode = options.mode ?? config.mode;
    this.startedAt = options.startedAt ?? new Date();

    // `stream` mode announces the session immediately so it appears as
    // in-progress in the Console while the agent is still thinking.
    if (this.config.enabled && this.mode === 'stream') {
      this.beginStream();
    }
  }

  /** True when the transport is off — integrations can skip work entirely. */
  get disabled(): boolean {
    return !this.config.enabled;
  }

  // ── Recording ────────────────────────────────────────────────────────────

  /**
   * Record a fully-formed event.
   *
   * Runs inside the traced application's own call stack, so it must not throw:
   * a payload can be actively hostile (a throwing getter, a revoked Proxy) and
   * an escaping exception would take the application down over a trace event.
   * Sanitisation failures degrade the event instead.
   */
  record(event: TraceEvent): TraceEvent | undefined {
    if (!this.config.enabled || this.finished) return undefined;

    const sequence = ++this.sequence;
    const normalized: TraceEvent = {
      ...event,
      id: event.id ?? `${this.sessionId}-${sequence}`,
      sequence,
      timestamp: event.timestamp ?? new Date().toISOString(),
      traceId: event.traceId ?? this.traceId,
      spanId: event.spanId ?? newSpanId(),
      parentSpanId: event.parentSpanId ?? this.rootSpanId,
      status: event.status ?? 'success',
      sections: this.guard(() => sanitizeSections(event.sections, this.config), undefined),
      // Metadata is the other channel by which caller data reaches the wire,
      // and it used to bypass redaction and the size cap entirely.
      metadata: this.guard(() => sanitizeMetadata(event.metadata, this.config), undefined),
    };

    this.accumulate(normalized);
    this.all.push(normalized);
    this.queue.push(normalized);
    this.maybeStream();
    return normalized;
  }

  /**
   * Open a span keyed by the framework's run id. Nothing is sent yet — the
   * event is emitted on `closeSpan`, so one framework start/end pair becomes
   * exactly one Console event carrying both sides.
   */
  openSpan(key: string, init: SpanInit = {}): void {
    if (!this.config.enabled || this.finished) return;
    this.open.set(key, {
      init,
      startedAt: (init.startedAt ?? new Date()).getTime(),
    });
  }

  /** True when `key` has an open span — lets integrations stay idempotent. */
  hasSpan(key: string): boolean {
    return this.open.has(key);
  }

  /** Merge extra data into an already-open span. */
  updateSpan(key: string, patch: Partial<SpanInit>): void {
    const span = this.open.get(key);
    if (!span) return;
    span.init = {
      ...span.init,
      ...patch,
      sections: [...(span.init.sections ?? []), ...(patch.sections ?? [])],
      metadata: { ...span.init.metadata, ...patch.metadata },
    };
  }

  /** Close a span and emit the resulting event. */
  closeSpan(key: string, close: SpanClose = {}): TraceEvent | undefined {
    if (!this.config.enabled || this.finished) return undefined;
    const span = this.open.get(key);
    this.open.delete(key);

    const startedAt = span?.startedAt ?? Date.now();
    const endedAt = (close.endedAt ?? new Date()).getTime();
    const init = span?.init ?? {};
    const error = close.error ? normalizeError(close.error) : undefined;

    return this.record({
      id: key,
      type: init.type ?? 'span',
      label: close.label ?? init.label,
      spanId: spanIdFrom(key),
      parentSpanId: init.parentKey ? spanIdFrom(init.parentKey) : this.rootSpanId,
      timestamp: new Date(startedAt).toISOString(),
      durationMs: Math.max(0, endedAt - startedAt),
      status: error ? 'error' : (close.status ?? 'success'),
      model: close.model ?? init.model,
      actor: init.actor,
      toolName: close.toolName ?? init.toolName,
      toolExecutionId: close.toolExecutionId ?? init.toolExecutionId,
      toolDefinitions: close.toolDefinitions ?? init.toolDefinitions,
      responseFormat: close.responseFormat ?? init.responseFormat,
      sections: [...(init.sections ?? []), ...(close.sections ?? [])],
      metadata: { ...init.metadata, ...close.metadata },
      inputTokens: close.inputTokens,
      outputTokens: close.outputTokens,
      cachedInputTokens: close.cachedInputTokens,
      reasoningTokens: close.reasoningTokens,
      finishReason: close.finishReason,
      totalTokens: close.totalTokens,
      error,
    });
  }

  /** Abandon every still-open span — used when a run ends abnormally. */
  private closeDanglingSpans(status: TraceStatus): void {
    for (const key of [...this.open.keys()]) {
      this.closeSpan(key, { status });
    }
  }

  // ── Delivery ─────────────────────────────────────────────────────────────

  /**
   * Close the session and deliver everything. Safe to call twice; the second
   * call resolves once the first delivery settles.
   */
  async end(options: { status?: TraceStatus; error?: unknown } = {}): Promise<void> {
    if (!this.config.enabled) return;
    if (this.finished) {
      await this.settle();
      return;
    }

    this.closeDanglingSpans(options.status === 'error' ? 'error' : 'success');
    if (options.error) {
      this.errors.push(normalizeError(options.error));
    }
    this.finished = true;

    const endedAt = new Date();
    const status: TraceStatus =
      options.status ?? (this.errors.length > 0 ? 'error' : 'success');
    const durationMs = endedAt.getTime() - this.startedAt.getTime();

    if (this.streaming) {
      this.flushQueued();
      this.enqueue(() =>
        this.transport.endStream(this.sessionId, {
          status,
          endedAt: endedAt.toISOString(),
          durationMs,
          summary: { ...this.summary },
          errors: this.errors,
        }),
      );
    } else {
      const events = this.all;
      this.queue = [];
      this.enqueue(() =>
        this.transport.ingestSession({
          sessionId: this.sessionId,
          threadId: this.threadId,
          traceId: this.traceId,
          rootSpanId: this.rootSpanId,
          agent: this.agent,
          config: this.sessionConfig,
          metadata: this.metadata,
          status,
          startedAt: this.startedAt.toISOString(),
          endedAt: endedAt.toISOString(),
          durationMs,
          summary: { ...this.summary },
          errors: this.errors,
          events,
        }),
      );
    }

    this.onClose(this);
    await this.settle();
  }

  /**
   * Attach a conversation id discovered after the session opened.
   *
   * Needed by the OpenTelemetry exporter: a trace's root span ends LAST, so
   * the span carrying `session.id` routinely arrives after child spans have
   * already opened the session. Without this, every OTel-sourced run would
   * land unthreaded.
   *
   * A no-op once a thread id is set or the session has been delivered. When
   * the session is already streaming, `/start` is re-issued — the server's
   * reopen path merges the new thread id and keeps every running total.
   */
  setThreadId(threadId: string | undefined): void {
    if (!threadId || this.finished || this.threadId) return;
    this.threadId = threadId;
    if (this.streaming) {
      this.enqueue(() =>
        this.transport.startStream(this.sessionId, {
          threadId,
          traceId: this.traceId,
          rootSpanId: this.rootSpanId,
          agent: this.agent,
          metadata: this.metadata,
          startedAt: this.startedAt.toISOString(),
        }),
      );
    }
  }

  /**
   * Merge agent identity discovered after the session opened. Same reason as
   * `setThreadId` — an OTel root span names the agent, and it ends last.
   */
  setAgent(agent: TraceAgent | undefined): void {
    if (!agent || this.finished) return;
    for (const [key, value] of Object.entries(agent)) {
      if (value !== undefined && value !== null && value !== '') this.agent[key] = value;
    }
  }

  /** Await every in-flight request without closing the session. */
  async flush(): Promise<void> {
    if (this.streaming) this.flushQueued();
    await this.settle();
  }

  /**
   * Await the delivery chain until it stops growing.
   *
   * `await this.chain` alone captures the chain as it stands at that instant;
   * anything enqueued while it is being awaited — a late `setThreadId`, a
   * concurrent `flush()` — would be left in flight and lost by a caller that
   * exits immediately afterwards.
   *
   * Public because the client awaits it for sessions that have already ended:
   * `end()` returns before the network settles, so this is the only handle on
   * a delivery that is still on the wire.
   */
  async settle(): Promise<void> {
    let awaited: Promise<void> | undefined;
    while (awaited !== this.chain) {
      awaited = this.chain;
      await awaited;
    }
  }

  /** Snapshot of the running totals — handy in tests and for logging. */
  getSummary(): TraceSummary {
    return { ...this.summary };
  }

  private accumulate(event: TraceEvent): void {
    const type = event.type ?? 'span';
    this.summary.eventCounts[type] = (this.summary.eventCounts[type] ?? 0) + 1;
    this.summary.totalInputTokens += event.inputTokens ?? 0;
    this.summary.totalOutputTokens += event.outputTokens ?? 0;
    this.summary.totalCachedInputTokens += event.cachedInputTokens ?? 0;
    this.summary.totalDurationMs += event.durationMs ?? 0;
    if (event.status === 'error') {
      this.errors.push({
        eventId: event.id,
        type,
        message:
          typeof event.error === 'string'
            ? event.error
            : ((event.error?.message as string | undefined) ?? event.label ?? 'Event error'),
        timestamp: event.timestamp,
      });
    }
  }

  /** Decide whether this session should go live, and do it if so. */
  private maybeStream(): void {
    if (this.finished || this.mode === 'batch') return;
    // Already live: hand the event straight to the transport. Without this
    // the session would open a stream and then sit on every subsequent event
    // until `end()`, which is delivery-correct but defeats the point of
    // streaming — the run would not appear in the Console until it finished.
    if (this.streaming) {
      this.flushQueued();
      return;
    }
    if (this.mode === 'stream') {
      this.beginStream();
      return;
    }
    const elapsed = Date.now() - this.startedAt.getTime();
    if (elapsed >= this.config.streamAfterMs || this.queue.length >= this.config.streamAfterEvents) {
      this.beginStream();
    }
  }

  /** Open the streaming session and backfill everything buffered so far. */
  private beginStream(): void {
    if (this.streaming) return;
    this.streaming = true;
    this.enqueue(() =>
      this.transport.startStream(this.sessionId, {
        threadId: this.threadId,
        traceId: this.traceId,
        rootSpanId: this.rootSpanId,
        agent: this.agent,
        config: this.sessionConfig,
        metadata: this.metadata,
        startedAt: this.startedAt.toISOString(),
      }),
    );
    this.flushQueued();
  }

  private flushQueued(): void {
    const pending = this.queue;
    this.queue = [];
    for (const event of pending) {
      this.enqueue(() => this.transport.appendEvent(this.sessionId, event));
    }
  }

  /**
   * Append work to the serialized chain. Ordering matters (`/start` before
   * `/events` before `/end`) and the Console processes each request's writes
   * in a background task, so requests are issued one at a time.
   */
  private enqueue(task: () => Promise<void>): void {
    this.chain = this.chain.then(task).catch((error: unknown) => {
      // The user's `onError` is arbitrary code. Letting it throw here would
      // poison the chain permanently: nobody awaits it while a session is
      // open, so Node's default unhandled-rejection policy would kill the
      // process — and once the session IS awaited, the handler's exception
      // would replace the application's own error.
      reportSafely(this.config, error);
    });
  }

  /**
   * Run a sanitisation step that must never escape into the caller.
   * Reports through `onError` and falls back to `fallback`.
   */
  private guard<T>(fn: () => T, fallback: T): T {
    try {
      return fn();
    } catch (error) {
      reportSafely(this.config, error);
      return fallback;
    }
  }
}


/** Turn anything a framework calls an "error" into a stable record. */
export function normalizeError(error: unknown): Record<string, unknown> {
  if (error instanceof Error) {
    return { message: error.message, type: error.name, stack: error.stack };
  }
  if (typeof error === 'string') return { message: error };
  if (error && typeof error === 'object') {
    const record = error as Record<string, unknown>;
    return {
      message:
        typeof record.message === 'string' ? record.message : stringifyContent(error).slice(0, 2_000),
      ...(typeof record.type === 'string' ? { type: record.type } : {}),
    };
  }
  return { message: String(error) };
}
