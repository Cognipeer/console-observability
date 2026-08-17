/**
 * OpenTelemetry span exporter — the "works with anything" route.
 *
 * Most 2026-era agent frameworks emit OTel spans, either natively (Pydantic
 * AI, Google ADK, AWS Strands, Semantic Kernel, Microsoft Agent Framework) or
 * through an OpenInference / OpenLLMetry instrumentor (CrewAI, LlamaIndex,
 * smolagents, DSPy, Haystack, Agno …). Point this exporter at your tracer
 * provider and those traces become Console sessions:
 *
 * ```ts
 * import { NodeSDK } from '@opentelemetry/sdk-node';
 * import { init } from '@cognipeer/observability';
 * import { CognipeerSpanExporter } from '@cognipeer/observability/otel';
 *
 * init({ apiKey: process.env.COGNIPEER_API_KEY });
 * new NodeSDK({ traceExporter: new CognipeerSpanExporter() }).start();
 * ```
 *
 * It maps onto the native session model rather than forwarding raw OTLP,
 * because the mapping is better done here than server-side: this process knows
 * which convention the span speaks, can group a trace into one session, and
 * can reconstruct the tool menu from three different attribute shapes.
 *
 * **Where a session begins and ends.** OpenTelemetry has no session
 * lifecycle — there is no "run started" or "run finished" signal, only spans.
 * A session is therefore opened by the first span of a trace and closed by
 * whichever comes first: the trace's root span ending, or `sessionIdleMs`
 * elapsing with no further spans. Both are needed. A batching span processor
 * delivers children before their parents, so the root really is the last span
 * of a finished trace; but a trace that is dropped, sampled away or still
 * running would otherwise stay open forever.
 *
 * **Threads.** No instrumentation emits a conversation id on its own. In
 * OpenInference the application must wrap its calls in `using_session()` /
 * `setSession()`; in OpenLLMetry it must call `set_association_properties()`
 * or `set_conversation_id()`. Without one of those, every trace arrives as its
 * own orphan session and the Threads view has nothing to group.
 */

import { getClient, type CognipeerObservability } from '../core/client';
import { spanIdFrom } from '../core/ids';
import type { TraceSession } from '../core/session';
import type { TraceEvent, TraceToolDefinition } from '../core/types';
import {
  extractAgentName,
  extractThreadId,
  normalizeSpan,
  type Attributes,
  type OtelSpanLike,
} from './normalize';

export * from './normalize';

/* ------------------------------------------------------------------ */
/*  Minimal OTel surface                                              */
/* ------------------------------------------------------------------ */

/** Result codes from `@opentelemetry/core#ExportResultCode`. */
export const ExportResultCode = {
  SUCCESS: 0 as const,
  FAILED: 1 as const,
};

export interface ExportResult {
  code: number;
  error?: Error;
}

/**
 * Structural subset of `@opentelemetry/sdk-trace-base#ReadableSpan`.
 *
 * Typed structurally on purpose: an observability package must not force an
 * `@opentelemetry/*` version on the application it is watching, and the same
 * shape is produced by every SDK generation.
 */
export interface ReadableSpanLike {
  name: string;
  spanContext(): { traceId: string; spanId: string };
  /** SDK v1. */
  parentSpanId?: string;
  /** SDK v2 renamed it; both are read. */
  parentSpanContext?: { spanId?: string };
  /** `[seconds, nanoseconds]`. */
  startTime: [number, number];
  endTime: [number, number];
  status?: { code?: number; message?: string };
  attributes?: Attributes;
  events?: Array<{ name: string; attributes?: Attributes }>;
  resource?: { attributes?: Attributes };
}

/* ------------------------------------------------------------------ */
/*  Exporter                                                          */
/* ------------------------------------------------------------------ */

export interface CognipeerSpanExporterOptions {
  /** Client to send through. Defaults to the process-wide one. */
  client?: CognipeerObservability;
  /**
   * Close a session this long after its last span when no root span arrived
   * (default 30 000 ms). OTel has no end-of-run signal, so this is the only
   * backstop for a trace that never completes.
   */
  sessionIdleMs?: number;
  /** Agent name for every session, overriding what the spans report. */
  agentName?: string;
  /** Thread id for every session, overriding what the spans report. */
  threadId?: string;
  /** Prefix for the derived session id (default `otel-`). */
  sessionIdPrefix?: string;
  /** Delivery mode override; defaults to the client's configuration. */
  mode?: 'auto' | 'stream' | 'batch';
  /**
   * Ceiling on concurrently open traces (default 1 000). The oldest is closed
   * when it is passed, so a misbehaving producer cannot grow the map without
   * bound.
   */
  maxOpenTraces?: number;
  /** Called instead of throwing when a span cannot be mapped. */
  onError?: (error: Error) => void;
}

interface OpenTrace {
  session: TraceSession;
  /** `spanId`s already recorded — the same span can arrive twice. */
  seen: Set<string>;
  timer?: ReturnType<typeof setTimeout>;
  openedAt: number;
}

/**
 * OTel `SpanExporter` that ships traces to Cognipeer Console.
 *
 * One OTel trace becomes one Console session; each span becomes one event,
 * linked by `parentSpanId` so the Console renders the real call tree.
 */
export class CognipeerSpanExporter {
  private readonly options: CognipeerSpanExporterOptions;
  private readonly traces = new Map<string, OpenTrace>();
  /** Sessions closing in the background, awaited by `shutdown`/`forceFlush`. */
  private readonly closing = new Set<Promise<void>>();
  /**
   * Identical tool menus repeat verbatim on every turn. Interning them by
   * content hash makes those turns share one array instead of holding a copy
   * each; the menu still travels on every event, because it can change between
   * turns and the Console prices it per call.
   */
  private readonly toolMenus = new Map<string, TraceToolDefinition[]>();
  /** Same reasoning for the structured-output contract: one schema, repeated
   *  on every turn of a run that enforces it. */
  private readonly responseFormats = new Map<string, NonNullable<TraceEvent['responseFormat']>>();
  private stopped = false;

  constructor(options: CognipeerSpanExporterOptions = {}) {
    this.options = options;
  }

  /**
   * Called by the OTel SDK with a batch of finished spans.
   *
   * Never throws: a failure is reported through the result callback, because
   * an exporter that throws takes the host's telemetry pipeline down with it.
   */
  export(spans: ReadableSpanLike[], resultCallback: (result: ExportResult) => void): void {
    try {
      this.consume(spans);
      resultCallback({ code: ExportResultCode.SUCCESS });
    } catch (error) {
      const wrapped = error instanceof Error ? error : new Error(String(error));
      this.options.onError?.(wrapped);
      resultCallback({ code: ExportResultCode.FAILED, error: wrapped });
    }
  }

  /** Close every open session and wait for delivery. */
  async shutdown(): Promise<void> {
    this.stopped = true;
    for (const traceId of [...this.traces.keys()]) this.closeTrace(traceId);
    await Promise.all([...this.closing]);
    await (this.options.client ?? getClient()).flush();
  }

  /** Await in-flight deliveries without closing open sessions. */
  async forceFlush(): Promise<void> {
    await Promise.all([...this.closing]);
    await (this.options.client ?? getClient()).flush();
  }

  /* ---------------------------------------------------------------- */
  /*  Internals                                                       */
  /* ---------------------------------------------------------------- */

  private consume(spans: ReadableSpanLike[]): void {
    if (this.stopped || spans.length === 0) return;

    // A batch arrives in completion order, which is roughly reverse causal
    // order. Sorting by start time gives the Console a sensible `sequence`.
    const normalized = spans
      .map((span) => toSpanLike(span))
      .filter((span) => span.traceId && span.spanId)
      .sort((a, b) => a.startTimeMs - b.startTimeMs)
      .map((span) => ({ span, result: normalizeSpan(span) }));

    // Group before recording. A session's identity is fixed when it is
    // created, and the thread id is frequently present on only some of a
    // trace's spans — an instrumentation that reads it from context stamps
    // the LLM span but not the enclosing agent span. Scanning the batch first
    // means the session is opened with the identity the trace actually
    // carries, rather than whatever the earliest span happened to know.
    const byTrace = new Map<string, typeof normalized>();
    for (const entry of normalized) {
      const bucket = byTrace.get(entry.span.traceId);
      if (bucket) bucket.push(entry);
      else byTrace.set(entry.span.traceId, [entry]);
    }

    const finished = new Set<string>();

    for (const [traceId, entries] of byTrace) {
      const threadId = entries.find((entry) => entry.result.threadId)?.result.threadId;
      // A root span names the agent; a leaf usually only knows the service.
      const rootAgentName = entries.find(
        (entry) => entry.result.isRoot && entry.result.agentName,
      )?.result.agentName;
      const agentName =
        rootAgentName ?? entries.find((entry) => entry.result.agentName)?.result.agentName;

      const trace = this.openTrace(entries[0].span, threadId, agentName, rootAgentName);
      if (!trace) continue;

      for (const { span, result } of entries) {
        if (trace.seen.has(span.spanId)) continue;
        trace.seen.add(span.spanId);

        const event = result.event;
        // A root span has no parent of its own; hanging it off the session
        // root keeps the tree single-trunked instead of leaving it dangling.
        if (!event.parentSpanId) event.parentSpanId = trace.session.rootSpanId;
        if (event.toolDefinitions) event.toolDefinitions = this.internMenu(event.toolDefinitions);
        if (event.responseFormat) event.responseFormat = this.internResponseFormat(event.responseFormat);
        if (result.userId) event.metadata = { ...event.metadata, userId: result.userId };

        trace.session.record(event);
        if (result.isRoot) finished.add(traceId);
      }

      if (!finished.has(traceId)) this.armIdleTimer(traceId);
    }

    // Close only after the whole batch is recorded: the root often shares a
    // batch with its children, and closing on sight would drop the rest.
    for (const traceId of finished) this.closeTrace(traceId);
  }

  private openTrace(
    span: OtelSpanLike,
    threadId: string | undefined,
    agentName: string | undefined,
    /** Set only when THIS batch carried the root — the authoritative name. */
    rootAgentName?: string,
  ): OpenTrace | undefined {
    const existing = this.traces.get(span.traceId);
    if (existing) {
      // A trace's ROOT span ends LAST, so with a SimpleSpanProcessor — or any
      // time the root misses its children's batch — the span carrying
      // `session.id` and the agent name arrives after the session is already
      // open. Backfill both while it is still open; otherwise every
      // OTel-sourced run lands unthreaded and named after a leaf.
      existing.session.setThreadId(this.options.threadId ?? threadId);
      // Only a root may rename the agent: a leaf-only batch knows the service,
      // not the agent, and would otherwise overwrite a better name.
      if (!this.options.agentName && rootAgentName) {
        existing.session.setAgent({ name: rootAgentName });
      }
      return existing;
    }
    if (this.stopped) return undefined;

    this.enforceCeiling();

    const client = this.options.client ?? getClient();
    const session = client.startSession({
      sessionId: `${this.options.sessionIdPrefix ?? 'otel-'}${span.traceId}`,
      // The session's identity is fixed at creation, so it takes whatever the
      // first span of the trace revealed. With a batching processor that is a
      // leaf, which is why `service.name` from the resource is the fallback —
      // it is present on every span regardless of convention.
      threadId: this.options.threadId ?? threadId,
      agent: {
        name:
          this.options.agentName ??
          agentName ??
          extractAgentName(span.attributes, span.resource) ??
          'otel-agent',
      },
      traceId: span.traceId,
      // Derived from the trace id rather than random: a trace that resumes
      // after an idle close re-opens under the same session id, and a fresh
      // root span id would re-parent the new events under a trunk the earlier
      // ones never knew about, splitting one run's tree in two.
      rootSpanId: spanIdFrom(`otel-root:${span.traceId}`),
      ...(this.options.mode ? { mode: this.options.mode } : {}),
    });

    const trace: OpenTrace = { session, seen: new Set(), openedAt: Date.now() };
    this.traces.set(span.traceId, trace);
    this.armIdleTimer(span.traceId);
    return trace;
  }

  /**
   * (Re)start the idle countdown for a trace.
   *
   * The timer is `unref`ed where the runtime supports it, so a pending session
   * can never be the reason a process refuses to exit.
   */
  private armIdleTimer(traceId: string): void {
    const trace = this.traces.get(traceId);
    if (!trace) return;
    if (trace.timer) clearTimeout(trace.timer);

    const timer = setTimeout(
      () => this.closeTrace(traceId),
      this.options.sessionIdleMs ?? 30_000,
    );
    (timer as { unref?: () => void }).unref?.();
    trace.timer = timer;
  }

  private closeTrace(traceId: string): void {
    const trace = this.traces.get(traceId);
    if (!trace) return;
    this.traces.delete(traceId);
    if (trace.timer) clearTimeout(trace.timer);

    const closed = trace.session
      .end()
      .catch((error: unknown) => {
        const wrapped = error instanceof Error ? error : new Error(String(error));
        this.options.onError?.(wrapped);
      })
      .finally(() => {
        this.closing.delete(closed);
      });
    this.closing.add(closed);
  }

  /** Drop the oldest trace when the open-trace ceiling is reached. */
  private enforceCeiling(): void {
    const ceiling = this.options.maxOpenTraces ?? 1_000;
    while (this.traces.size >= ceiling) {
      let oldestId: string | undefined;
      let oldestAt = Infinity;
      for (const [traceId, trace] of this.traces) {
        if (trace.openedAt < oldestAt) {
          oldestAt = trace.openedAt;
          oldestId = traceId;
        }
      }
      if (!oldestId) return;
      this.closeTrace(oldestId);
    }
  }

  /** Return a shared instance for a menu we have already seen. */
  private internMenu(menu: TraceToolDefinition[]): TraceToolDefinition[] {
    const key = hash(JSON.stringify(menu));
    const cached = this.toolMenus.get(key);
    if (cached) return cached;
    // Bounded: a long-lived process with a churning tool set must not grow
    // this map forever.
    if (this.toolMenus.size > 256) this.toolMenus.clear();
    this.toolMenus.set(key, menu);
    return menu;
  }

  /** Return a shared instance for a contract we have already seen. */
  private internResponseFormat(
    format: NonNullable<TraceEvent['responseFormat']>,
  ): NonNullable<TraceEvent['responseFormat']> {
    const key = hash(JSON.stringify(format));
    const cached = this.responseFormats.get(key);
    if (cached) return cached;
    if (this.responseFormats.size > 256) this.responseFormats.clear();
    this.responseFormats.set(key, format);
    return format;
  }
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

/** `[seconds, nanoseconds]` → epoch milliseconds. */
function hrTimeToMs(time: [number, number] | undefined): number {
  if (!time) return Date.now();
  return time[0] * 1_000 + time[1] / 1_000_000;
}

/** Reduce an SDK span to the shape the normalizer consumes. */
function toSpanLike(span: ReadableSpanLike): OtelSpanLike {
  const context = typeof span.spanContext === 'function' ? span.spanContext() : undefined;
  return {
    traceId: context?.traceId ?? '',
    spanId: context?.spanId ?? '',
    // SDK v2 moved the parent onto a span context; v1 kept a bare id.
    parentSpanId: span.parentSpanContext?.spanId ?? span.parentSpanId ?? undefined,
    name: span.name,
    startTimeMs: hrTimeToMs(span.startTime),
    endTimeMs: hrTimeToMs(span.endTime),
    attributes: span.attributes ?? {},
    status: span.status,
    events: span.events,
    resource: span.resource?.attributes ?? {},
  };
}

/** FNV-1a, for interning identical tool menus. Not a security hash. */
function hash(input: string): string {
  let value = 0x811c9dc5;
  for (let index = 0; index < input.length; index++) {
    value ^= input.charCodeAt(index);
    value =
      (value + ((value << 1) + (value << 4) + (value << 7) + (value << 8) + (value << 24))) >>> 0;
  }
  return `${value.toString(16)}:${input.length}`;
}
