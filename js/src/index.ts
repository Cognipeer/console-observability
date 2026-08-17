/**
 * Cognipeer Observability — ship agent traces into Cognipeer Console.
 *
 * ```ts
 * import { init } from '@cognipeer/observability';
 * import { CognipeerCallbackHandler } from '@cognipeer/observability/langchain';
 *
 * init({ apiKey: 'cpeer_…', agent: { name: 'support-bot' } });
 * await chain.invoke(input, { callbacks: [new CognipeerCallbackHandler()] });
 * ```
 *
 * Framework integrations live behind subpath exports so installing this
 * package pulls in nothing you do not use:
 *
 * | Subpath | Framework |
 * |---|---|
 * | `@cognipeer/observability/langchain` | LangChain (JS) |
 * | `@cognipeer/observability/langgraph` | LangGraph (JS) |
 * | `@cognipeer/observability/openai-agents` | OpenAI Agents SDK |
 * | `@cognipeer/observability/claude-agent-sdk` | Claude Agent SDK |
 * | `@cognipeer/observability/vercel-ai` | Vercel AI SDK |
 * | `@cognipeer/observability/otel` | any OpenTelemetry-instrumented agent |
 */

export { CognipeerObservability, getClient, init, resetClient } from './core/client';
export { ENV, normalizeBaseUrl, resolveConfig, type ResolvedConfig } from './core/config';
export {
  contextReady,
  getCurrentSession,
  getCurrentSpanKey,
  withContext,
  type TraceContext,
} from './core/context';
export { newSessionId, newSpanId, newTraceId, spanIdFrom, traceIdFrom } from './core/ids';
export { observe, trace, type ObserveOptions, type TraceOptions } from './core/observe';
export { redactString, REDACTED, stringifyContent } from './core/redact';
export {
  normalizeError,
  TraceSession,
  type SessionOptions,
  type SpanClose,
  type SpanInit,
} from './core/session';
export { Transport, TransportError } from './core/transport';
export type {
  CaptureMode,
  CognipeerConfig,
  Logger,
  StreamEndPayload,
  StreamStartPayload,
  TraceActor,
  TraceAgent,
  TraceEvent,
  TraceEventType,
  TraceSection,
  TraceSessionPayload,
  TraceStatus,
  TraceSummary,
  TraceToolDefinition,
  TraceUsage,
} from './core/types';

import { getClient } from './core/client';

/**
 * Block until every queued trace has been delivered. Call before a
 * short-lived process exits (a script, a Lambda handler, a CI job);
 * long-running services do not need it.
 */
export async function flush(): Promise<void> {
  await getClient().flush();
}

/** End every open session and flush. Safe to call more than once. */
export async function shutdown(): Promise<void> {
  await getClient().shutdown();
}

export const VERSION = '0.2.0';
