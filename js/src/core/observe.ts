/**
 * Manual instrumentation: `trace()` and `observe()`.
 *
 * The escape hatch for code no framework integration covers — a bespoke
 * orchestration loop, a retrieval helper, a business rule worth seeing in the
 * timeline next to the model calls.
 */

import { getClient } from './client';
import { contextReady, getCurrentSession, getCurrentSpanKey, withContext } from './context';
import { newSpanId } from './ids';
import type { SessionOptions, TraceSession } from './session';
import type { TraceEvent, TraceSection } from './types';

export interface TraceOptions extends SessionOptions {
  /** Shorthand for `agent: { name }` — how the Console groups runs. */
  name?: string;
}

/**
 * Open a session, bind it as the ambient one, and close it when `fn` settles.
 * The session is marked `error` and the exception re-thrown untouched on
 * failure — tracing never swallows application errors.
 */
export async function trace<T>(
  options: TraceOptions,
  fn: (session: TraceSession) => Promise<T> | T,
): Promise<T> {
  // On runtimes where AsyncLocalStorage cannot be resolved synchronously
  // (Node 18/20), waiting for it here is the difference between one session
  // and one session per `await` inside `fn`. Already resolved in practice, so
  // this costs a microtask.
  await contextReady();
  const { name, ...sessionOptions } = options;
  const session = getClient().startSession({
    ...sessionOptions,
    agent: { ...(name ? { name } : {}), ...sessionOptions.agent },
  });

  try {
    const result = await withContext({ session, spanKey: undefined }, () => fn(session));
    await session.end({ status: 'success' });
    return result;
  } catch (error) {
    await session.end({ status: 'error', error });
    throw error;
  }
}

export interface ObserveOptions {
  /** Event label. Defaults to the wrapped function's `name`. */
  name?: string;
  /** Event type — `tool_call` renders as a tool invocation. */
  type?: TraceEvent['type'];
  toolName?: string;
  metadata?: Record<string, unknown>;
  /** Record the arguments as an Input section (default true). */
  captureInput?: boolean;
  /** Record the return value as an Output section (default true). */
  captureOutput?: boolean;
}

/**
 * Wrap a function so every call becomes one event.
 *
 * Nesting is automatic: a wrapped function called from inside another one
 * becomes its child span. When no session is active a root session is created
 * for the outermost call and closed when it settles, so wrapping a single
 * entry point is enough to get a complete trace.
 *
 * ```ts
 * const searchFlights = observe(
 *   async (city: string) => api.search(city),
 *   { type: 'tool_call', toolName: 'search_flights' },
 * );
 * ```
 */
export function observe<A extends unknown[], R>(
  fn: (...args: A) => Promise<R> | R,
  options: ObserveOptions = {},
): (...args: A) => Promise<R> {
  const label = options.name ?? fn.name ?? 'observed';

  return async function observed(...args: A): Promise<R> {
    const client = getClient();
    if (!client.enabled) return await fn(...args);
    await contextReady();

    const ambient = getCurrentSession();
    const session = ambient ?? client.startSession({ agent: { name: label } });
    const ownsSession = !ambient;
    const key = newSpanId();

    const sections: TraceSection[] = [];
    if (options.captureInput !== false && args.length > 0) {
      sections.push({ kind: 'message', label: 'Input', content: args });
    }

    session.openSpan(key, {
      type: options.type ?? 'span',
      label,
      parentKey: getCurrentSpanKey(),
      toolName: options.toolName,
      sections,
      metadata: options.metadata,
    });

    try {
      const result = await withContext({ session, spanKey: key }, () => fn(...args));
      session.closeSpan(key, {
        status: 'success',
        sections:
          options.captureOutput !== false && result !== undefined
            ? [{ kind: 'message', label: 'Output', content: result }]
            : undefined,
      });
      if (ownsSession) await session.end({ status: 'success' });
      return result;
    } catch (error) {
      session.closeSpan(key, { status: 'error', error });
      if (ownsSession) await session.end({ status: 'error', error });
      throw error;
    }
  };
}
