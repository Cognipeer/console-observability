/**
 * LangGraph ergonomics on top of the LangChain callback handler.
 *
 * LangGraph executes through LangChain's callback manager, so
 * {@link CognipeerCallbackHandler} already captures nodes, model calls, tool
 * calls and retrievers. What LangGraph adds is a threading problem this module
 * solves:
 *
 * - **A resume is a new run.** `graph.invoke(new Command({ resume }), config)`
 *   starts a fresh root run with a fresh trace id. Only `configurable.thread_id`
 *   ties the invocations together, so a conversation is one *thread* of many
 *   sessions in the Console — never one session.
 * - **`thread_id` no longer reaches handlers.** Recent `@langchain/core`
 *   releases stopped copying `configurable` into `metadata` for anything that
 *   is not a LangSmith tracer, so a handler reading `metadata.thread_id` gets
 *   nothing. {@link langgraphConfig} and {@link withCognipeerTracing} read it
 *   out of `configurable` and set both.
 * - **A handler instance is per-run state.** Sharing one across concurrent
 *   invocations of the same compiled graph mixes their spans together, so a
 *   fresh handler is created per call.
 *
 * ```ts
 * import { withCognipeerTracing } from '@cognipeer/observability/langgraph';
 *
 * const graph = withCognipeerTracing(workflow.compile({ checkpointer }));
 * await graph.invoke(state, { configurable: { thread_id: 'conv-42' } });
 * ```
 *
 * Interrupts are handled for you: LangGraph raises them through the same path
 * as real failures, and the handler recognises them as control flow rather than
 * marking every human-in-the-loop pause as an error.
 *
 * One thing no in-process hook can fix: resuming after an interrupt re-runs the
 * interrupted node from its start, so model and tool calls made before the
 * `interrupt()` execute — and are traced — a second time. Those are real calls
 * with real cost, so they are recorded rather than hidden; `langgraph_step` and
 * `checkpoint_ns` on each event identify the replay.
 */

import type { RunnableConfig } from '@langchain/core/runnables';

import {
  CognipeerCallbackHandler,
  cognipeerConfig,
  type CognipeerCallbackHandlerOptions,
} from './langchain';

export {
  CognipeerCallbackHandler,
  cognipeerConfig,
  installLangChainTracing,
  NOISE_CHAIN_NAMES,
  type CognipeerCallbackHandlerOptions,
} from './langchain';

/** Graph entry points that accept `(input, config)` and should be traced. */
const TRACED_METHODS: ReadonlySet<string> = new Set([
  'invoke',
  'stream',
  'streamEvents',
  'streamLog',
  'batch',
]);

export interface CognipeerLangGraphOptions extends CognipeerCallbackHandlerOptions {
  /**
   * Conversation id. Defaults to `configurable.thread_id` from the config of
   * each call, which is the value the checkpointer already uses.
   */
  threadId?: string;
}

/**
 * Build a `RunnableConfig` for one graph invocation.
 *
 * Writes `thread_id` to both `configurable` (for the checkpointer) and
 * `metadata` (so the handler can see it), and attaches a fresh handler.
 *
 * ```ts
 * await graph.invoke(state, langgraphConfig('conv-42'));
 * ```
 */
export function langgraphConfig(
  threadId?: string,
  options: CognipeerLangGraphOptions & { base?: RunnableConfig } = {},
): RunnableConfig {
  const { base, ...handlerOptions } = options;
  return cognipeerConfig(threadId ?? handlerOptions.threadId, { base, ...handlerOptions });
}

/**
 * Wrap a compiled graph so every `invoke` / `stream` / `streamEvents` /
 * `streamLog` / `batch` call is traced, with a fresh handler per call and the
 * thread id lifted out of `configurable`.
 *
 * The returned value is the same graph — every other property and method is
 * forwarded untouched, and its type is preserved — so this is safe to apply at
 * the point of compilation:
 *
 * ```ts
 * export const graph = withCognipeerTracing(workflow.compile({ checkpointer }));
 * ```
 */
export function withCognipeerTracing<T extends object>(
  graph: T,
  options: CognipeerLangGraphOptions = {},
): T {
  return new Proxy(graph, {
    get(target, property, receiver) {
      // Read off the target rather than the proxy: forwarding the receiver
      // makes `this` the proxy inside getters, which breaks any class using
      // private fields.
      const value = Reflect.get(target, property);
      if (typeof value !== 'function') return value;
      if (typeof property !== 'string' || !TRACED_METHODS.has(property)) {
        return value.bind(target);
      }

      return function tracedCall(input: unknown, config?: unknown, ...rest: unknown[]) {
        return (value as (...args: unknown[]) => unknown).call(
          target,
          input,
          withTracingConfig(config, options),
          ...rest,
        );
      };
    },
  });
}

/**
 * Attach tracing to one config value. `batch` takes an array of configs — one
 * per input — and each needs its own handler, or their spans interleave into a
 * single session.
 */
function withTracingConfig(config: unknown, options: CognipeerLangGraphOptions): unknown {
  if (Array.isArray(config)) {
    return config.map((entry) => withTracingConfig(entry, options));
  }

  const base = (config ?? {}) as RunnableConfig;
  const threadId = options.threadId ?? threadIdOf(base);
  const { threadId: _ignored, ...handlerOptions } = options;

  return cognipeerConfig(threadId, {
    base,
    ...handlerOptions,
    handler: new CognipeerCallbackHandler({ ...handlerOptions, threadId }),
  });
}

function threadIdOf(config: RunnableConfig): string | undefined {
  const candidates = [
    (config.configurable as Record<string, unknown> | undefined)?.thread_id,
    (config.metadata as Record<string, unknown> | undefined)?.thread_id,
  ];
  for (const candidate of candidates) {
    if (candidate !== undefined && candidate !== null && candidate !== '') {
      return String(candidate);
    }
  }
  return undefined;
}
