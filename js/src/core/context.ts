/**
 * Ambient session/span context.
 *
 * Framework integrations get their own handles, but manual instrumentation
 * (`observe`) and cross-cutting helpers need to know "which run am I inside
 * right now". On Node that is `AsyncLocalStorage`, which survives `await`
 * boundaries and keeps concurrent runs apart. Runtimes without it (edge
 * workers, browsers) fall back to a single module-level frame — correct for
 * the one-request-per-isolate model those runtimes use, and never worse than
 * having no context at all.
 */

import type { TraceSession } from './session';

export interface TraceContext {
  session?: TraceSession;
  /** Span key of the enclosing `observe` call — becomes the parent key. */
  spanKey?: string;
}

interface Storage {
  getStore(): TraceContext | undefined;
  run<T>(store: TraceContext, fn: () => T): T;
}

/** Single-frame storage for runtimes without async context tracking. */
function createFallbackStorage(): Storage {
  let current: TraceContext | undefined;
  return {
    getStore: () => current,
    run<T>(store: TraceContext, fn: () => T): T {
      const previous = current;
      current = store;
      try {
        return fn();
      } finally {
        current = previous;
      }
    },
  };
}

/**
 * Held in a variable rather than written inline: TypeScript only resolves a
 * literal `import()` specifier, and this package must type-check and build
 * without `@types/node` for consumers targeting edge runtimes.
 */
const ASYNC_HOOKS_MODULE = 'node:async_hooks';

type StorageCtor = new () => Storage;

/**
 * Resolve `AsyncLocalStorage` synchronously when the runtime allows it.
 *
 * This matters more than it looks. An async import only lands on the next
 * microtask, and the normal startup shape — `init()` and then `main()` — does
 * real work in that same tick. Anything traced inside that window would run on
 * the single-frame fallback below, whose frame is restored the moment a
 * function's synchronous part returns: a nested `observe()` called after the
 * first `await` would find no session and open a second one, splitting one run
 * across two sessions. Node ≥ 22.3 exposes `process.getBuiltinModule`, which
 * closes the window entirely.
 */
function resolveStorageSync(): Storage | undefined {
  const getBuiltinModule = (
    globalThis as { process?: { getBuiltinModule?: (id: string) => unknown } }
  ).process?.getBuiltinModule;
  if (typeof getBuiltinModule !== 'function') return undefined;
  try {
    const mod = getBuiltinModule(ASYNC_HOOKS_MODULE) as { AsyncLocalStorage?: StorageCtor };
    return mod?.AsyncLocalStorage ? new mod.AsyncLocalStorage() : undefined;
  } catch {
    return undefined;
  }
}

const synchronous = resolveStorageSync();
let storage: Storage = synchronous ?? createFallbackStorage();

/** Node 18/20 have no `getBuiltinModule` — there, the dynamic import is it. */
const upgraded: Promise<void> = (async () => {
  if (synchronous) return;
  try {
    const mod = (await import(/* @vite-ignore */ ASYNC_HOOKS_MODULE)) as {
      AsyncLocalStorage?: StorageCtor;
    };
    if (mod.AsyncLocalStorage) {
      storage = new mod.AsyncLocalStorage();
    }
  } catch {
    // Not Node — the fallback frame stays in place.
  }
})();

/** Await the async-context upgrade. Only tests should need this. */
export function contextReady(): Promise<void> {
  return upgraded;
}

/** The session enclosing the current execution, if any. */
export function getCurrentSession(): TraceSession | undefined {
  return storage.getStore()?.session;
}

/** The span key enclosing the current execution — used as a parent key. */
export function getCurrentSpanKey(): string | undefined {
  return storage.getStore()?.spanKey;
}

/** Run `fn` with `context` merged over the current one. */
export function withContext<T>(context: TraceContext, fn: () => T): T {
  return storage.run({ ...storage.getStore(), ...context }, fn);
}
