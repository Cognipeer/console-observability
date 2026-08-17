/**
 * The client every integration hangs off.
 *
 * Most users never construct one: `init()` installs a process-wide default and
 * the framework integrations pick it up. An explicit client is there for
 * multi-tenant servers that ship traces to more than one Console project.
 */

import { resolveConfig, type ResolvedConfig } from './config';
import { TraceSession, type SessionOptions } from './session';
import { Transport } from './transport';
import type { CognipeerConfig } from './types';

export class CognipeerObservability {
  readonly config: ResolvedConfig;
  private readonly transport: Transport;
  /** Sessions that have not been ended yet — flushed on shutdown. */
  private readonly live = new Set<TraceSession>();
  /**
   * Sessions that have been ended but whose delivery may still be in flight.
   *
   * `end()` returns before the network settles — that is deliberate, so a
   * framework callback is never blocked on an export. But it means a session
   * leaves `live` while its last request is still on the wire, and `flush()`
   * would then have nothing to await. A short-lived process following the
   * documented `await flush()` before exit would silently lose its final
   * session.
   */
  private readonly settling = new Set<TraceSession>();
  private exitHooksInstalled = false;

  constructor(options: CognipeerConfig = {}) {
    this.config = resolveConfig(options);
    this.transport = new Transport(this.config);
    this.installExitHooks();
  }

  /** True when traces are actually being shipped. */
  get enabled(): boolean {
    return this.config.enabled;
  }

  /** Open a session. Remember to `end()` it — that is what delivers it. */
  startSession(options: SessionOptions = {}): TraceSession {
    const session = new TraceSession(
      this.config,
      this.transport,
      (closed) => {
        this.live.delete(closed);
        // Track it until its delivery settles, so `flush()` still covers it.
        this.settling.add(closed);
        void closed
          .settle()
          .catch(() => {})
          .finally(() => this.settling.delete(closed));
      },
      options,
    );
    this.live.add(session);
    return session;
  }

  /**
   * Run `fn` inside a session that closes itself — including on throw, where
   * the session is marked `error` and the exception is re-thrown untouched.
   */
  async trace<T>(
    options: SessionOptions,
    fn: (session: TraceSession) => Promise<T> | T,
  ): Promise<T> {
    const session = this.startSession(options);
    try {
      const result = await fn(session);
      await session.end({ status: 'success' });
      return result;
    } catch (error) {
      await session.end({ status: 'error', error });
      throw error;
    }
  }

  /**
   * Await delivery of every session opened by this client — open ones AND
   * ones that have ended but are still on the wire.
   *
   * This is the call a script, a Lambda handler or a CI job makes before
   * exiting. It has to cover both sets, because a session that ended a
   * millisecond ago is exactly the one most likely to still be in flight.
   */
  async flush(): Promise<void> {
    await Promise.all([
      ...[...this.live].map((session) => session.flush()),
      ...[...this.settling].map((session) => session.settle()),
    ]);
  }

  /**
   * Flush and close every still-open session, and deregister from the
   * process exit hook.
   *
   * Deregistering is what keeps a server that builds a client per tenant, or a
   * test suite that calls `init()` in a loop, from retaining every client it
   * ever made — along with each one's config closures, transport and session
   * set — for the life of the process.
   */
  async shutdown(): Promise<void> {
    liveClients.delete(this);
    this.exitHooksInstalled = false;
    await Promise.all([...this.live].map((session) => session.end()));
    // `end()` hands delivery to the chain; shutting down means waiting for it.
    await this.flush();
  }

  /** Escape hatch for the OTel exporter and for custom senders. */
  getTransport(): Transport {
    return this.transport;
  }

  /**
   * Best-effort delivery when the host process is going away.
   *
   * Registered once per process rather than once per client: a server that
   * builds a client per tenant would otherwise pile up `beforeExit` listeners
   * and trip Node's max-listeners warning. `beforeExit` still allows async
   * work, which is why it is the hook used — but a short-lived process should
   * still call `flush()` explicitly rather than rely on it.
   */
  private installExitHooks(): void {
    if (this.exitHooksInstalled || !this.config.enabled) return;
    this.exitHooksInstalled = true;
    liveClients.add(this);
    installProcessExitHook();
  }
}

/**
 * Clients with exit hooks armed, drained by the single process listener.
 *
 * Membership is the only strong reference this module holds to a client, so
 * `shutdown()` must remove it — see the note there.
 */
const liveClients = new Set<CognipeerObservability>();
let processHookInstalled = false;

function installProcessExitHook(): void {
  if (processHookInstalled) return;
  // Typed structurally rather than as `NodeJS.Process`: the package also
  // targets edge runtimes, where @types/node is not present.
  const proc = (globalThis as { process?: { on?: (event: string, cb: () => void) => void } })
    .process;
  if (!proc?.on) return;
  processHookInstalled = true;
  proc.on('beforeExit', () => {
    for (const client of liveClients) void client.shutdown();
  });
}

let defaultClient: CognipeerObservability | undefined;

/**
 * Configure the process-wide client. Call once at startup, before creating
 * agents. Calling it again replaces the client (the previous one is flushed in
 * the background) — useful in tests.
 */
export function init(options: CognipeerConfig = {}): CognipeerObservability {
  const previous = defaultClient;
  defaultClient = new CognipeerObservability(options);
  // `shutdown()` also deregisters the replaced client, so repeated `init()`
  // calls (a per-tenant server, a test suite) do not accumulate clients.
  if (previous) void previous.shutdown();
  return defaultClient;
}

/**
 * The process-wide client, created from environment variables on first use.
 * With no `COGNIPEER_API_KEY` present this returns a disabled client, so
 * integrations can be wired up unconditionally.
 */
export function getClient(): CognipeerObservability {
  if (!defaultClient) defaultClient = new CognipeerObservability();
  return defaultClient;
}

/** Reset the process-wide client. Intended for tests. */
export function resetClient(): void {
  if (defaultClient) void defaultClient.shutdown();
  defaultClient = undefined;
}

/** How many clients are still registered for exit-time flushing. Tests only. */
export function liveClientCount(): number {
  return liveClients.size;
}
