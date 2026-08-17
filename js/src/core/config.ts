import type { CaptureMode, CognipeerConfig, Logger, TraceAgent } from './types';

/** Environment variables read when the matching option is omitted. */
export const ENV = {
  apiKey: 'COGNIPEER_API_KEY',
  baseUrl: 'COGNIPEER_BASE_URL',
  agentName: 'COGNIPEER_AGENT_NAME',
  agentVersion: 'COGNIPEER_AGENT_VERSION',
  enabled: 'COGNIPEER_TRACING_ENABLED',
  capture: 'COGNIPEER_CAPTURE_CONTENT',
  debug: 'COGNIPEER_DEBUG',
  mode: 'COGNIPEER_TRACING_MODE',
} as const;

const DEFAULT_BASE_URL = 'https://console.cognipeer.com';

function env(name: string): string | undefined {
  const value = (globalThis as { process?: { env?: Record<string, string | undefined> } })
    .process?.env?.[name];
  return value && value.length > 0 ? value : undefined;
}

function envBool(name: string, fallback: boolean): boolean {
  const raw = env(name);
  if (raw === undefined) return fallback;
  return !['0', 'false', 'no', 'off'].includes(raw.toLowerCase());
}

/**
 * Normalize a user-supplied base URL to the host root.
 * Accepts the host (`https://console.cognipeer.com`) and the legacy
 * fully-qualified ingest path (`.../api/client/v1`), which other Cognipeer
 * SDKs asked for.
 */
export function normalizeBaseUrl(value: string): string {
  return value
    .trim()
    .replace(/\/+$/, '')
    .replace(/\/api\/client\/v1$/, '');
}

function resolveCapture(value: unknown): CaptureMode {
  const raw = typeof value === 'string' ? value.toLowerCase() : undefined;
  if (raw === 'none' || raw === 'off' || raw === 'false') return 'none';
  if (raw === 'metadata' || raw === 'meta' || raw === 'structure') return 'metadata';
  return 'all';
}

const noopLogger: Logger = {
  debug: () => {},
  warn: () => {},
  error: () => {},
};

function defaultLogger(debug: boolean): Logger {
  if (!debug) {
    return {
      debug: () => {},
      warn: (...args) => console.warn('[cognipeer]', ...args),
      error: (...args) => console.error('[cognipeer]', ...args),
    };
  }
  return {
    debug: (...args) => console.log('[cognipeer]', ...args),
    warn: (...args) => console.warn('[cognipeer]', ...args),
    error: (...args) => console.error('[cognipeer]', ...args),
  };
}

export interface ResolvedConfig {
  apiKey: string;
  baseUrl: string;
  agent: TraceAgent;
  metadata: Record<string, string>;
  threadId?: string;
  enabled: boolean;
  capture: CaptureMode;
  redactPatterns: RegExp[];
  maxContentChars: number;
  mode: 'auto' | 'stream' | 'batch';
  streamAfterMs: number;
  streamAfterEvents: number;
  timeout: number;
  maxRetries: number;
  headers: Record<string, string>;
  fetch: typeof fetch;
  debug: boolean;
  logger: Logger;
  onError: (error: Error) => void;
}

/**
 * Merge explicit options with environment defaults.
 *
 * A missing API key is NOT an error: it disables the exporter and warns once.
 * Telemetry that throws at import time in production is worse than no
 * telemetry, and it is a common failure mode for observability SDKs.
 */
export function resolveConfig(options: CognipeerConfig = {}): ResolvedConfig {
  const debug = options.debug ?? envBool(ENV.debug, false);
  const logger = options.logger ?? (options.enabled === false ? noopLogger : defaultLogger(debug));
  const apiKey = options.apiKey ?? env(ENV.apiKey) ?? '';
  const capture = options.capture ?? resolveCapture(env(ENV.capture));
  const configuredEnabled = options.enabled ?? envBool(ENV.enabled, true);
  const enabled = configuredEnabled && capture !== 'none' && apiKey.length > 0;

  if (configuredEnabled && capture !== 'none' && apiKey.length === 0) {
    logger.warn(
      `no API key — tracing disabled. Set ${ENV.apiKey} or pass { apiKey } to init().`,
    );
  }

  const agentName = options.agent?.name ?? env(ENV.agentName);
  const agentVersion = options.agent?.version ?? env(ENV.agentVersion);
  const envMode = env(ENV.mode);
  const mode = options.mode
    ?? (envMode === 'stream' || envMode === 'batch' || envMode === 'auto' ? envMode : 'auto');

  return {
    apiKey,
    baseUrl: normalizeBaseUrl(options.baseUrl ?? env(ENV.baseUrl) ?? DEFAULT_BASE_URL),
    agent: {
      ...options.agent,
      ...(agentName ? { name: agentName } : {}),
      ...(agentVersion ? { version: agentVersion } : {}),
    },
    metadata: { ...options.metadata },
    threadId: options.threadId,
    enabled,
    capture,
    redactPatterns: options.redactPatterns ?? [],
    maxContentChars: options.maxContentChars ?? 50_000,
    mode,
    streamAfterMs: options.streamAfterMs ?? 2_000,
    streamAfterEvents: options.streamAfterEvents ?? 25,
    timeout: options.timeout ?? 30_000,
    maxRetries: options.maxRetries ?? 3,
    headers: options.headers ?? {},
    fetch: options.fetch ?? globalThis.fetch?.bind(globalThis),
    debug,
    logger,
    onError:
      options.onError ??
      ((error: Error) => logger.warn('trace export failed:', error.message)),
  };
}

/**
 * Invoke the configured error handler without letting it escape.
 *
 * Every failure report in this package goes through here. A user's `onError`
 * is arbitrary code — often a Sentry call that can itself throw before it is
 * initialised — and the one thing worse than a lost trace is a tracing
 * callback taking down the application it was watching. In the delivery chain
 * a throwing handler is worse still: it poisons the chain, which nobody awaits
 * while a session is open, so Node's default unhandled-rejection policy kills
 * the process.
 */
export function reportSafely(config: ResolvedConfig, error: unknown): void {
  try {
    config.onError(error instanceof Error ? error : new Error(String(error)));
  } catch {
    // Nothing left to report to — the reporter itself is broken.
  }
}
