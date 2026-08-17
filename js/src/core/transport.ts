/**
 * HTTP transport for the Console tracing ingest API.
 *
 * Contract with the rest of the package: this never throws at the caller.
 * A failed export is reported through `onError` and dropped. Losing a trace
 * is acceptable; breaking the traced application is not.
 */

import { reportSafely, type ResolvedConfig } from './config';
import type {
  StreamEndPayload,
  StreamStartPayload,
  TraceEvent,
  TraceSessionPayload,
} from './types';

/** Statuses worth retrying: transient server/edge failures and rate limits. */
const RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

export class TransportError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly body?: string,
  ) {
    super(message);
    this.name = 'TransportError';
  }
}

export class Transport {
  constructor(private readonly config: ResolvedConfig) {}

  /** Batch ingest — one request carrying the whole session. */
  async ingestSession(payload: TraceSessionPayload): Promise<void> {
    await this.post('/api/client/v1/tracing/sessions', payload);
  }

  /** Open a streaming session. Must complete before events are accepted. */
  async startStream(sessionId: string, payload: StreamStartPayload): Promise<void> {
    await this.post(
      `/api/client/v1/tracing/sessions/stream/${encodeURIComponent(sessionId)}/start`,
      payload,
    );
  }

  /** Append one event to an open streaming session. */
  async appendEvent(sessionId: string, event: TraceEvent): Promise<void> {
    await this.post(
      `/api/client/v1/tracing/sessions/stream/${encodeURIComponent(sessionId)}/events`,
      { event },
    );
  }

  /** Close a streaming session. */
  async endStream(sessionId: string, payload: StreamEndPayload): Promise<void> {
    await this.post(
      `/api/client/v1/tracing/sessions/stream/${encodeURIComponent(sessionId)}/end`,
      payload,
    );
  }

  /** Raw OTLP/HTTP JSON ingest — used by the OTel exporter. */
  async ingestOtlp(payload: unknown): Promise<void> {
    await this.post('/api/client/v1/traces', payload);
  }

  private async post(path: string, body: unknown): Promise<void> {
    const url = `${this.config.baseUrl}${path}`;
    const serialized = JSON.stringify(body);
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      if (attempt > 0) {
        await sleep(backoffMs(attempt));
      }

      // One deadline for the whole exchange — headers AND body. `fetch`
      // resolves as soon as headers arrive, so a timer cleared at that point
      // leaves the error-body read below unbounded: a half-dead upstream that
      // answers 502 and then dribbles its body would pin `end()`, and through
      // it `trace()` and `observe()`, indefinitely.
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.config.timeout);
      try {
        const response = await this.config.fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.config.apiKey}`,
            'User-Agent': USER_AGENT,
            ...this.config.headers,
          },
          body: serialized,
          signal: controller.signal,
        });

        if (response.ok) {
          if (this.config.debug) {
            this.config.logger.debug(`POST ${path} → ${response.status}`);
          }
          return;
        }

        const text = await response.text().catch(() => '');
        const error = new TransportError(
          `POST ${path} failed: ${response.status} ${text.slice(0, 500)}`,
          response.status,
          text,
        );

        if (!RETRYABLE_STATUS.has(response.status)) {
          this.report(error);
          return;
        }
        lastError = error;
      } catch (cause) {
        // Network error / abort — retryable.
        lastError = cause instanceof Error ? cause : new Error(String(cause));
      } finally {
        clearTimeout(timer);
      }
    }

    this.report(lastError ?? new TransportError(`POST ${path} failed after retries`));
  }

  /** Report through the user's handler without letting it escape. */
  private report(error: Error): void {
    reportSafely(this.config, error);
  }
}

export const USER_AGENT = 'cognipeer-observability-js';

/** Exponential backoff with jitter, capped at 8s. */
function backoffMs(attempt: number): number {
  const base = Math.min(8_000, 250 * 2 ** (attempt - 1));
  return base + Math.random() * base * 0.25;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
