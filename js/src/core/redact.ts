/**
 * Content sanitisation applied to every section before it leaves the process.
 *
 * Three jobs, in order:
 *   1. capture policy — `metadata` drops content entirely, `all` keeps it;
 *   2. secret redaction — built-in patterns for the credential shapes that
 *      routinely end up inside prompts, plus caller-supplied regexes;
 *   3. size capping — a single oversized message must not blow the ingest
 *      body limit (`TRACING_MAX_BODY_SIZE_MB`, 10 MB by default) for the
 *      whole session.
 */

import type { CaptureMode, TraceSection } from './types';

/**
 * Credential shapes worth redacting by default. Deliberately narrow: a
 * false positive silently destroys the prompt the user came to read, so these
 * only match tokens that are unambiguous by construction.
 */
const BUILTIN_PATTERNS: RegExp[] = [
  /\bsk-[A-Za-z0-9_-]{16,}\b/g, // OpenAI-style
  /\bsk-ant-[A-Za-z0-9_-]{16,}\b/g, // Anthropic
  /\bcpeer_[A-Za-z0-9_-]{16,}\b/g, // Cognipeer API token
  /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g, // GitHub
  /\bAKIA[0-9A-Z]{16}\b/g, // AWS access key id
  /\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*/gi,
  /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g, // JWT
];

export const REDACTED = '[redacted]';

/**
 * Inline binary payloads. Multimodal messages embed images, audio and PDFs as
 * base64 data URLs; one of them can be tens of megabytes, which would blow the
 * ingest body limit for the whole session and render as a wall of noise.
 */
const DATA_URL = /data:[\w.+-]+\/[\w.+-]+;base64,[A-Za-z0-9+/=]{100,}/g;

function stripDataUrls(value: string): string {
  DATA_URL.lastIndex = 0;
  return value.replace(DATA_URL, (match) => {
    const mime = match.slice(5, match.indexOf(';'));
    return `data:${mime};base64,[stripped ${match.length} chars]`;
  });
}

export interface SanitizeOptions {
  capture: CaptureMode;
  maxContentChars: number;
  redactPatterns: RegExp[];
}

/** Apply the redaction patterns to a string. */
export function redactString(value: string, extra: RegExp[] = []): string {
  let out = stripDataUrls(value);
  for (const pattern of [...BUILTIN_PATTERNS, ...extra]) {
    // Patterns are shared across calls; reset the lastIndex of /g regexes so
    // a previous call cannot make this one skip the start of the string.
    pattern.lastIndex = 0;
    out = out.replace(pattern, REDACTED);
  }
  return out;
}

/** Stand-in for a value that could not be serialized at all. */
export const UNSERIALIZABLE = '[unserializable]';

/**
 * Serialize any value to the string the Console will render.
 *
 * Never throws. A payload can be actively hostile — a null-prototype object
 * with a throwing getter, a Proxy whose `get` trap raises, a revoked Proxy —
 * and on those `JSON.stringify` throws AND the `String(value)` fallback throws
 * too. Since this runs inside the traced application's own call stack, an
 * escaping exception would take that application down over a log line.
 */
export function stringifyContent(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value, jsonSafeReplacer(), 2) ?? asStringOrPlaceholder(value);
  } catch {
    return asStringOrPlaceholder(value);
  }
}

/** `String(value)` itself throws on some shapes; that is the last resort. */
function asStringOrPlaceholder(value: unknown): string {
  try {
    return String(value);
  } catch {
    return UNSERIALIZABLE;
  }
}

/**
 * Replacer that survives the values real agent frameworks put in payloads:
 * circular graphs, BigInt, Error, Map/Set, and class instances.
 *
 * Cycle detection tracks the ANCESTOR PATH, not every object ever visited. A
 * visited-set would be wrong in the common case: agent payloads routinely
 * reference one object twice — the same system message in two turns, one tool
 * definition in two menus — and marking the second occurrence `[circular]`
 * silently destroys content that is not circular at all.
 *
 * `JSON.stringify` binds the replacer's `this` to the holder of the current
 * key, which is what makes the stack unwindable: anything deeper than the
 * holder has been left behind.
 */
function jsonSafeReplacer(): (this: unknown, key: string, value: unknown) => unknown {
  const ancestors: unknown[] = [];
  return function replace(this: unknown, _key: string, value: unknown): unknown {
    if (typeof value === 'bigint') return value.toString();
    if (value instanceof Error) {
      return { name: value.name, message: value.message, stack: value.stack };
    }
    if (value instanceof Map) return Object.fromEntries(value);
    if (value instanceof Set) return [...value];
    if (typeof value === 'object' && value !== null) {
      while (ancestors.length > 0 && ancestors[ancestors.length - 1] !== this) ancestors.pop();
      if (ancestors.includes(value)) return '[circular]';
      ancestors.push(value);
    }
    return value;
  };
}

/** Truncate to `max` characters, appending a visible marker when cut. */
export function capString(value: string, max: number): { value: string; truncated: boolean } {
  if (value.length <= max) return { value, truncated: false };
  return {
    value: `${value.slice(0, max)}\n…[truncated ${value.length - max} chars]`,
    truncated: true,
  };
}

/**
 * Sanitize one section in place-safe fashion, returning a new section.
 * Returns `undefined` when the capture policy removes the section entirely.
 */
export function sanitizeSection(
  section: TraceSection,
  options: SanitizeOptions,
): TraceSection | undefined {
  if (options.capture === 'none') return undefined;

  // `metadata` capture keeps the shape of the run — which tools ran, in which
  // order, with what schema — but never the message bodies.
  if (options.capture === 'metadata') {
    if (section.kind === 'tool_definitions') return section;
    const { content: _content, ...rest } = section;
    return { ...rest, content: undefined, redacted: true };
  }

  if (section.content === undefined) return section;

  const raw = stringifyContent(section.content);
  const redacted = redactString(raw, options.redactPatterns);
  const { value, truncated } = capString(redacted, options.maxContentChars);

  return {
    ...section,
    content: value,
    ...(truncated ? { truncated: true as const } : {}),
  };
}

/** Sanitize a section list, dropping anything the capture policy removes. */
export function sanitizeSections(
  sections: TraceSection[] | undefined,
  options: SanitizeOptions,
): TraceSection[] | undefined {
  if (!sections?.length) return undefined;
  const out = sections
    .map((section) => sanitizeSection(section, options))
    .filter((section): section is TraceSection => Boolean(section));
  return out.length > 0 ? out : undefined;
}

/** How deep a metadata object is walked before the rest is summarised. */
const METADATA_MAX_DEPTH = 6;
/** Entries kept per object/array level in metadata. */
const METADATA_MAX_ENTRIES = 100;

/**
 * Sanitize an event's `metadata` or a session's `config`.
 *
 * These are the other channel by which caller data reaches the wire — the AI
 * SDK copies `telemetry.metadata.*` verbatim, the Agents SDK puts
 * `trace.metadata` into the session config, LangChain passes user tags
 * through — and until this existed they bypassed redaction, base64 stripping
 * and the size cap under EVERY capture mode, including the default one whose
 * documented promise is "redacted by the configured patterns".
 *
 * Structure is preserved rather than flattened, because the Console renders
 * these as key/value blocks: strings are redacted and capped in place, and
 * depth/breadth are bounded so a payload accidentally routed through metadata
 * cannot blow the ingest body limit for the whole session.
 */
export function sanitizeMetadata<T extends Record<string, unknown> | undefined>(
  metadata: T,
  options: SanitizeOptions,
): T {
  if (!metadata || Object.keys(metadata).length === 0) return metadata;
  if (options.capture === 'none') return undefined as T;
  return sanitizeMetadataValue(metadata, options, 0) as T;
}

function sanitizeMetadataValue(value: unknown, options: SanitizeOptions, depth: number): unknown {
  try {
    return sanitizeMetadataValueUnsafe(value, options, depth);
  } catch {
    // A throwing getter or a hostile Proxy anywhere in the tree degrades that
    // subtree to a placeholder rather than failing the whole event.
    return UNSERIALIZABLE;
  }
}

function sanitizeMetadataValueUnsafe(
  value: unknown,
  options: SanitizeOptions,
  depth: number,
): unknown {
  if (value === null || value === undefined) return value;

  if (typeof value === 'string') {
    // Values are capped individually and more tightly than a message body: a
    // metadata field is a label, never a transcript.
    const { value: capped } = capString(
      redactString(value, options.redactPatterns),
      Math.min(options.maxContentChars, METADATA_VALUE_MAX_CHARS),
    );
    return capped;
  }

  if (typeof value === 'number' || typeof value === 'boolean') return value;
  if (typeof value === 'bigint') return value.toString();
  if (typeof value !== 'object') return sanitizeMetadataValue(stringifyContent(value), options, depth);

  if (depth >= METADATA_MAX_DEPTH) {
    return sanitizeMetadataValue(stringifyContent(value), options, depth);
  }

  if (Array.isArray(value)) {
    const kept = value.slice(0, METADATA_MAX_ENTRIES).map((item) =>
      sanitizeMetadataValue(item, options, depth + 1),
    );
    if (value.length > METADATA_MAX_ENTRIES) {
      kept.push(`…[${value.length - METADATA_MAX_ENTRIES} more]`);
    }
    return kept;
  }

  // Anything with its own rendering (Error, Map, Set, class instance) goes
  // through the same serializer the content path uses, then gets redacted.
  if (!isPlainObject(value)) {
    return sanitizeMetadataValue(stringifyContent(value), options, depth);
  }

  const out: Record<string, unknown> = {};
  let kept = 0;
  for (const [key, item] of Object.entries(value)) {
    if (kept >= METADATA_MAX_ENTRIES) {
      out['…'] = `[${Object.keys(value).length - kept} more keys]`;
      break;
    }
    out[key] = sanitizeMetadataValue(item, options, depth + 1);
    kept++;
  }
  return out;
}

/** Per-value cap inside metadata. */
const METADATA_VALUE_MAX_CHARS = 4_000;

function isPlainObject(value: object): boolean {
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}
