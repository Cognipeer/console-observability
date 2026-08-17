/**
 * W3C-shaped identifier helpers.
 *
 * The Console stores `traceId` / `spanId` / `parentSpanId` as opaque strings,
 * but keeping them W3C-valid (32 / 16 lower-hex chars) means a trace can be
 * correlated with whatever other OTel backend the customer already runs.
 * Framework run ids are usually UUIDs, so `spanIdFrom` folds an arbitrary
 * string into a stable 16-hex id instead of inventing a new one.
 */

const HEX = '0123456789abcdef';

function randomHex(length: number): string {
  const bytes = new Uint8Array(length / 2);
  const cryptoObj = (globalThis as { crypto?: Crypto }).crypto;
  if (cryptoObj?.getRandomValues) {
    cryptoObj.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  let out = '';
  for (const byte of bytes) {
    out += HEX[(byte >> 4) & 0xf] + HEX[byte & 0xf];
  }
  return out;
}

/** New random 32-hex trace id. */
export function newTraceId(): string {
  return randomHex(32);
}

/** New random 16-hex span id. */
export function newSpanId(): string {
  return randomHex(16);
}

/** Readable session id: `sess_<24 hex>`. */
export function newSessionId(prefix = 'sess'): string {
  return `${prefix}_${randomHex(24)}`;
}

/**
 * Deterministically fold any string (a UUID run id, an OTel span id, a
 * framework-generated key) into a 16-hex span id. Same input always yields the
 * same span id, which is what lets `parentSpanId` be resolved from a parent
 * run id that we may never have seen as an event ourselves.
 *
 * Only an input that is *already* a span id passes through. Truncating a
 * longer identifier is not safe: LangChain run ids are UUIDv7, whose first 16
 * hex digits are a millisecond timestamp plus 12 bits of entropy, so
 * truncation collides between runs started in the same millisecond — and
 * colliding span ids silently corrupt the trace tree.
 *
 * FNV-1a doubled to 64 bits — enough for span identity, and it keeps the
 * package dependency-free.
 */
export function spanIdFrom(value: string): string {
  if (/^[0-9a-f]{16}$/i.test(value)) return value.toLowerCase();
  return `${fnv1a(value)}${fnv1a(`${value}#2`)}`;
}

/** Fold any string into a 32-hex trace id (see `spanIdFrom`). */
export function traceIdFrom(value: string): string {
  if (/^[0-9a-f]{32}$/i.test(value)) return value.toLowerCase();
  return `${fnv1a(value)}${fnv1a(`${value}#2`)}${fnv1a(`${value}#3`)}${fnv1a(`${value}#4`)}`;
}

/** 32-bit FNV-1a rendered as 8 lower-hex chars. */
function fnv1a(input: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    // hash *= 16777619, kept in 32-bit space without BigInt
    hash = (hash + ((hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24))) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}
