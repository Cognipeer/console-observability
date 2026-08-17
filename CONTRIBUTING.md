# Contributing

Thanks for helping. The most useful contribution is usually **a new framework
integration** — the work is deliberately mechanical, and the sections below
describe exactly what it involves.

## Layout

```
js/       @cognipeer/observability      TypeScript package
python/   cognipeer-observability       Python package
examples/ runnable examples per framework
docs/     per-framework integration guides
```

The two packages are independent — no shared build, no code generation. They
are kept behaviourally in step by mirroring each other's structure and by
sharing one specification, [`docs/data-model.md`](docs/data-model.md).

## Local setup

```bash
# TypeScript
cd js && npm install && npm run build && npm test

# Python
cd python && pip install -e ".[all,dev]" && pytest
```

## The rules an integration must follow

These are not style preferences. Each one exists because breaking it turns an
observability package into an outage.

1. **Never raise into the traced application.** Wrap your handler bodies. Most
   frameworks swallow exporter exceptions and log at warn level, so a crashing
   integration is invisible and looks like "tracing just doesn't work".
2. **Never block the traced application.** Do not await network I/O in a
   callback. Record into the session; the transport handles delivery on its
   own thread (Python) or promise chain (JS).
3. **Import the framework lazily.** It is an optional peer dependency in JS and
   an extra in Python. Importing the core must never pull it in.
4. **Absent, not zero.** If the framework did not report token usage, leave the
   field unset. A zero silently under-reports spend.
5. **Say what you cannot capture.** Every integration's doc page has a "what is
   captured" table with honest ✗ entries and a one-line reason. A framework
   that cannot expose tool schemas is a fact to document, not to paper over.
6. **Do not invent data.** If a value is reconstructed rather than observed —
   as n8n's tool menu is, from the workflow JSON — mark its provenance in
   `metadata`.

## Adding a framework

1. **Find the seam.** In order of preference: a first-class tracing/callback
   interface → an event bus → an OTel instrumentor already emitting
   OpenInference or OpenLLMetry attributes (in which case the existing OTLP
   exporter may already cover it and no new code is needed) → middleware →
   monkeypatching (last resort; say so in the docs).
2. **Read the framework's source, not just its docs.** This ecosystem moves
   fast enough that published docs are routinely a version or two behind, and
   argument order has been known to differ between a package's `.d.ts` and its
   runtime.
3. **Write the mapping** onto the model in
   [`docs/data-model.md`](docs/data-model.md). Use `session.openSpan` /
   `closeSpan` (JS) or `open_span` / `close_span` (Python) when the framework
   gives you paired start/end callbacks — one pair becomes one event carrying
   both sides.
4. **Register the entry point.** JS: an `exports` subpath in `package.json` and
   an entry in `tsup.config.ts`. Python: a top-level module plus an extra in
   `pyproject.toml`.
5. **Add a runnable example** under `examples/`.
6. **Add the doc page** under `docs/`, following the shape of the existing
   ones: orientation, minimal wiring, options table, thread grouping, what is
   captured, version matrix, gotchas.

## Testing

Unit-test the mapping, not the network. Both packages let you swap the
transport, so an integration test is: drive the real framework with a fake
model, capture what would have gone on the wire, and assert on the payload.

```python
from cognipeer_observability import _transport
sent = []
_transport.Transport._submit = lambda self, path, payload: sent.append((path, payload))
```

A test that needs a live model API key does not belong in CI.

## Pull requests

Keep them to one framework. Include the framework versions you tested against —
"works on LangChain" is not checkable, "verified on langchain-core 1.2.8 and
0.3.75" is.
