/**
 * n8n external hook — mirrors every finished execution into Cognipeer Console.
 *
 * Push-based and immediate, unlike the polling bridge, but self-hosted only:
 * it needs a file on the n8n container's filesystem.
 *
 * INSTALL
 *   1. Make this file and the SDK reachable from the n8n container, e.g.
 *        docker cp cognipeer-hook.js n8n:/opt/cognipeer/cognipeer-hook.js
 *        docker exec n8n npm install -g @cognipeer/observability
 *   2. Set on the n8n container (ALL of them in queue mode — main, workers
 *      and webhook processors):
 *        EXTERNAL_HOOK_FILES=/opt/cognipeer/cognipeer-hook.js
 *        COGNIPEER_API_KEY=cpeer_…
 *        COGNIPEER_BASE_URL=https://console.cognipeer.com   # self-hosted only
 *        NODE_PATH=/usr/local/lib/node_modules              # so require() finds the global install
 *   3. Restart n8n, run any workflow, and look in Tracing → Sessions.
 *
 * CommonJS on purpose: n8n `require()`s this file, so there is no build step
 * and no ESM interop to get wrong.
 */

const { init } = require('@cognipeer/observability');
const { createN8nExternalHook } = require('@cognipeer/observability/n8n');

if (!process.env.COGNIPEER_API_KEY) {
  // Never throw from a hook file — n8n loads it during boot, and a throw here
  // takes the whole instance down. Warn and export a no-op instead.
  console.warn('[cognipeer] COGNIPEER_API_KEY is not set; n8n executions will not be mirrored.');
}

init({
  apiKey: process.env.COGNIPEER_API_KEY,
  baseUrl: process.env.COGNIPEER_BASE_URL,
  debug: process.env.COGNIPEER_DEBUG === '1',
});

module.exports = createN8nExternalHook({
  // Leave manual runs traced so clicking "Test workflow" verifies the wiring.
  // Flip to true once you are in production and only want scheduled/triggered
  // executions.
  productionOnly: process.env.COGNIPEER_N8N_PRODUCTION_ONLY === '1',

  // n8n has no first-class conversation id. If your workflows set one with
  // `$execution.customData.set('threadId', …)` it is picked up automatically;
  // otherwise pin every execution of this instance to one thread here.
  // threadId: 'n8n-main',
});
