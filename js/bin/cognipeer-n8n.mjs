#!/usr/bin/env node
/**
 * `cognipeer-n8n` — mirror n8n executions into Cognipeer Console.
 *
 *   npx @cognipeer/observability cognipeer-n8n \
 *     --n8n-url https://n8n.acme.com \
 *     --n8n-api-key $N8N_API_KEY \
 *     --api-key $COGNIPEER_API_KEY
 *
 * Every flag also reads an environment variable, so a container needs no
 * arguments at all. `--once` mirrors the current page and exits, which is what
 * you want in a cron job or to verify the wiring.
 */

import { init } from '../dist/index.js';
import { N8nBridge } from '../dist/integrations/n8n.js';

const FLAGS = {
  'n8n-url': 'N8N_URL',
  'n8n-api-key': 'N8N_API_KEY',
  'api-key': 'COGNIPEER_API_KEY',
  'base-url': 'COGNIPEER_BASE_URL',
  'workflow-id': 'N8N_WORKFLOW_ID',
  interval: 'COGNIPEER_N8N_INTERVAL',
  'page-size': 'COGNIPEER_N8N_PAGE_SIZE',
  since: 'COGNIPEER_N8N_SINCE',
};

/** Flags that may appear more than once and collect into an array. */
const REPEATABLE = new Set(['workflow-id']);

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const [flag, inlineValue] = token.slice(2).split('=');
    if (flag === 'once' || flag === 'help' || flag === 'debug') {
      args[flag] = true;
      continue;
    }
    const value = inlineValue ?? argv[++i];
    if (REPEATABLE.has(flag)) {
      args[flag] = [...(args[flag] ?? []), value];
      continue;
    }
    args[flag] = value;
  }
  for (const [flag, envName] of Object.entries(FLAGS)) {
    if (args[flag] !== undefined || !process.env[envName]) continue;
    // The env form takes a comma-separated list for repeatable flags.
    args[flag] = REPEATABLE.has(flag)
      ? process.env[envName].split(',').map((value) => value.trim()).filter(Boolean)
      : process.env[envName];
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));

if (args.help) {
  console.log(`cognipeer-n8n — mirror n8n executions into Cognipeer Console

  --n8n-url <url>        n8n base URL              (env N8N_URL)
  --n8n-api-key <key>    n8n API key               (env N8N_API_KEY)
  --api-key <key>        Cognipeer API token       (env COGNIPEER_API_KEY)
  --base-url <url>       Console URL               (env COGNIPEER_BASE_URL)
  --workflow-id <id>     Only this workflow; repeat for several
                         (env N8N_WORKFLOW_ID, comma-separated)
  --interval <seconds>   Poll interval, default 15
  --page-size <n>        Executions per poll, default 50, max 250
  --since <iso>          Ignore runs that finished before this instant
  --once                 Mirror one page and exit
  --debug                Log every mirrored session
`);
  process.exit(0);
}

const missing = ['n8n-url', 'n8n-api-key', 'api-key'].filter((flag) => !args[flag]);
if (missing.length > 0) {
  console.error(`cognipeer-n8n: missing ${missing.map((flag) => `--${flag}`).join(', ')}`);
  console.error('Run with --help for usage.');
  process.exit(1);
}

init({ apiKey: args['api-key'], baseUrl: args['base-url'], debug: Boolean(args.debug) });

const bridge = new N8nBridge({
  n8nUrl: args['n8n-url'],
  n8nApiKey: args['n8n-api-key'],
  workflowIds: args['workflow-id'],
  intervalMs: args.interval ? Number(args.interval) * 1000 : undefined,
  pageSize: args['page-size'] ? Number(args['page-size']) : undefined,
  // Default to mirroring history from process start; `--since` widens it.
  since: args.since ? new Date(args.since) : undefined,
  onSession: (payload) => {
    console.log(
      `[cognipeer] mirrored ${payload.sessionId} (${payload.events?.length ?? 0} events)`,
    );
  },
  onError: (error) => console.error('[cognipeer] n8n bridge:', error.message),
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    bridge.stop();
    process.exit(0);
  });
}

if (args.once) {
  const count = await bridge.runOnce();
  console.log(`[cognipeer] mirrored ${count} execution(s)`);
} else {
  console.log(`[cognipeer] watching ${args['n8n-url']} …`);
  await bridge.start();
}
