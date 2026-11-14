// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The brief at $MAKEFUN_PROMPT_FILE carries the provider directory, the sealed
// quality table and the workload manifest. This server probes every provider's
// catalog route in its own wire, builds one catalog, registers its port and
// catalog with the env at $MAKEFUN_ENV_URL, serves the client's requests, and
// exits 0 when GET /work says the workload is done.
//
// Routing ranks every free model that meets a request's requirements by the
// quality table's rating, then same-wire first (a request that needs no
// translation keeps its echo intact), then fewest calls used so far, which
// spreads the load over the providers' per-trial budgets. Each provider is
// capped at a small number of calls a trial; a provider that refuses twice in a
// row is treated as spent or out and skipped until every candidate is capped or
// spent, when the router tries them again best first. A JSON-mode request from
// an OpenAI client never goes to an Anthropic-wire provider, whose wire cannot
// carry response_format.
//
// Every upstream answer is read whole before it is converted, so a refused call
// (402, 429, 5xx, or a body that is not the answer it promised) is retried on
// the next candidate without ever sending a partial answer, and a 2xx is never
// retried: the env counts exactly one successful provider call a request.
//
// Node 22's standard library alone: nothing is installed.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsFromGeminiText,
  geminiRequestFromAnthropic,
  geminiRequestFromOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIEventsFromGeminiText,
} from './lib/gemini.ts';
import {
  anthropicToOpenAIResponse,
  openAIEventsFromAnthropicText,
  openAIToAnthropicRequest,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most upstream calls one provider serves before the router rests it. */
const PROVIDER_CALL_CAP = 4;
/** Consecutive refusals after which a provider is treated as spent or out. */
const DEAD_AFTER_FAILURES = 2;

type ClientWire = 'openai' | 'anthropic';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let arrivalIndex = 0;

/** The trial state of one provider: calls spent, refusals in a row, and whether it is rested. */
interface ProviderState {
  calls: number;
  failuresInRow: number;
  dead: boolean;
}

const states = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = states.get(providerId);
  if (state === undefined) {
    state = { calls: 0, failuresInRow: 0, dead: false };
    states.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function loadCatalog(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** One free model that can serve a request, with the rating that ranks it. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

const compareText = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * Every free model meeting `requires`, best first: the quality table's rating,
 * then the client's own wire, then the provider with the fewest calls spent,
 * then provider and model id, so the order is the same on every run.
 */
function rankedCandidates(requires: Requirement, clientWire: ClientWire, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && clientWire === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const aSame = a.probe.provider.wire === clientWire ? 0 : 1;
    const bSame = b.probe.provider.wire === clientWire ? 0 : 1;
    if (aSame !== bSame) return aSame - bSame;
    const calls = stateOf(a.probe.provider.id).calls - stateOf(b.probe.provider.id).calls;
    if (calls !== 0) return calls;
    const provider = compareText(a.probe.provider.id, b.probe.provider.id);
    if (provider !== 0) return provider;
    return compareText(a.model.id, b.model.id);
  });
  return out;
}

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => compareText(a.id, b.id))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[arrivalIndex];
  arrivalIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function sendEvents(res: ServerResponse, events: string[]): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for (const event of events) res.write(event);
  res.end();
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream answer, read whole: its status and its body text. */
interface UpstreamAnswer {
  status: number;
  contentType: string;
  text: string;
}

/**
 * One inference call to a candidate's provider, in the provider's own wire, with
 * the client's body translated into it. The answer is buffered whole, so a
 * refusal never sends a partial answer downstream.
 */
async function callUpstream(candidate: Candidate, clientWire: ClientWire, body: Json): Promise<UpstreamAnswer> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody =
      clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (upstreamBody.stream === true) upstreamBody.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody =
      clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
  } else {
    const stream = body.stream === true;
    const translated = clientWire === 'openai' ? geminiRequestFromOpenAI(body, candidate.model.id) : geminiRequestFromAnthropic(body, candidate.model.id);
    url = `${base}/v1beta/models/${encodeURIComponent(translated.model)}:${stream ? 'streamGenerateContent' : 'generateContent'}${stream ? '?alt=sse' : ''}`;
    upstreamBody = translated.body;
  }
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  return { status: res.status, contentType: res.headers.get('content-type') ?? 'application/json', text: await res.text() };
}

/** A buffered upstream answer converted into the client's wire, or null when it is not a usable answer. */
async function convertAnswer(
  candidate: Candidate,
  clientWire: ClientWire,
  body: Json,
  answer: UpstreamAnswer,
): Promise<{ json: Json | null; events: string[] | null; passthrough: string | null }> {
  const providerWire = candidate.probe.provider.wire;
  const stream = body.stream === true;
  if (clientWire === providerWire) {
    // The same wire both ways: the provider's bytes are the answer.
    return { json: null, events: null, passthrough: answer.text };
  }
  if (stream) {
    if (clientWire === 'openai' && providerWire === 'anthropic') {
      return { json: null, events: openAIEventsFromAnthropicText(answer.text, candidate.model.id), passthrough: null };
    }
    if (clientWire === 'openai' && providerWire === 'gemini') {
      return { json: null, events: openAIEventsFromGeminiText(answer.text, candidate.model.id), passthrough: null };
    }
    if (clientWire === 'anthropic' && providerWire === 'openai') {
      const streamBody = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(answer.text));
          controller.close();
        },
      });
      const events: string[] = [];
      for await (const event of anthropicEventsFromOpenAI(streamBody, candidate.model.id)) events.push(event);
      return { json: null, events, passthrough: null };
    }
    // Anthropic client, Gemini provider.
    return { json: null, events: anthropicEventsFromGeminiText(answer.text, candidate.model.id), passthrough: null };
  }
  const parsed: unknown = JSON.parse(answer.text);
  if (!isRecord(parsed)) return { json: null, events: null, passthrough: null };
  if (clientWire === 'openai' && providerWire === 'anthropic') {
    return { json: anthropicToOpenAIResponse(parsed, candidate.model.id), events: null, passthrough: null };
  }
  if (clientWire === 'openai' && providerWire === 'gemini') {
    return { json: geminiToOpenAI(parsed, candidate.model.id), events: null, passthrough: null };
  }
  if (clientWire === 'anthropic' && providerWire === 'openai') {
    return { json: openAIToAnthropic(parsed, candidate.model.id), events: null, passthrough: null };
  }
  // Anthropic client, Gemini provider.
  return { json: geminiToAnthropic(parsed, candidate.model.id), events: null, passthrough: null };
}

/** Record one upstream attempt: a success clears the refusal streak, a refusal grows it. */
function recordAttempt(candidate: Candidate, ok: boolean): void {
  const state = stateOf(candidate.probe.provider.id);
  state.calls += 1;
  if (ok) {
    state.failuresInRow = 0;
    return;
  }
  state.failuresInRow += 1;
  if (state.failuresInRow >= DEAD_AFTER_FAILURES) state.dead = true;
}

/**
 * Serve one client request: walk the ranked candidates, best first, resting
 * capped or spent providers on the first pass and retrying them anyway when
 * every candidate is rested. A provider that already refused this request is
 * not tried again until nothing else is left.
 */
async function serveRequest(clientWire: ClientWire, body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const format = clientWire === 'openai' && isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonMode = format === 'json_object' || format === 'json_schema';
  const ranked = rankedCandidates(requires, clientWire, jsonMode);
  if (ranked.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const refusedThisRequest = new Set<string>();
  const candidateKey = (candidate: Candidate): string => `${candidate.probe.provider.id}/${candidate.model.id}`;
  const passes: Candidate[][] = [
    ranked.filter((candidate) => {
      const state = stateOf(candidate.probe.provider.id);
      return !state.dead && state.calls < PROVIDER_CALL_CAP;
    }),
    ranked,
  ];
  for (const pass of passes) {
    for (const candidate of pass) {
      const key = candidateKey(candidate);
      if (refusedThisRequest.has(key)) continue;
      let answer: UpstreamAnswer;
      try {
        answer = await callUpstream(candidate, clientWire, body);
      } catch (err) {
        recordAttempt(candidate, false);
        refusedThisRequest.add(key);
        log(`upstream ${candidate.probe.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
        continue;
      }
      if (answer.status < 200 || answer.status >= 300) {
        recordAttempt(candidate, false);
        refusedThisRequest.add(key);
        log(`upstream ${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${answer.status}`);
        continue;
      }
      let converted;
      try {
        converted = await convertAnswer(candidate, clientWire, body, answer);
      } catch (err) {
        recordAttempt(candidate, false);
        refusedThisRequest.add(key);
        log(`upstream ${candidate.probe.provider.id}/${candidate.model.id} answer unusable: ${(err as Error).message}`);
        continue;
      }
      if (converted.json === null && converted.events === null && converted.passthrough === null) {
        recordAttempt(candidate, false);
        refusedThisRequest.add(key);
        continue;
      }
      recordAttempt(candidate, true);
      if (converted.passthrough !== null) {
        sendText(res, 200, answer.contentType, converted.passthrough);
        return;
      }
      if (converted.events !== null) {
        sendEvents(res, converted.events);
        return;
      }
      sendJson(res, 200, converted.json);
      return;
    }
  }
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chatRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serveRequest('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messagesRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serveRequest('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chatRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messagesRoute(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await loadCatalog();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
