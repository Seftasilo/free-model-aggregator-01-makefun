# Free Model Aggregator

One server, two client wires, ten free providers. It serves the OpenAI Chat Completions
API (`POST /v1/chat/completions`) and the Anthropic Messages API (`POST /v1/messages`),
in JSON and in server-sent events, routing every request to the best-rated free model
that meets its requirements, and prints the catalog it found with `--list`.

## How it works

- The brief at `$MAKEFUN_PROMPT_FILE` carries the provider directory, the sealed quality
  table and the workload manifest. The server probes every provider's catalog route in
  its own wire (OpenAI, Anthropic, Gemini), builds one catalog, registers its port and
  catalog with the env at `$MAKEFUN_ENV_URL`, serves the client's 20 requests, and exits
  0 when `GET /work` says the workload is done.
- Routing ranks free candidates by the quality table's `rating_milli`, then same-wire
  first (no translation), then fewest calls used so far (which spreads the load across
  providers' per-trial budgets). Each provider is capped at 4 calls a trial; a provider
  that refuses twice in a row is marked dead for the trial (one refusal is its failure
  schedule, two in a row is a spent budget or an outage). When every candidate is capped
  or dead, the router retries them anyway, best first.
- A JSON-mode request from an OpenAI client never goes to an Anthropic-wire provider,
  whose wire cannot carry `response_format`; Gemini carries it as
  `responseMimeType: application/json`.
- Every upstream answer is buffered whole before conversion, so a refused call
  (402/429/5xx) is retried on the next candidate without ever sending a partial answer,
  and a 2xx is never retried: the env counts exactly one successful provider call per
  request.
- OpenAI streams always set `stream_options.include_usage` upstream: without the usage
  chunk the client's usage check fails, and the client never asks for it itself.
- Deterministic: no wall clock, no unseeded randomness; every choice is a function of
  the brief, the requests and the providers' answers.

## Layout

- `main.ts` — the server: brief, catalog, routing with failover, register and poll.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the catalog probe
  (from the kit's starter).
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation and the SSE writers (from the
  kit's starter).
- `lib/gemini.ts` — the Gemini wire: request and response translation both ways.
- `lib/translate.ts` — the Anthropic upstream wire for an OpenAI client, and its answers
  translated back to the OpenAI wire.

## Run

```sh
node main.ts --list   # print the catalog document and exit 0
node main.ts          # serve (needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL)
```

Node 22's standard library alone: nothing is installed.
