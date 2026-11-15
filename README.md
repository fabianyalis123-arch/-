# Free-model aggregator

One server, two APIs. It exposes OpenAI Chat Completions
(`POST /v1/chat/completions`) and Anthropic Messages (`POST /v1/messages`),
plus `GET /v1/models` in both shapes, and forwards every request to one of
ten free-tier providers, translating between the OpenAI, Anthropic and
Gemini wires in both directions. A single self-contained file, Python 3
standard library only.

```sh
python3 main.py            # serve (needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE)
python3 main.py --list     # print the catalog document
```

## How it works

### The brief
The trial brief (`MAKEFUN_PROMPT_FILE`) is a text file with three embedded
JSON documents: a **provider directory** (the ten providers, each with its
wire, auth style, key env var and base path), a **sealed quality table**
(rating in milli, free flag, context length, tools/vision/JSON support per
model) and the **workload manifest** (the 20 requests, their weights and
their requirements). The server reads all three with a brace-matching scan
that ignores the surrounding prose.

### The catalog
At startup the server probes every provider's catalog route in that
provider's own wire (`GET /v1/models`, or `GET /v1beta/models` for Gemini),
with the provider's own auth header, and builds one catalog:

```json
{"providers": [{"id": "groq", "base_url": "/p/groq",
                "key_env": "GROQ_API_KEY", "wire": "openai",
                "models": [{"id": "gpt-oss-120b", "free": true,
                            "context_length": 131072}]}]}
```

`base_url` is the provider **path** only, never joined to the environment
URL. The same object is printed by `--list` and sent in the register body,
byte for byte. If a probe fails, the sealed rating table fills in that
provider's models. `GET /v1/models` lists exactly the free models from that
catalog — OpenAI shape for a `Bearer` client, Anthropic shape for an
`x-api-key` client.

### Routing: best model first, planned up front
For each request the server ranks every free model by its sealed rating,
subject to the request's requirements (tools, vision, JSON mode, minimum
context) and to whether the provider's wire can carry JSON mode (the
Anthropic wire never echoes `json_mode`, so OpenAI JSON requests never go
there). It plans the whole workload up front, giving each provider a soft
call cap equal to the smallest budget the trial generator can draw for that
provider, so the best model is used as much as possible and never exhausted
mid-request. The workload's weights decide who gets the best models first.

### Failover: retry when transient, move on when exhausted
A refusal is a 402 (free allowance gone, pollinations) or a 429 (rate limit,
the other nine), sometimes with `retry-after: 1`. When a provider refuses,
the server retries the **same** provider once if its call count is still
below the soft cap — a scheduled failure early in a provider's life is
transient, and retrying keeps the best model at the same call cost as
failing over. Once the cap is used up, a refusal means the provider is out,
so it is marked exhausted and the request moves to the next-best model.
Two consecutive failures retire a provider as well. Only one successful
provider call is ever made per request.

### Wire fidelity: translate, don't rewrite
Requests are parsed once into a canonical form (system prompt, messages
with text/images/tool calls, tools, `max_tokens`, stop sequences, JSON
mode) and rendered in the provider's wire:

* **OpenAI ↔ OpenAI**: the body passes through, model swapped.
* **Anthropic ↔ Anthropic**: the body passes through, model swapped.
* **OpenAI/Anthropic → Gemini**: `systemInstruction`, `contents`,
  `functionDeclarations`, `generationConfig` (`maxOutputTokens`,
  `stopSequences`, `responseMimeType: "application/json"` for JSON mode).
* **Gemini is never a client wire**, only a provider wire.

Images are carried as base64 data (or as URLs) in each wire's own shape,
so the provider's echo of the image hash matches the client's. Streamed
OpenAI calls always set `stream_options.include_usage: true`, so the
provider emits the final usage chunk the client needs to verify usage.

The provider's answer is returned verbatim in the wire the client used,
streamed or not as asked, with the provider's own usage counts and the stop
reason mapped onto the client's family (`stop`/`end_turn`/`STOP`,
`length`/`max_tokens`/`MAX_TOKENS`, `tool_calls`/`tool_use`,
`content_filter`/`refusal`/`SAFETY`). Provider streams are buffered and
re-emitted, so chunk boundaries are deterministic.

### Lifecycle and determinism
The server listens on an ephemeral port (never 8080), registers with
`POST $MAKEFUN_ENV_URL/register` (`{protocol_version: 1, port, catalog}`),
serves, polls `GET $MAKEFUN_ENV_URL/work` every 100 ms and exits 0 when the
environment says done. There is no wall clock and no unseeded randomness
anywhere: ordering falls back to provider order and model id, never to
timings or hashes, so two runs of the same trial produce the same result.

The same code, given real keys and base URLs, runs against the real
services — the trial providers are just the real wires with simulated
budgets and failures.

## The ten providers

| Provider | Wire | Auth | Best free model (milli) |
| --- | --- | --- | --- |
| groq | OpenAI | Bearer | qwen3.8-27b (1434) |
| cerebras | OpenAI | Bearer | qwen-3.8-27b (1434) |
| gemini | Gemini | x-goog-api-key | gemini-3.8-flash (1458) |
| zai | Anthropic | Bearer | glm-4.7-flash (1470) |
| openrouter | OpenAI | Bearer | z-ai/glm-5.3-flash:free (1475) |
| sambanova | OpenAI | Bearer | deepseek-v3.2 (1465) |
| pollinations | OpenAI | none | openai-fast (1390) |
| cloudflare | OpenAI | Bearer | glm-4.7-flash (1470) |
| ovh | OpenAI | none | qwen3.5-397b (1440) |
| kilol | OpenAI | none | kilo-auto/free (1420) |

`gemini-3.8-flash` is the only top model that carries both tools and
vision; `z-ai/glm-5.3-flash:free` on openrouter is the best overall model.
Paid models (e.g. openrouter's `anthropic/claude-opus-5.5`) are never
routed to.

## Testing

Run the kit's mock environment and score a practice trial:

```sh
MAKEFUN_ENV_DATA_DIR=kit/free-model-aggregator-01-kit/practice/trial-0/input/env \
MAKEFUN_STATE_DIR=/tmp/state-0 MAKEFUN_ENV_PORT=8080 \
  node kit/free-model-aggregator-01-kit/program/env/server.mjs &
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=kit/free-model-aggregator-01-kit/practice/trial-0/input/prompt.txt \
  python3 main.py
# then SIGTERM the env server to export state, and score it:
MAKEFUN_OUTPUT_DIR=/tmp/state-0 \
MAKEFUN_REFERENCE_DIR=kit/free-model-aggregator-01-kit/practice/trial-0/reference \
MAKEFUN_SCORES_DIR=/tmp/scores-0 \
  node kit/free-model-aggregator-01-kit/program/scorer/score.mjs
```

## License

MIT. See `LICENSE`.
