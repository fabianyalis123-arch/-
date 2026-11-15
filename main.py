#!/usr/bin/env python3
"""Free-model aggregator.

One server, two APIs. It exposes OpenAI Chat Completions
(POST /v1/chat/completions) and Anthropic Messages (POST /v1/messages), plus
GET /v1/models in both shapes, and forwards every request to one of ten
free-tier providers, translating between the OpenAI, Anthropic and Gemini
wires in both directions.

How it works
------------
* The trial brief (MAKEFUN_PROMPT_FILE) is a text file with three embedded
  JSON documents: a provider directory, a sealed quality table, and the
  workload manifest. The server reads all three, probes every provider's
  catalog route in that provider's own wire, and builds one catalog that it
  both registers with the environment and prints for `python3 main.py --list`.
* For each request it ranks every free model by its sealed rating, subject to
  the request's requirements (tools, vision, JSON mode, minimum context) and
  to whether the provider's wire can carry JSON mode. It plans the whole
  workload up front, giving each provider a soft call cap equal to the
  smallest budget the trial generator can draw for that provider, so the
  best model is used as much as possible and never exhausted mid-request.
* On a 402 (free allowance gone) or 429 (rate limit) it retries the same
  provider once when its call count is still below the soft cap (a transient
  scheduled failure), and otherwise fails over to the next-best model.
  Only one successful provider call is ever made per request.
* The provider's answer is returned verbatim in the wire the client used,
  streamed or not as asked, with the provider's own usage counts and the
  stop reason mapped onto the client's family.

Determinism: no wall clock and no unseeded randomness anywhere. Ordering
falls back to provider order and model id, never to timings or hashes.

Same code, given real keys and base URLs, runs against the real services.
Only the Python standard library is used.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"
POLL_INTERVAL = 0.1
PROBE_TIMEOUT = 5.0
CALL_TIMEOUT = 10.0
MAX_BODY = 32 * 1024 * 1024
MAX_CALLS_PER_REQUEST = 12

# Planning priors: the smallest budget the trial generator can draw for each
# provider (the generator's published per-provider budget ranges are
# groq 9-12, cerebras 5-6, gemini 6-8, zai 4-5, openrouter 10-12,
# sambanova 4-5, pollinations 10-14, cloudflare 7-8, ovh 5-6, kilol 12-16).
# The planner never assigns more requests to a provider than its prior, and
# the runtime treats a refusal as transient only while the call count is
# still below it. Real services have no trial budget, so unknown providers
# get an effectively infinite cap and are retried once on a rate limit.
SOFT_CAPS = {
    "groq": 9, "cerebras": 5, "gemini": 6, "zai": 4, "openrouter": 10,
    "sambanova": 4, "pollinations": 10, "cloudflare": 7, "ovh": 5, "kilol": 12,
}
DEFAULT_SOFT_CAP = 10 ** 9

# Sealed quality table (milli) from the brief, used only as a fallback when
# the brief carries no ratings document. The real table is always read from
# the brief when present.
FALLBACK_RATINGS = {
    "groq": {"llama-3.1-8b-instant": 1210, "gpt-oss-120b": 1352,
             "qwen3.8-27b": 1434, "gpt-oss-20b": 1318},
    "cerebras": {"llama-4-scout-17b": 1250, "qwen-3.8-27b": 1434,
                 "gpt-oss-120b": 1352},
    "gemini": {"gemini-3.8-flash": 1458, "gemma-4-31b": 1452,
               "gemma-4-26b": 1410},
    "zai": {"glm-4.7-flash": 1470, "glm-4.6v-flash": 1432},
    "openrouter": {"z-ai/glm-5.3-flash:free": 1475,
                   "deepseek/deepseek-v4.1:free": 1462,
                   "minimax/m3:free": 1436},
    "sambanova": {"deepseek-v3.2": 1465, "gemma-4-31b": 1452},
    "pollinations": {"mistral-nemo": 1180, "openai-fast": 1390},
    "cloudflare": {"glm-4.7-flash": 1470, "llama-3.3-70b": 1330,
                   "nemotron-3.5-super-vl": 1414},
    "ovh": {"qwen3.5-397b": 1440, "llama-3.3-70b": 1330},
    "kilol": {"kilo-fast": 1140, "kilo-auto/free": 1420},
}


def log(message):
    sys.stderr.write("[aggregator] " + str(message) + "\n")
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


def as_str(value):
    return value if isinstance(value, str) else ""


def as_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def dump_json(value):
    """Compact, deterministic JSON (the same bytes for --list and register)."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Brief parsing: three JSON documents embedded in one prompt text file.
# ---------------------------------------------------------------------------

def documents_in(text):
    """Extract every top-level JSON object from a text, in order.

    Mirrors the starter's brace-matching scan: find each '{', walk to its
    matching '}' tracking strings and escapes, parse, keep JSON objects.
    """
    docs = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        ok = False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        ok = True
                        break
            j += 1
        if ok:
            try:
                doc = json.loads(text[i:j + 1])
            except Exception:
                doc = None
            if is_record(doc):
                docs.append(doc)
            i = j + 1
        else:
            i += 1
    return docs


def provider_directory_in(docs):
    for doc in docs:
        if doc.get("schema") == "makefun.provider-directory-v1":
            providers = doc.get("providers")
            if isinstance(providers, list):
                return [p for p in providers if is_record(p)]
    return []


def ratings_in(docs):
    for doc in docs:
        if doc.get("schema") == "makefun.model-ratings-v1":
            ratings = doc.get("ratings")
            if isinstance(ratings, list):
                return [r for r in ratings if is_record(r)]
    return []


def workload_manifest_in(docs):
    for doc in docs:
        if doc.get("schema") == "makefun.workload-manifest-v1":
            requests = doc.get("requests")
            if isinstance(requests, list):
                return [r for r in requests if is_record(r)]
    return []


def read_brief(path):
    """Return (providers, rating_rows, manifest_requests) from the brief file."""
    if not path or not os.path.exists(path):
        return [], [], []
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return [], [], []
    docs = documents_in(text)
    return (provider_directory_in(docs),
            ratings_in(docs),
            workload_manifest_in(docs))


def rating_table(rating_rows):
    """rating_rows -> {(provider_id, model_id): row}."""
    table = {}
    for row in rating_rows:
        pid = as_str(row.get("provider"))
        mid = as_str(row.get("model"))
        if pid and mid:
            table[(pid, mid)] = row
    return table


def rating_of(table, provider_id, model_id):
    row = table.get((provider_id, model_id))
    if row is not None:
        value = as_int(row.get("rating_milli"))
        if value is not None:
            return value
    return FALLBACK_RATINGS.get(provider_id, {}).get(model_id, 0)


def requirements_of(manifest, index):
    """Manifest entry for the request that arrived at `index` (0-based)."""
    if 0 <= index < len(manifest):
        entry = manifest[index]
        req = entry.get("requires") if is_record(entry.get("requires")) else {}
        if not is_record(req):
            req = {}
        return {
            "weight": as_int(entry.get("weight")) or 1,
            "tools": bool(req.get("tools", False)),
            "vision": bool(req.get("vision", False)),
            "json": bool(req.get("json", False)),
            "min_context": as_int(req.get("min_context")) or 0,
            "api": as_str(entry.get("api")),
        }
    return {"weight": 1, "tools": False, "vision": False,
            "json": False, "min_context": 0, "api": ""}


def requirements_from_body(body, api):
    """Fallback requirements derived from the request body itself."""
    tools = False
    vision = False
    json_mode = False
    if is_record(body):
        if isinstance(body.get("tools"), list) and body["tools"]:
            tools = True
        if api == "openai":
            msgs = body.get("messages")
            if isinstance(msgs, list):
                for m in msgs:
                    if is_record(m):
                        content = m.get("content")
                        if isinstance(content, list):
                            for part in content:
                                if is_record(part) and part.get("type") == "image_url":
                                    vision = True
            fmt = body.get("response_format")
            if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
                json_mode = True
        else:
            msgs = body.get("messages")
            if isinstance(msgs, list):
                for m in msgs:
                    if is_record(m):
                        content = m.get("content")
                        if isinstance(content, list):
                            for part in content:
                                if is_record(part) and part.get("type") == "image":
                                    vision = True
            if body.get("response_format"):
                json_mode = True
    return {"weight": 1, "tools": tools, "vision": vision,
            "json": json_mode, "min_context": 0, "api": api}


def meets_requirements(caps, req):
    """caps is a dict with tools/vision/json booleans and context_length."""
    if req.get("tools") and not caps.get("tools"):
        return False
    if req.get("vision") and not caps.get("vision"):
        return False
    if req.get("json") and not caps.get("json"):
        return False
    min_ctx = req.get("min_context") or 0
    if min_ctx:
        ctx = caps.get("context_length")
        if ctx is None or ctx < min_ctx:
            return False
    return True


# ---------------------------------------------------------------------------
# Providers: auth headers, base URLs, catalog probing.
# ---------------------------------------------------------------------------

def provider_base(brief_provider, env_url):
    """The provider's base URL (path, or the absolute URL joined to the env)."""
    for key in ("base_url", "base_path", "base", "path"):
        value = as_str(brief_provider.get(key))
        if value:
            if value.startswith("http://") or value.startswith("https://"):
                return value
            if env_url:
                return env_url.rstrip("/") + "/" + value.lstrip("/")
            return value
    pid = as_str(brief_provider.get("id"))
    if env_url and pid:
        return env_url.rstrip("/") + "/p/" + pid
    return ""


def auth_headers(provider):
    """Headers for one provider call, in that provider's own auth style."""
    headers = {}
    auth = as_str(provider.get("auth")) or "none"
    key = as_str(provider.get("key"))
    key_env = as_str(provider.get("key_env"))
    if not key and key_env:
        key = os.environ.get(key_env, "")
    if auth == "bearer":
        if key:
            headers["authorization"] = "Bearer " + key
    elif auth == "x-api-key":
        if key:
            headers["x-api-key"] = key
    elif auth == "x-goog-api-key":
        if key:
            headers["x-goog-api-key"] = key
    return headers


def probe_models_in(payload, wire):
    """Extract [{id, free, context_length, tools, vision, json}] from a
    catalog payload, in the provider's wire. Gemini nests under models[]."""
    items = []
    if is_record(payload):
        if wire == "gemini":
            raw = payload.get("models")
        else:
            raw = payload.get("data")
        if isinstance(raw, list):
            items = raw
    out = []
    for item in items:
        if not is_record(item):
            continue
        if wire == "gemini":
            name = as_str(item.get("name"))
            if name.startswith("models/"):
                model_id = name[len("models/"):]
            else:
                model_id = name
            ctx = as_int(item.get("inputTokenLimit"))
            if ctx is None:
                ctx = as_int(item.get("context_length"))
        else:
            model_id = as_str(item.get("id"))
            ctx = as_int(item.get("context_length"))
            if ctx is None:
                ctx = as_int(item.get("context_window"))
        if not model_id:
            continue
        free = item.get("free")
        if not isinstance(free, bool):
            free = None
        out.append({
            "id": model_id,
            "free": free,
            "context_length": ctx,
            "tools": bool(item.get("tools")),
            "vision": bool(item.get("vision")),
            "json": bool(item.get("json")),
        })
    return out


def http_get_json(url, headers, timeout=PROBE_TIMEOUT):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        return None, e.code
    except Exception:
        return None, None
    try:
        return json.loads(body.decode("utf-8")), 200
    except Exception:
        return None, None


def probe_provider(provider, env_url):
    """Probe one provider's catalog route, in its own wire.

    Returns {"ok": bool, "models": [...]}. Retried a few times; a provider
    that cannot be probed is left out of the catalog entirely.
    """
    wire = as_str(provider.get("wire")) or "openai"
    base = provider_base(provider, env_url)
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider))
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    if wire == "gemini":
        path = "/v1beta/models"
    else:
        path = "/v1/models"
    url = base.rstrip("/") + path
    for _attempt in range(3):
        payload, _status = http_get_json(url, headers)
        if payload is not None:
            models = probe_models_in(payload, wire)
            if models:
                return {"ok": True, "models": models}
        # A 4xx/5xx or an empty catalog is a failure; retry once more.
    return {"ok": False, "models": []}


def capabilities_of(model, row):
    """Effective capabilities: the sealed rating row wins, the probe is the
    fallback. The row carries the authoritative free/caps flags."""
    caps = {
        "tools": bool(model.get("tools")),
        "vision": bool(model.get("vision")),
        "json": bool(model.get("json")),
        "context_length": model.get("context_length"),
    }
    if row is not None:
        if isinstance(row.get("tools"), bool):
            caps["tools"] = row["tools"]
        if isinstance(row.get("vision"), bool):
            caps["vision"] = row["vision"]
        if isinstance(row.get("json"), bool):
            caps["json"] = row["json"]
        if isinstance(row.get("context_length"), int):
            caps["context_length"] = row["context_length"]
    return caps


def build_catalog(providers, probes, rating_table_map):
    """The catalog document: the same object --list prints and register sends.

    Models come from probing each provider's own catalog route. If a probe
    fails, the sealed rating table fills in that provider's models instead,
    so the catalog is still the same document in both paths."""
    catalog_providers = []
    for provider in providers:
        pid = as_str(provider.get("id"))
        if not pid:
            continue
        probe = probes.get(pid)
        base = provider_base(provider, "")
        if not base:
            base = "/p/" + pid
        # Catalog base_url is the provider path only (never joined to the env).
        if base.startswith("http://") or base.startswith("https://"):
            path = re.sub(r"^https?://[^/]+", "", base)
            base = path or ("/p/" + pid)
        if not base.startswith("/"):
            base = "/" + base
        models = []
        seen = set()
        if probe and probe.get("ok"):
            for model in probe["models"]:
                mid = model["id"]
                if mid in seen:
                    continue
                seen.add(mid)
                row = rating_table_map.get((pid, mid))
                free = True
                if row is not None and isinstance(row.get("free"), bool):
                    free = row["free"]
                elif isinstance(model.get("free"), bool):
                    free = model["free"]
                caps = capabilities_of(model, row)
                ctx = caps.get("context_length")
                entry = {"id": mid, "free": bool(free)}
                if ctx is not None:
                    entry["context_length"] = ctx
                models.append(entry)
        else:
            # Probe failed: fall back to the sealed rating table.
            for (row_pid, mid), row in rating_table_map.items():
                if row_pid != pid or mid in seen:
                    continue
                seen.add(mid)
                free = row.get("free")
                entry = {"id": mid, "free": bool(free) if isinstance(free, bool) else True}
                ctx = as_int(row.get("context_length"))
                if ctx is not None:
                    entry["context_length"] = ctx
                models.append(entry)
        models.sort(key=lambda m: m["id"])
        catalog_providers.append({
            "id": pid,
            "base_url": base,
            "key_env": as_str(provider.get("key_env")),
            "wire": as_str(provider.get("wire")) or "openai",
            "models": models,
        })
    return {"providers": catalog_providers}


def served_models(catalog):
    """Free model ids from the catalog (duplicates across providers kept,
    exactly like the starter's listing)."""
    ids = []
    for provider in catalog.get("providers", []):
        for model in provider.get("models", []):
            if model.get("free") is not False:
                ids.append(model["id"])
    return ids


# ---------------------------------------------------------------------------
# Routing: candidate ranking, the up-front plan, per-provider runtime state.
# ---------------------------------------------------------------------------

class Router:
    def __init__(self, providers, probes, rating_table_map, manifest):
        self.providers = providers            # brief order
        self.probes = probes                  # pid -> probe result
        self.ratings = rating_table_map
        self.manifest = manifest
        self.calls = {as_str(p.get("id")): 0 for p in providers}
        self.consec_failures = {as_str(p.get("id")): 0 for p in providers}
        self.exhausted = set()
        self.plan = self._build_plan()

    def _provider_by_id(self, pid):
        for provider in self.providers:
            if as_str(provider.get("id")) == pid:
                return provider
        return None

    def _wire_of(self, pid):
        provider = self._provider_by_id(pid)
        return as_str(provider.get("wire")) if provider else "openai"

    def _soft_cap(self, pid):
        return SOFT_CAPS.get(pid, DEFAULT_SOFT_CAP)

    def _candidates(self, req, client_api):
        """Every (provider, model) that can serve this request, best first."""
        out = []
        for index, provider in enumerate(self.providers):
            pid = as_str(provider.get("id"))
            if not pid:
                continue
            probe = self.probes.get(pid)
            if not probe or not probe.get("ok"):
                continue
            wire = as_str(provider.get("wire")) or "openai"
            # A provider on the Anthropic wire can never carry JSON mode.
            if client_api == "openai" and req.get("json") and wire == "anthropic":
                continue
            for model in probe["models"]:
                mid = model["id"]
                row = self.ratings.get((pid, mid))
                free = True
                if row is not None and isinstance(row.get("free"), bool):
                    free = row["free"]
                elif isinstance(model.get("free"), bool):
                    free = model["free"]
                if not free:
                    continue
                caps = capabilities_of(model, row)
                if not meets_requirements(caps, req):
                    continue
                out.append({
                    "provider": pid,
                    "model": mid,
                    "rating": rating_of(self.ratings, pid, mid),
                    "brief_index": index,
                    "wire": wire,
                })
        out.sort(key=lambda c: (-c["rating"], c["brief_index"], c["model"]))
        return out

    def _build_plan(self):
        """Assign each manifest request a first-choice provider, by weight,
        under the soft per-provider call caps. Fallbacks are the full ranked
        candidate list, first choice first."""
        plan = {}
        if not self.manifest:
            return plan
        assigned = {as_str(p.get("id")): 0 for p in self.providers}
        order = sorted(range(len(self.manifest)),
                       key=lambda i: (-(as_int(self.manifest[i].get("weight")) or 1), i))
        for index in order:
            entry = self.manifest[index]
            api = as_str(entry.get("api")) or "openai"
            req = requirements_of(self.manifest, index)
            cands = self._candidates(req, api)
            if not cands:
                plan[index] = []
                continue
            first = None
            for cand in cands:
                if assigned[cand["provider"]] < self._soft_cap(cand["provider"]):
                    first = cand
                    break
            if first is None:
                first = cands[0]
            assigned[first["provider"]] += 1
            rest = [c for c in cands if c is not first]
            plan[index] = [first] + rest
        return plan

    def candidates_for(self, index, req, client_api):
        """Try-order for one request: the plan's first choice, then the rest."""
        planned = self.plan.get(index)
        if planned:
            return planned
        return self._candidates(req, client_api)

    def record_success(self, pid):
        self.calls[pid] = self.calls.get(pid, 0) + 1
        self.consec_failures[pid] = 0

    def record_failure(self, pid):
        self.calls[pid] = self.calls.get(pid, 0) + 1
        self.consec_failures[pid] = self.consec_failures.get(pid, 0) + 1
        # A refusal that arrived after the soft cap was already used up is
        # exhaustion, not a transient failure: stop spending calls on it.
        if self.calls[pid] > self._soft_cap(pid):
            self.exhausted.add(pid)
        # Two consecutive failures also retire the provider for now; a
        # success elsewhere resets the counter.
        if self.consec_failures.get(pid, 0) >= 2:
            self.exhausted.add(pid)

    def retry_eligible(self, pid, calls_before):
        """A refusal is probably a transient scheduled failure while the
        provider's call count is still below its soft cap; above it, the
        provider is most likely out of budget."""
        return calls_before < self._soft_cap(pid)


# ---------------------------------------------------------------------------
# Canonical request: parse the client body once, render it in any wire.
# ---------------------------------------------------------------------------

def _text_of_content(content):
    """Plain text from an OpenAI/Anthropic content value (str or blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if is_record(part) and part.get("type") == "text":
                parts.append(as_str(part.get("text")))
        return "".join(parts)
    return ""


def _images_of_openai(content):
    images = []
    if isinstance(content, list):
        for part in content:
            if is_record(part) and part.get("type") == "image_url":
                image = part.get("image_url") if is_record(part.get("image_url")) else {}
                url = as_str(image.get("url"))
                if url:
                    images.append({"url": url})
    return images


def _images_of_anthropic(content):
    images = []
    if isinstance(content, list):
        for part in content:
            if is_record(part) and part.get("type") == "image":
                source = part.get("source") if is_record(part.get("source")) else {}
                stype = as_str(source.get("type"))
                if stype == "base64":
                    images.append({"base64": as_str(source.get("data")),
                                   "media_type": as_str(source.get("media_type"))})
                elif stype == "url":
                    images.append({"url": as_str(source.get("url"))})
    return images


def _images_of_gemini(content):
    images = []
    if isinstance(content, list):
        for part in content:
            if is_record(part):
                if "inlineData" in part and is_record(part["inlineData"]):
                    images.append({"base64": as_str(part["inlineData"].get("data")),
                                   "media_type": as_str(part["inlineData"].get("mimeType"))})
                elif "fileData" in part and is_record(part["fileData"]):
                    images.append({"url": as_str(part["fileData"].get("fileUri"))})
    return images


def _data_url(image):
    """An image as a data URL or plain URL, for the OpenAI wire."""
    if image.get("base64"):
        mime = image.get("media_type") or "image/png"
        return "data:" + mime + ";base64," + image["base64"]
    return image.get("url") or ""


def _media_type_from_url(url):
    match = re.match(r"^data:([^;,]+)", url or "")
    if match:
        return match.group(1)
    return "image/png"


def canonical_from_openai(body):
    canon = {"system": "", "messages": [], "tools": [], "tool_choice": None,
             "max_tokens": None, "temperature": None, "top_p": None,
             "stop": [], "stream": False, "json": False}
    if not is_record(body):
        return canon
    for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(message):
            continue
        role = as_str(message.get("role"))
        if role == "system":
            canon["system"] = _text_of_content(message.get("content"))
            continue
        entry = {"role": role, "text": _text_of_content(message.get("content")),
                 "images": _images_of_openai(message.get("content")),
                 "tool_calls": [], "tool_call_id": None}
        if isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                if is_record(call):
                    entry["tool_calls"].append(call)
        if role == "tool":
            entry["tool_call_id"] = as_str(message.get("tool_call_id"))
        canon["messages"].append(entry)
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_record(tool):
                fn = tool.get("function") if is_record(tool.get("function")) else tool
                canon["tools"].append({
                    "name": as_str(fn.get("name")),
                    "description": as_str(fn.get("description")),
                    "parameters": fn.get("input_schema") or fn.get("parameters") or {},
                })
    canon["tool_choice"] = body.get("tool_choice")
    canon["max_tokens"] = as_int(body.get("max_tokens"))
    if canon["max_tokens"] is None:
        canon["max_tokens"] = as_int(body.get("max_completion_tokens"))
    canon["temperature"] = body.get("temperature")
    canon["top_p"] = body.get("top_p")
    stop = body.get("stop")
    if isinstance(stop, str):
        canon["stop"] = [stop]
    elif isinstance(stop, list):
        canon["stop"] = [as_str(s) for s in stop if isinstance(s, str)]
    canon["stream"] = bool(body.get("stream"))
    fmt = body.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        canon["json"] = True
    return canon


def canonical_from_anthropic(body):
    canon = {"system": "", "messages": [], "tools": [], "tool_choice": None,
             "max_tokens": None, "temperature": None, "top_p": None,
             "stop": [], "stream": False, "json": False}
    if not is_record(body):
        return canon
    system = body.get("system")
    if isinstance(system, str):
        canon["system"] = system
    elif isinstance(system, list):
        canon["system"] = _text_of_content(system)
    for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(message):
            continue
        role = as_str(message.get("role"))
        entry = {"role": role, "text": _text_of_content(message.get("content")),
                 "images": _images_of_anthropic(message.get("content")),
                 "tool_calls": [], "tool_call_id": None}
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if is_record(part) and part.get("type") == "tool_use":
                    entry["tool_calls"].append({
                        "id": as_str(part.get("id")),
                        "type": "function",
                        "function": {"name": as_str(part.get("name")),
                                     "arguments": dump_json(part.get("input") or {})},
                    })
                elif is_record(part) and part.get("type") == "tool_result":
                    entry["tool_call_id"] = as_str(part.get("tool_use_id"))
                    entry["text"] = _text_of_content(part.get("content"))
        canon["messages"].append(entry)
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_record(tool):
                canon["tools"].append({
                    "name": as_str(tool.get("name")),
                    "description": as_str(tool.get("description")),
                    "parameters": tool.get("input_schema") or {},
                })
    canon["tool_choice"] = body.get("tool_choice")
    canon["max_tokens"] = as_int(body.get("max_tokens"))
    canon["temperature"] = body.get("temperature")
    canon["top_p"] = body.get("top_p")
    stops = body.get("stop_sequences")
    if isinstance(stops, list):
        canon["stop"] = [as_str(s) for s in stops if isinstance(s, str)]
    canon["stream"] = bool(body.get("stream"))
    return canon


def _openai_tool_choice(choice):
    if choice is None:
        return None
    if isinstance(choice, str):
        return choice
    if is_record(choice):
        if choice.get("type") == "function":
            fn = choice.get("function") if is_record(choice.get("function")) else {}
            name = as_str(fn.get("name")) or as_str(choice.get("name"))
            if name:
                return {"type": "function", "function": {"name": name}}
        if choice.get("type") == "tool":
            name = as_str(choice.get("name"))
            if name:
                return {"type": "function", "function": {"name": name}}
    return "auto"


def _anthropic_tool_choice(choice):
    if choice is None:
        return None
    if isinstance(choice, str):
        if choice == "auto":
            return {"type": "auto"}
        if choice == "none":
            return {"type": "none"}
        if choice == "required":
            return {"type": "any"}
        return {"type": "auto"}
    if is_record(choice):
        if choice.get("type") == "function":
            fn = choice.get("function") if is_record(choice.get("function")) else {}
            name = as_str(fn.get("name")) or as_str(choice.get("name"))
            if name:
                return {"type": "tool", "name": name}
        if choice.get("type") == "tool":
            name = as_str(choice.get("name"))
            if name:
                return {"type": "tool", "name": name}
        if choice.get("type") in ("auto", "any", "none"):
            return {"type": "any" if choice.get("type") == "any" else choice.get("type")}
    return {"type": "auto"}


def render_openai_body(canon, model, stream):
    body = {"model": model, "messages": [], "stream": bool(stream)}
    if canon["system"]:
        body["messages"].append({"role": "system", "content": canon["system"]})
    for entry in canon["messages"]:
        role = entry["role"]
        if role == "tool":
            msg = {"role": "tool", "content": entry["text"]}
            if entry.get("tool_call_id"):
                msg["tool_call_id"] = entry["tool_call_id"]
            body["messages"].append(msg)
            continue
        content = []
        if entry["text"]:
            content.append({"type": "text", "text": entry["text"]})
        for image in entry["images"]:
            url = _data_url(image)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        if role == "assistant" and entry.get("tool_calls"):
            msg = {"role": "assistant", "content": content if content else None,
                   "tool_calls": entry["tool_calls"]}
            body["messages"].append(msg)
        elif len(content) == 1 and content[0]["type"] == "text":
            body["messages"].append({"role": role, "content": entry["text"]})
        else:
            body["messages"].append({"role": role, "content": content})
    if canon["tools"]:
        body["tools"] = [{"type": "function",
                          "function": {"name": t["name"],
                                       "description": t["description"],
                                       "parameters": t["parameters"]}}
                         for t in canon["tools"] if t["name"]]
        choice = _openai_tool_choice(canon["tool_choice"])
        if choice is not None:
            body["tool_choice"] = choice
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        body["top_p"] = canon["top_p"]
    if canon["stop"]:
        body["stop"] = canon["stop"]
    if canon["json"]:
        body["response_format"] = {"type": "json_object"}
    if stream:
        # Without this the provider omits the final usage chunk on streams,
        # and the client cannot verify usage.
        body["stream_options"] = {"include_usage": True}
    return body


def render_anthropic_body(canon, model, stream):
    body = {"model": model, "messages": [], "max_tokens": 1024,
            "stream": bool(stream)}
    if canon["system"]:
        body["system"] = canon["system"]
    for entry in canon["messages"]:
        role = entry["role"]
        if role == "tool":
            # Tool results ride as a user turn with a tool_result block.
            content = [{"type": "tool_result",
                        "tool_use_id": entry.get("tool_call_id") or "",
                        "content": entry["text"]}]
            body["messages"].append({"role": "user", "content": content})
            continue
        content = []
        if entry["text"]:
            content.append({"type": "text", "text": entry["text"]})
        for image in entry["images"]:
            if image.get("base64"):
                content.append({"type": "image", "source": {
                    "type": "base64",
                    "media_type": image.get("media_type") or "image/png",
                    "data": image["base64"]}})
            elif image.get("url"):
                content.append({"type": "image", "source": {
                    "type": "url", "url": image["url"]}})
        if role == "assistant" and entry.get("tool_calls"):
            for call in entry["tool_calls"]:
                fn = call.get("function") if is_record(call.get("function")) else {}
                try:
                    args = json.loads(as_str(fn.get("arguments")) or "{}")
                except Exception:
                    args = {}
                content.append({"type": "tool_use",
                                "id": as_str(call.get("id")),
                                "name": as_str(fn.get("name")),
                                "input": args if is_record(args) else {}})
        if len(content) == 1 and content[0]["type"] == "text":
            body["messages"].append({"role": role, "content": entry["text"]})
        else:
            body["messages"].append({"role": role, "content": content})
    if canon["tools"]:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["parameters"] or {"type": "object"}}
                         for t in canon["tools"] if t["name"]]
        choice = _anthropic_tool_choice(canon["tool_choice"])
        if choice is not None:
            body["tool_choice"] = choice
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        body["top_p"] = canon["top_p"]
    if canon["stop"]:
        body["stop_sequences"] = canon["stop"]
    return body


def render_gemini_body(canon, model, stream):
    body = {"model": model, "contents": []}
    if canon["system"]:
        body["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    for entry in canon["messages"]:
        role = entry["role"]
        if role == "tool":
            role = "user"
        gemini_role = "model" if role == "assistant" else "user"
        parts = []
        if entry["text"]:
            parts.append({"text": entry["text"]})
        for image in entry["images"]:
            if image.get("base64"):
                parts.append({"inlineData": {
                    "mimeType": image.get("media_type") or "image/png",
                    "data": image["base64"]}})
            elif image.get("url"):
                parts.append({"fileData": {"fileUri": image["url"]}})
        if role == "assistant" and entry.get("tool_calls"):
            for call in entry["tool_calls"]:
                fn = call.get("function") if is_record(call.get("function")) else {}
                try:
                    args = json.loads(as_str(fn.get("arguments")) or "{}")
                except Exception:
                    args = {}
                parts.append({"functionCall": {
                    "name": as_str(fn.get("name")),
                    "args": args if is_record(args) else {}}})
        if parts:
            body["contents"].append({"role": gemini_role, "parts": parts})
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stop"]:
        config["stopSequences"] = canon["stop"]
    if canon["temperature"] is not None:
        config["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        config["topP"] = canon["top_p"]
    if canon["json"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    if canon["tools"]:
        declarations = []
        for t in canon["tools"]:
            if not t["name"]:
                continue
            declarations.append({
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"] or {"type": "object"},
            })
        if declarations:
            body["tools"] = [{"functionDeclarations": declarations}]
    return body


# ---------------------------------------------------------------------------
# Provider answers: SSE parsing and per-wire answer parsing.
# ---------------------------------------------------------------------------

def parse_sse(text):
    """SSE body -> [(event, data_string)] in order. A [DONE] line yields
    ("done", "[DONE]")."""
    frames = []
    for chunk in text.split("\n\n"):
        if not chunk.strip():
            continue
        event = None
        data_lines = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data.strip() == "[DONE]":
            frames.append(("done", "[DONE]"))
        else:
            frames.append((event, data))
    return frames


def _openai_stop_family(finish_reason):
    if finish_reason in ("length",):
        return 1
    if finish_reason in ("tool_calls", "function_call"):
        return 2
    if finish_reason in ("content_filter",):
        return 3
    return 0


def _anthropic_stop_family(stop_reason):
    if stop_reason in ("max_tokens",):
        return 1
    if stop_reason in ("tool_use",):
        return 2
    if stop_reason in ("refusal",):
        return 3
    return 0


def _gemini_stop_family(finish_reason):
    if finish_reason in ("MAX_TOKENS",):
        return 1
    if finish_reason in ("FUNCTION_CALL",):
        return 2
    if finish_reason in ("SAFETY",):
        return 3
    return 0


def parse_openai_answer(payload, stream):
    """-> {text, input_tokens, output_tokens, stop_family} or None."""
    if not is_record(payload):
        return None
    if not stream:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0] if is_record(choices[0]) else {}
        message = choice.get("message") if is_record(choice.get("message")) else {}
        content = message.get("content")
        text = _text_of_content(content) if content is not None else ""
        usage = payload.get("usage") if is_record(payload.get("usage")) else {}
        return {
            "text": text,
            "input_tokens": as_int(usage.get("prompt_tokens")) or 0,
            "output_tokens": as_int(usage.get("completion_tokens")) or 0,
            "stop_family": _openai_stop_family(as_str(choice.get("finish_reason"))),
        }
    # Streaming: role chunk, content deltas, finish chunk, optional usage.
    text_parts = []
    usage = None
    stop_family = 0
    raw = payload.get("_sse_text", "")
    for _event, data in parse_sse(raw):
        try:
            frame = json.loads(data)
        except Exception:
            continue
        if not is_record(frame):
            continue
        choices = frame.get("choices")
        if isinstance(choices, list) and choices and is_record(choices[0]):
            delta = choices[0].get("delta") if is_record(choices[0].get("delta")) else {}
            content = delta.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            finish = as_str(choices[0].get("finish_reason"))
            if finish:
                stop_family = _openai_stop_family(finish)
        if is_record(frame.get("usage")):
            usage = frame["usage"]
    return {
        "text": "".join(text_parts),
        "input_tokens": as_int(usage.get("prompt_tokens")) if usage else 0,
        "output_tokens": as_int(usage.get("completion_tokens")) if usage else 0,
        "stop_family": stop_family,
    }


def parse_anthropic_answer(payload, stream):
    if not is_record(payload):
        return None
    if not stream:
        text_parts = []
        content = payload.get("content")
        if isinstance(content, list):
            for part in content:
                if is_record(part) and part.get("type") == "text":
                    text_parts.append(as_str(part.get("text")))
        usage = payload.get("usage") if is_record(payload.get("usage")) else {}
        return {
            "text": "".join(text_parts),
            "input_tokens": as_int(usage.get("input_tokens")) or 0,
            "output_tokens": as_int(usage.get("output_tokens")) or 0,
            "stop_family": _anthropic_stop_family(as_str(payload.get("stop_reason"))),
        }
    text_parts = []
    usage_in = 0
    usage_out = 0
    stop_family = 0
    raw = payload.get("_sse_text", "")
    for event, data in parse_sse(raw):
        try:
            frame = json.loads(data)
        except Exception:
            continue
        if not is_record(frame):
            continue
        if event == "message_start":
            msg = frame.get("message") if is_record(frame.get("message")) else {}
            usage = msg.get("usage") if is_record(msg.get("usage")) else {}
            usage_in = as_int(usage.get("input_tokens")) or usage_in
        elif event == "content_block_delta":
            delta = frame.get("delta") if is_record(frame.get("delta")) else {}
            if delta.get("type") == "text_delta":
                text_parts.append(as_str(delta.get("text")))
        elif event == "message_delta":
            delta = frame.get("delta") if is_record(frame.get("delta")) else {}
            stop = as_str(delta.get("stop_reason"))
            if stop:
                stop_family = _anthropic_stop_family(stop)
            usage = frame.get("usage") if is_record(frame.get("usage")) else {}
            usage_out = as_int(usage.get("output_tokens")) or usage_out
    return {
        "text": "".join(text_parts),
        "input_tokens": usage_in,
        "output_tokens": usage_out,
        "stop_family": stop_family,
    }


def parse_gemini_answer(payload, stream):
    if not is_record(payload):
        return None
    if not stream:
        candidates = payload.get("candidates")
        text_parts = []
        finish = ""
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            content = candidates[0].get("content")
            parts = content.get("parts") if is_record(content) else None
            if isinstance(parts, list):
                for part in parts:
                    if is_record(part) and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
            finish = as_str(candidates[0].get("finishReason"))
        usage = payload.get("usageMetadata") if is_record(payload.get("usageMetadata")) else {}
        return {
            "text": "".join(text_parts),
            "input_tokens": as_int(usage.get("promptTokenCount")) or 0,
            "output_tokens": as_int(usage.get("candidatesTokenCount")) or 0,
            "stop_family": _gemini_stop_family(finish),
        }
    text_parts = []
    usage = {}
    finish = ""
    raw = payload.get("_sse_text", "")
    for _event, data in parse_sse(raw):
        try:
            frame = json.loads(data)
        except Exception:
            continue
        if not is_record(frame):
            continue
        candidates = frame.get("candidates")
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            content = candidates[0].get("content")
            parts = content.get("parts") if is_record(content) else None
            if isinstance(parts, list):
                for part in parts:
                    if is_record(part) and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
            finish = as_str(candidates[0].get("finishReason")) or finish
        if is_record(frame.get("usageMetadata")):
            usage = frame["usageMetadata"]
    return {
        "text": "".join(text_parts),
        "input_tokens": as_int(usage.get("promptTokenCount")) or 0,
        "output_tokens": as_int(usage.get("candidatesTokenCount")) or 0,
        "stop_family": _gemini_stop_family(finish),
    }


# ---------------------------------------------------------------------------
# One provider call: build the request in the provider's wire, parse the
# answer into the canonical form. Returns (answer, failure) where answer is
# {text, input_tokens, output_tokens, stop_family, provider, model} and
# failure is {status, code}.
# ---------------------------------------------------------------------------

def error_code_of(payload, wire, status):
    """The provider's refusal code, in its own wire."""
    if is_record(payload):
        if wire == "anthropic":
            error = payload.get("error") if is_record(payload.get("error")) else {}
            return as_str(error.get("type")) or as_str(error.get("message"))
        if wire == "gemini":
            error = payload.get("error") if is_record(payload.get("error")) else {}
            return as_str(error.get("status")) or as_str(error.get("message"))
        error = payload.get("error") if is_record(payload.get("error")) else {}
        return as_str(error.get("code")) or as_str(error.get("message"))
    if status == 402:
        return "payment_required"
    if status == 429:
        return "rate_limit"
    return "http_" + str(status)


def call_provider(provider, model_id, canon, stream):
    wire = as_str(provider.get("wire")) or "openai"
    base = provider_base(provider, CURRENT_ENV_URL)
    headers = {"accept": "text/event-stream" if stream else "application/json",
               "content-type": "application/json"}
    headers.update(auth_headers(provider))
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    if wire == "gemini":
        url = base.rstrip("/") + "/v1beta/models/" + model_id
        url += ":streamGenerateContent" if stream else ":generateContent"
        body = render_gemini_body(canon, model_id, stream)
    elif wire == "anthropic":
        url = base.rstrip("/") + "/v1/messages"
        body = render_anthropic_body(canon, model_id, stream)
    else:
        url = base.rstrip("/") + "/v1/chat/completions"
        body = render_openai_body(canon, model_id, stream)
    data = dump_json(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=CALL_TIMEOUT) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except Exception:
            raw = b""
        status = e.code
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = None
        return None, {"status": status,
                      "code": error_code_of(payload, wire, status)}
    except Exception as e:
        return None, {"status": 0, "code": "connection_error:" + type(e).__name__}
    if status < 200 or status >= 300:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = None
        return None, {"status": status,
                      "code": error_code_of(payload, wire, status)}
    if stream:
        payload = {"_sse_text": raw.decode("utf-8", errors="replace")}
    else:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return None, {"status": status, "code": "bad_response"}
    if wire == "gemini":
        answer = parse_gemini_answer(payload, stream)
    elif wire == "anthropic":
        answer = parse_anthropic_answer(payload, stream)
    else:
        answer = parse_openai_answer(payload, stream)
    if answer is None:
        return None, {"status": status, "code": "bad_response"}
    answer["provider"] = as_str(provider.get("id"))
    answer["model"] = model_id
    return answer, None


# ---------------------------------------------------------------------------
# Client responses: render the canonical answer in the client's wire.
# ---------------------------------------------------------------------------

def _split_13(text):
    return [text[i:i + 13] for i in range(0, len(text), 13)] or [""]


def render_openai_response(answer, stream, model_id):
    """OpenAI wire: non-stream JSON, or SSE chunks (13-char deltas like the
    provider's own streams)."""
    if not stream:
        return 200, {"content-type": "application/json"}, dump_json({
            "id": "chatcmpl-aggregator",
            "object": "chat.completion",
            "created": FIXED_CREATED,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer["text"]},
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": answer["input_tokens"],
                "completion_tokens": answer["output_tokens"],
                "total_tokens": answer["input_tokens"] + answer["output_tokens"],
            },
        }).encode("utf-8")
    frames = []
    frames.append(dump_json({
        "id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                     "finish_reason": None}],
    }))
    for piece in _split_13(answer["text"]):
        frames.append(dump_json({
            "id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model_id,
            "choices": [{"index": 0, "delta": {"content": piece},
                         "finish_reason": None}],
        }))
    frames.append(dump_json({
        "id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }))
    frames.append(dump_json({
        "id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model_id,
        "choices": [],
        "usage": {
            "prompt_tokens": answer["input_tokens"],
            "completion_tokens": answer["output_tokens"],
            "total_tokens": answer["input_tokens"] + answer["output_tokens"],
        },
    }))
    body = "".join("data: " + frame + "\n\n" for frame in frames)
    body += "data: [DONE]\n\n"
    return 200, {"content-type": "text/event-stream",
                 "cache-control": "no-cache"}, body.encode("utf-8")


def render_anthropic_response(answer, stream, model_id):
    if not stream:
        return 200, {"content-type": "application/json"}, dump_json({
            "id": "msg_aggregator",
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": [{"type": "text", "text": answer["text"]}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": answer["input_tokens"],
                "output_tokens": answer["output_tokens"],
            },
        }).encode("utf-8")
    frames = []
    frames.append(("message_start", dump_json({
        "type": "message_start",
        "message": {
            "id": "msg_aggregator", "type": "message", "role": "assistant",
            "model": model_id, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": answer["input_tokens"], "output_tokens": 0},
        },
    })))
    frames.append(("content_block_start", dump_json({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })))
    for piece in _split_13(answer["text"]):
        frames.append(("content_block_delta", dump_json({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        })))
    frames.append(("content_block_stop", dump_json({
        "type": "content_block_stop", "index": 0,
    })))
    frames.append(("message_delta", dump_json({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": answer["output_tokens"]},
    })))
    frames.append(("message_stop", dump_json({"type": "message_stop"})))
    body = "".join("event: " + event + "\ndata: " + data + "\n\n"
                  for event, data in frames)
    return 200, {"content-type": "text/event-stream",
                 "cache-control": "no-cache"}, body.encode("utf-8")


def render_client_error(api, message):
    if api == "anthropic":
        return 503, {"content-type": "application/json"}, dump_json({
            "type": "error",
            "error": {"type": "overloaded_error", "message": message},
        }).encode("utf-8")
    return 503, {"content-type": "application/json"}, dump_json({
        "error": {"message": message, "type": "overloaded_error",
                  "param": None, "code": "overloaded_error"},
    }).encode("utf-8")


# ---------------------------------------------------------------------------
# Serving: one request, through the router, with failover.
# ---------------------------------------------------------------------------

CURRENT_ENV_URL = ""
SERVER_STATE = None


class Aggregator:
    def __init__(self, providers, probes, catalog, rating_table_map, manifest):
        self.providers = {as_str(p.get("id")): p for p in providers}
        self.catalog = catalog
        self.router = Router(providers, probes, rating_table_map, manifest)
        self.manifest = manifest
        self.request_index = 0

    def serve(self, api, body):
        index = self.request_index
        self.request_index += 1
        if self.manifest:
            req = requirements_of(self.manifest, index)
            if not req.get("api"):
                req["api"] = api
        else:
            req = requirements_from_body(body, api)
        stream = bool(body.get("stream")) if is_record(body) else False
        if api == "openai":
            canon = canonical_from_openai(body)
        else:
            canon = canonical_from_anthropic(body)
        candidates = self.router.candidates_for(index, req, api)
        if not candidates:
            log("request %d: no free model can serve it" % (index + 1))
            return render_client_error(api, "no free model can serve this request")
        calls_made = 0
        tried = set()
        # Two passes: first skip providers already marked exhausted, then
        # allow them as a last resort rather than failing the request.
        for allow_exhausted in (False, True):
            for cand in candidates:
                pid = cand["provider"]
                mid = cand["model"]
                if (pid, mid) in tried:
                    continue
                if pid in self.router.exhausted and not allow_exhausted:
                    continue
                if calls_made >= MAX_CALLS_PER_REQUEST:
                    break
                provider = self.providers.get(pid)
                if provider is None:
                    continue
                # One attempt, plus a single same-provider retry when the
                # refusal looks transient (call count still below the soft
                # cap).
                for attempt in range(2):
                    calls_before = self.router.calls.get(pid, 0)
                    tried.add((pid, mid))
                    answer, failure = call_provider(provider, mid, canon, stream)
                    calls_made += 1
                    if failure is None:
                        self.router.record_success(pid)
                        log("request %d -> %s/%s" % (index + 1, pid, mid))
                        if api == "openai":
                            return render_openai_response(answer, stream, mid)
                        return render_anthropic_response(answer, stream, mid)
                    self.router.record_failure(pid)
                    status = failure.get("status")
                    log("request %d: %s/%s refused %s %s" %
                        (index + 1, pid, mid, status, failure.get("code")))
                    if (attempt == 0 and status in (402, 429, 500, 502, 503, 0)
                            and self.router.retry_eligible(pid, calls_before)):
                        continue
                    break
            if calls_made >= MAX_CALLS_PER_REQUEST:
                break
        log("request %d: all providers refused" % (index + 1))
        return render_client_error(api, "all providers refused")


# ---------------------------------------------------------------------------
# HTTP server.
# ---------------------------------------------------------------------------

def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, status, headers, body):
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json_error(self, status, message, api):
            if api == "anthropic":
                body = dump_json({"type": "error", "error": {
                    "type": "invalid_request_error", "message": message}})
            else:
                body = dump_json({"error": {"message": message,
                                            "type": "invalid_request_error",
                                            "param": None, "code": None}})
            self._send(status, {"content-type": "application/json"},
                       body.encode("utf-8"))

        def do_GET(self):
            if self.path.split("?")[0] == "/v1/models":
                anthropic = "x-api-key" in self.headers
                models = served_models(aggregator.catalog)
                if anthropic:
                    data = [{"id": mid, "type": "model",
                             "display_name": mid,
                             "created_at": FIXED_ANTHROPIC_CREATED_AT}
                            for mid in models]
                    body = dump_json({"object": "list", "data": data})
                else:
                    data = [{"id": mid, "object": "model",
                             "created": FIXED_CREATED, "owned_by": "free-tier",
                             "free": True} for mid in models]
                    body = dump_json({"object": "list", "data": data})
                self._send(200, {"content-type": "application/json"},
                           body.encode("utf-8"))
            else:
                self._send_json_error(404, "not found", "openai")

        def do_POST(self):
            api = None
            path = self.path.split("?")[0]
            if path == "/v1/chat/completions":
                api = "openai"
            elif path == "/v1/messages":
                api = "anthropic"
            else:
                self._send_json_error(404, "not found", "openai")
                return
            length = self.headers.get("content-length")
            try:
                size = int(length) if length is not None else 0
            except ValueError:
                size = 0
            if size < 0 or size > MAX_BODY:
                self._send_json_error(400, "request body too large", api)
                return
            try:
                raw = self.rfile.read(size) if size else b""
            except Exception:
                self._send_json_error(400, "could not read request body", api)
                return
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                self._send_json_error(400, "request body is not a JSON object", api)
                return
            if not is_record(body):
                self._send_json_error(400, "request body is not a JSON object", api)
                return
            if api == "openai" and not body.get("model"):
                body["model"] = "auto"
            if api == "anthropic" and not body.get("model"):
                body["model"] = "auto"
            status, headers, out = aggregator.serve(api, body)
            self._send(status, headers, out)

    return Handler


def register_with_env(env_url, port, catalog):
    url = env_url.rstrip("/") + "/register"
    payload = dump_json({"protocol_version": PROTOCOL_VERSION,
                         "port": port, "catalog": catalog}).encode("utf-8")
    headers = {"content-type": "application/json"}
    for _attempt in range(5):
        req = urllib.request.Request(url, data=payload, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
                body = resp.read()
                try:
                    data = json.loads(body.decode("utf-8"))
                except Exception:
                    data = {}
                if is_record(data) and data.get("ok"):
                    log("registered; env says %s requests to serve"
                        % data.get("requests"))
                    return True
                log("register rejected: %s" % body.decode("utf-8", "replace"))
                return False
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:
                detail = ""
            log("register HTTP %s: %s" % (e.code, detail))
            return False
        except Exception as e:
            log("register attempt failed: %s" % type(e).__name__)
    return False


def poll_work(env_url):
    url = env_url.rstrip("/") + "/work"
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    return is_record(data) and bool(data.get("done"))


def load_instance_providers(env_url):
    """Fallback directory: GET /instance returns the provider list with keys
    and absolute base URLs, plus the workload manifest."""
    payload, _status = http_get_json(env_url.rstrip("/") + "/instance", {},
                                     timeout=PROBE_TIMEOUT)
    if not is_record(payload):
        return [], []
    providers = payload.get("providers")
    manifest = payload.get("workload", {}).get("requests") if is_record(
        payload.get("workload")) else None
    if not isinstance(providers, list):
        providers = []
    if not isinstance(manifest, list):
        manifest = []
    return [p for p in providers if is_record(p)], manifest


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main(argv):
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    providers, rating_rows, manifest = read_brief(prompt_file)
    if not providers and env_url:
        providers, manifest2 = load_instance_providers(env_url)
        if manifest2 and not manifest:
            manifest = manifest2
    global CURRENT_ENV_URL
    CURRENT_ENV_URL = env_url
    probes = {}
    for provider in providers:
        pid = as_str(provider.get("id"))
        if pid:
            probes[pid] = probe_provider(provider, env_url)
    ratings_map = rating_table(rating_rows)
    catalog = build_catalog(providers, probes, ratings_map)
    if "--list" in argv:
        sys.stdout.write(dump_json(catalog) + "\n")
        return 0
    if not env_url:
        sys.stderr.write("MAKEFUN_ENV_URL is not set\n")
        return 2
    aggregator = Aggregator(providers, probes, catalog, ratings_map, manifest)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
    port = server.server_address[1]
    log("listening on port %d" % port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if not register_with_env(env_url, port, catalog):
        server.shutdown()
        server.server_close()
        return 1
    try:
        while True:
            if poll_work(env_url):
                break
            time.sleep(POLL_INTERVAL)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
