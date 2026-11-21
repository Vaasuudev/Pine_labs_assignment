"""Model-provider layer: a Provider interface, a deterministic mock, a real Gemini provider
(stdlib only) and a resilient client (retry w/ backoff, provider failover, usage/cost tracking).

Error taxonomy (this is what drives workflow behaviour):
  ProviderUnavailable  transient (timeouts, 429, 5xx): retried, then failed over, then the run PAUSES
  ProviderFatal        permanent (bad key, bad request, blocked prompt): never retried, run FAILS
  WorkflowPaused       every provider exhausted; state is persisted and the run can be resumed
"""
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import quote

PRICE_PER_1K = {"mock-small": (0.0008, 0.004), "mock-large": (0.003, 0.015)}
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"   # override with GEMINI_MODEL; unverified from the build sandbox


class ProviderUnavailable(Exception):
    """Transient provider failure (outage, 429, 5xx, timeout)."""

    def __init__(self, msg, retry_after=0.0):
        super().__init__(msg)
        self.retry_after = retry_after


class ProviderFatal(Exception):
    """Permanent provider failure: retrying cannot help (auth, bad request, blocked content)."""


class WorkflowPaused(Exception):
    """All providers exhausted; the run is persisted and can be resumed."""


def _tokens(text):
    return max(1, len(text) // 4)


class Provider:
    name = "base"
    model = "base"
    last_usage = None     # providers that know real token counts set {"tokens_in":..,"tokens_out":..}

    def complete(self, agent, system, payload):  # -> str (raw model text, expected JSON)
        raise NotImplementedError

    def price_per_1k(self):
        return PRICE_PER_1K.get(self.model, (0.0, 0.0))


class MockProvider(Provider):
    """Deterministic stand-in for an LLM. It is rule-based, but it *reads the real repo files* it
    is handed, so downstream checks run against real generated code. Faults are injected by name."""
    name = "mock"
    model = "mock-large"

    def __init__(self, behaviours, faults=None):
        self.behaviours = behaviours      # agent name -> fn(payload, call_no) -> dict
        self.faults = faults or {}        # e.g. {"outage_on": "developer", "outage_calls": 99}
        self.calls = {}
        self.outage_budget = dict(self.faults).get("outage_calls", 0)

    def complete(self, agent, system, payload):
        n = self.calls[agent] = self.calls.get(agent, 0) + 1
        if self.faults.get("outage_on") == agent and self.outage_budget > 0:
            self.outage_budget -= 1
            raise ProviderUnavailable(f"mock provider 503 while serving {agent}")
        if self.faults.get("malformed_on") == agent and n <= self.faults.get("malformed_calls", 1):
            return "Sure! Here is the result: {not json"
        if self.faults.get("unsupported_on") == agent and n <= 1:
            return json.dumps({"unexpected": "shape"})
        time.sleep(0.01)
        return json.dumps(self.behaviours[agent](payload, n))


class GeminiProvider(Provider):
    """Google Gemini over plain HTTPS (generateContent, JSON mode). Standard library only.

    The key is read from GEMINI_API_KEY and sent only in the x-goog-api-key header; it is
    redacted from every error message so it cannot leak into logs, state or the dashboard."""
    name = "gemini"

    def __init__(self, model=None, api_key=None, base_url=None, timeout=90, temperature=0.2):
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
        self.base = (base_url or os.environ.get("GEMINI_BASE_URL")
                     or "https://generativelanguage.googleapis.com").rstrip("/")
        self.timeout, self.temperature = timeout, temperature
        self._opener = urllib.request.build_opener(_NoRedirect())

    def price_per_1k(self):
        """Dollars per 1k tokens from GEMINI_PRICE_IN_PER_1M / GEMINI_PRICE_OUT_PER_1M (default 0:
        free tier / unpriced). Set them to see real cost estimates."""
        def f(name):
            try:
                return float(os.environ.get(name, "0")) / 1000.0
            except ValueError:
                return 0.0
        return f("GEMINI_PRICE_IN_PER_1M"), f("GEMINI_PRICE_OUT_PER_1M")

    def _redact(self, text):
        text = str(text)
        if self.key:
            text = text.replace(self.key, "[redacted]")
        return re.sub(r"AIza[\w-]{20,}", "[redacted]", text)

    def complete(self, agent, system, payload):
        if not self.key:
            raise ProviderFatal("GEMINI_API_KEY is not set. Set it in the terminal that starts this "
                                "program (see README), or use --provider mock.")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, default=str)}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "temperature": self.temperature, "maxOutputTokens": 8192},
        }
        url = f"{self.base}/v1beta/models/{quote(self.model, safe='')}:generateContent"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read(2_000_000))
        except urllib.error.HTTPError as e:
            detail = self._http_detail(e)
            if e.code in (408, 429) or e.code >= 500:
                ra = e.headers.get("Retry-After", "") if e.headers else ""
                raise ProviderUnavailable(f"Gemini HTTP {e.code}: {detail}",
                                          float(ra) if ra.replace(".", "", 1).isdigit() else 0.0) from None
            raise ProviderFatal(f"Gemini HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            raise ProviderUnavailable(f"Gemini transport error: {self._redact(type(e).__name__)}: "
                                      f"{self._redact(getattr(e, 'reason', e))}") from None
        except ValueError:
            raise ProviderUnavailable("Gemini returned a non-JSON HTTP body") from None
        return self._extract(data)

    def _http_detail(self, e):
        try:
            err = json.loads(e.read(32768))["error"]
            msg = " ".join(str(x) for x in (err.get("status"), err.get("message")) if x)
        except Exception:  # noqa: BLE001 - error bodies are untrusted and may not be JSON
            msg = ""
        return " ".join(self._redact(msg or "no readable error detail").split())[:500]

    def _extract(self, data):
        usage = data.get("usageMetadata") or {}
        self.last_usage = {"tokens_in": int(usage.get("promptTokenCount", 0)),
                           "tokens_out": int(usage.get("candidatesTokenCount", 0))
                                         + int(usage.get("thoughtsTokenCount", 0))}
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise ProviderFatal(f"Gemini blocked the prompt ({block})")
        try:
            cand = data["candidates"][0]
        except (KeyError, IndexError, TypeError):
            raise ProviderUnavailable("Gemini returned no candidates") from None
        reason = cand.get("finishReason", "STOP")
        if reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"):
            raise ProviderFatal(f"Gemini stopped the response ({reason})")
        parts = (cand.get("content") or {}).get("parts") or []
        # MAX_TOKENS yields truncated JSON: returned as-is so the schema validator triggers a repair
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):      # never follow redirects with an API key attached
        return None


class ResilientClient:
    def __init__(self, providers, store, run_id, retries=2, backoff=0.0):
        self.providers, self.store, self.run_id = providers, store, run_id
        self.retries, self.backoff = retries, backoff

    def call(self, agent, system, payload):
        last = None
        for p in self.providers:
            for i in range(self.retries + 1):
                t0 = time.perf_counter()
                p.last_usage = None
                try:
                    text = p.complete(agent, system, payload)
                except ProviderUnavailable as e:
                    last = e
                    self.store.event(self.run_id, "provider_error", f"{p.name}/{agent} try {i+1}: {e}")
                    if i < self.retries:
                        time.sleep(min(30.0, max(self.backoff * (2 ** i), e.retry_after)))
                    continue
                ms = (time.perf_counter() - t0) * 1000
                real = p.last_usage
                ti = real["tokens_in"] if real else _tokens(system + json.dumps(payload, default=str))
                to = real["tokens_out"] if real else _tokens(text)
                pin, pout = p.price_per_1k()
                usage = {"model": p.model, "provider": p.name, "tokens_in": ti, "tokens_out": to,
                         "cost_usd": ti / 1000 * pin + to / 1000 * pout, "latency_ms": ms}
                return text, usage
            self.store.event(self.run_id, "provider_failover", f"{p.name} exhausted")
        raise WorkflowPaused(f"all providers unavailable: {last}")
