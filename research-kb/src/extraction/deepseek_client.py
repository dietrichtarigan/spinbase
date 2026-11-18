"""DeepSeek API wrapper (OpenAI-compatible /chat/completions).

* **Privacy assertion guard**: every call passes the full serialized payload through
  :func:`src.core.privacy.assert_public` (taint flags + private-text fingerprints)
  *before* any socket is opened. Callers must also pass the source objects
  (``sources=[chunks...]``) so private-tagged items are rejected by flag.
* Text only. There is deliberately no way to attach files or images.
* Token & cost ledger in SQLite (``llm_usage``) plus a per-process budget stop.
* JSON mode with Pydantic validation and one automatic repair round-trip.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Iterable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config.settings import settings
from src.core.http import GuardedClient
from src.core.privacy import PrivacyViolation, assert_public

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, kb=None, model: str | None = None):
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (put it in research-kb/.env)")
        from src.core.db import get_kb
        self.kb = kb or get_kb()
        self.model = model or settings.deepseek_model
        self.http = GuardedClient(timeout=settings.deepseek_timeout_s, headers={
            "Authorization": f"Bearer {settings.deepseek_api_key.get_secret_value()}",
            "Content-Type": "application/json"})
        self.session_cost = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------- core call
    def chat(self, messages: list[dict], *, purpose: str, sources: Iterable[Any] = (),
             json_mode: bool = False, temperature: float = 0.0, max_tokens: int = 4096,
             model: str | None = None) -> str:
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise PrivacyViolation("Only plain-text message content is allowed (no files/images).")
        payload_text = "\n".join(m["content"] for m in messages)
        assert_public(items=list(sources), text=payload_text, kb=self.kb)  # <-- the air-gap
        if self.session_cost >= settings.deepseek_budget_usd:
            raise BudgetExceeded(f"Budget {settings.deepseek_budget_usd} USD reached")
        body = {"model": model or self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "stream": False}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self.http.post(f"{settings.deepseek_base_url.rstrip('/')}/chat/completions", json=body,
                           retries=settings.deepseek_max_retries)
        if r.status_code != 200:
            raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        self._account(j.get("usage") or {}, purpose, body["model"])
        return j["choices"][0]["message"]["content"] or ""

    def _account(self, u: dict, purpose: str, model: str) -> None:
        pt = int(u.get("prompt_tokens", 0))
        cached = int(u.get("prompt_cache_hit_tokens", 0) or (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        ct = int(u.get("completion_tokens", 0))
        cost = ((pt - cached) * settings.deepseek_price_input_per_m
                + cached * settings.deepseek_price_input_cached_per_m
                + ct * settings.deepseek_price_output_per_m) / 1e6
        with self._lock:
            self.session_cost += cost
        with self.kb.tx() as c:
            c.execute("INSERT INTO llm_usage (purpose, model, prompt_tokens, cached_tokens, completion_tokens, cost_usd)"
                      " VALUES (?,?,?,?,?,?)", (purpose, model, pt, cached, ct, cost))

    # ------------------------------------------------------------ JSON helpers
    @staticmethod
    def _parse_json(txt: str) -> Any:
        txt = txt.strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
        if m:
            txt = m.group(1)
        return json.loads(txt)

    def chat_json(self, messages: list[dict], *, purpose: str, sources: Iterable[Any] = (), **kw) -> Any:
        sources = list(sources)
        out = self.chat(messages, purpose=purpose, sources=sources, json_mode=True, **kw)
        try:
            return self._parse_json(out)
        except json.JSONDecodeError:
            fix = self.chat(messages + [{"role": "assistant", "content": out},
                                        {"role": "user", "content": "That was not valid JSON. Return ONLY the corrected JSON object."}],
                            purpose=purpose + ":repair", sources=sources, json_mode=True, **kw)
            return self._parse_json(fix)

    def structured(self, messages: list[dict], schema: Type[T], *, purpose: str,
                   sources: Iterable[Any] = (), **kw) -> T:
        sources = list(sources)
        data = self.chat_json(messages, purpose=purpose, sources=sources, **kw)
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            fix = self.chat_json(messages + [
                {"role": "assistant", "content": json.dumps(data)[:20000]},
                {"role": "user", "content": f"The JSON failed schema validation:\n{str(e)[:3000]}\n"
                                            "Return the corrected full JSON object only."}],
                purpose=purpose + ":repair", sources=sources, **kw)
            return schema.model_validate(fix)


_client: DeepSeekClient | None = None


def get_llm() -> DeepSeekClient:
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client
