"""
OpenAI-compatible LLM client with disk cache, retry on 429, and a FakeLLM for tests.
Swap provider entirely via .env (LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

CACHE_DIR = Path(".cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_PROVIDER = os.getenv("LLM_PROVIDER", "groq")


def _cache_key(model: str, messages: list[dict]) -> str:
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["content"]
    return None


def _save_cache(key: str, content: str) -> None:
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({"content": content}))


class FakeLLM:
    """Deterministic stub; used when LLM_PROVIDER=fake so tests never hit the network."""

    def chat(self, messages: list[dict], model: str = "fake", **_: Any) -> str:
        # Return minimal valid JSON that satisfies every schema the app expects.
        last = messages[-1]["content"] if messages else ""
        if "extract" in last.lower() or "requirements" in last.lower():
            return json.dumps([
                {
                    "id": "req_1",
                    "text": "Python experience",
                    "type": "must_have",
                }
            ])
        if "score" in last.lower():
            return json.dumps({
                "label": "met",
                "evidence": "Python experience",
                "confidence": 0.9,
            })
        if "rewrite" in last.lower():
            return json.dumps({"bullet": "Developed Python pipelines processing 1M rows/day."})
        # Default: echo a safe placeholder for chat
        return json.dumps({"reply": "I can help you improve your resume."})


class LLMClient:
    """Thin wrapper around the OpenAI SDK with caching and retry."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", ""),
        )

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        m = model or self.model
        key = _cache_key(m, messages)
        cached = _load_cache(key)
        if cached is not None:
            return cached

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=m,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    **kwargs,
                )
                content = resp.choices[0].message.content or ""
                _save_cache(key, content)
                return content
            except RateLimitError as exc:
                last_err = exc
                wait = self.backoff_base ** attempt
                time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_err


def get_llm() -> LLMClient | FakeLLM:
    """Factory — returns FakeLLM when LLM_PROVIDER=fake, real client otherwise."""
    if _PROVIDER == "fake":
        return FakeLLM()
    return LLMClient()
