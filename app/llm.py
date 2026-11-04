"""
OpenAI-compatible LLM client with disk cache, retry on 429, and a FakeLLM for tests.
Swap provider entirely via .env (LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI, RateLimitError

load_dotenv()

CACHE_DIR = Path(".cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_PROVIDER = os.getenv("LLM_PROVIDER", "groq")


def _cache_key(model: str, messages: list[dict], extra: dict | None = None) -> str:
    body: dict = {"model": model, "messages": messages}
    if extra:
        body["extra"] = extra  # different request options must not share a cache entry
    payload = json.dumps(body, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    return None


def _save_cache(key: str, content: str) -> None:
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({"content": content}), encoding="utf-8")


class FakeLLM:
    """Deterministic stub; used when LLM_PROVIDER=fake so tests never hit the network."""

    _IDS = re.compile(r'"id":\s*"([^"]+)"')

    def chat(self, messages: list[dict], model: str = "fake", json_mode: bool = False,
             **_: Any) -> str:
        # Return minimal valid JSON that satisfies every schema the app expects.
        # The system prompt drives dispatch; check the batched prompts first.
        all_content = " ".join(m.get("content", "") for m in messages)
        low = all_content.lower()
        if "screener" in low and '"scores"' in low:
            ids = self._IDS.findall(all_content) or ["req_1"]
            return json.dumps({"scores": [
                {"id": i, "label": "met", "evidence": "Python", "confidence": 0.9} for i in ids
            ]})
        if "writing coach" in low and '"suggestions"' in low:
            return json.dumps({"suggestions": []})
        if "extract" in low or "requirements" in low:
            reqs = [{"id": "req_1", "text": "Python experience", "type": "must_have"}]
            return json.dumps({"requirements": reqs} if json_mode else reqs)
        if "score" in low or "screener" in low:
            return json.dumps({"label": "met", "evidence": "Python experience", "confidence": 0.9})
        if "rewrite" in low or "writing coach" in low:
            return json.dumps({"bullet": "Developed Python pipelines processing 1M rows/day."})
        # Default: plain-text chat reply
        return "I can help you improve your resume."


class LLMClient:
    """Thin wrapper around the OpenAI SDK with caching, retry and speed options."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        timeout: float = 45.0,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        # gpt-oss models "think" before answering; low effort is much faster and
        # plenty for structured extraction/scoring. Override with LLM_REASONING_EFFORT.
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "low")
        self._client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", ""),
            timeout=timeout,
            max_retries=0,  # we handle retries ourselves
        )

    def _speed_options(self, model: str, json_mode: bool) -> dict:
        opts: dict = {}
        if json_mode:
            opts["response_format"] = {"type": "json_object"}
        if "gpt-oss" in model and self.reasoning_effort:
            opts["reasoning_effort"] = self.reasoning_effort
            opts["include_reasoning"] = False  # don't send the thinking text back
        return opts

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        m = model or self.model
        options = {**self._speed_options(m, json_mode), **kwargs}
        key = _cache_key(m, messages, options or None)
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
                    **options,
                )
                content = resp.choices[0].message.content or ""
                _save_cache(key, content)
                return content
            except (BadRequestError, TypeError) as exc:
                # A provider that doesn't support an option (e.g. Groq/Gemini and
                # reasoning_effort): drop the speed options and try again once.
                if options and options != kwargs:
                    options = dict(kwargs)
                    last_err = exc
                    continue
                raise
            except RateLimitError as exc:
                last_err = exc
                # Honour the suggested wait time from the error body, e.g.
                # "Please try again in 24.4425s".  Fall back to exponential backoff.
                m = re.search(r"try again in (\d+(?:\.\d+)?)s", str(exc))
                wait = float(m.group(1)) + 1.0 if m else self.backoff_base ** (attempt + 2)
                time.sleep(wait)
            except (APITimeoutError, APIConnectionError) as exc:
                last_err = exc
                time.sleep(self.backoff_base ** attempt)

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_err


def get_llm() -> LLMClient | FakeLLM:
    """Factory — returns FakeLLM when LLM_PROVIDER=fake, real client otherwise."""
    if _PROVIDER == "fake":
        return FakeLLM()
    return LLMClient()
