"""Unit tests for app/llm.py — FakeLLM and cache logic."""
import json
import os
import shutil
import pytest

os.environ["LLM_PROVIDER"] = "fake"

from app.llm import FakeLLM, _cache_key, _load_cache, _save_cache, CACHE_DIR


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    """Redirect cache writes to a temp directory so tests stay isolated."""
    import app.llm as llm_mod
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path)
    yield tmp_path


def test_fake_llm_requirements():
    llm = FakeLLM()
    result = llm.chat([{"role": "user", "content": "extract requirements from the JD"}])
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert parsed[0]["id"] == "req_1"


def test_fake_llm_score():
    llm = FakeLLM()
    result = llm.chat([{"role": "user", "content": "score this requirement"}])
    parsed = json.loads(result)
    assert parsed["label"] in ("met", "partial", "missing")


def test_fake_llm_rewrite():
    llm = FakeLLM()
    result = llm.chat([{"role": "user", "content": "rewrite this bullet"}])
    parsed = json.loads(result)
    assert "bullet" in parsed


def test_fake_llm_chat_default():
    llm = FakeLLM()
    result = llm.chat([{"role": "user", "content": "What skills should I add?"}])
    parsed = json.loads(result)
    assert "reply" in parsed


def test_cache_key_is_deterministic():
    messages = [{"role": "user", "content": "hello"}]
    k1 = _cache_key("model-a", messages)
    k2 = _cache_key("model-a", messages)
    assert k1 == k2


def test_cache_key_differs_by_model():
    messages = [{"role": "user", "content": "hello"}]
    assert _cache_key("model-a", messages) != _cache_key("model-b", messages)


def test_cache_roundtrip(tmp_path):
    import app.llm as llm_mod
    # Temporarily point CACHE_DIR to tmp_path
    original = llm_mod.CACHE_DIR
    llm_mod.CACHE_DIR = tmp_path
    try:
        key = "testkey123"
        _save_cache(key, "cached response")
        assert _load_cache(key) == "cached response"
    finally:
        llm_mod.CACHE_DIR = original


def test_cache_miss_returns_none(tmp_path):
    import app.llm as llm_mod
    original = llm_mod.CACHE_DIR
    llm_mod.CACHE_DIR = tmp_path
    try:
        assert _load_cache("nonexistent_key") is None
    finally:
        llm_mod.CACHE_DIR = original
