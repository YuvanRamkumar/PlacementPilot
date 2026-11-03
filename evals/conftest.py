"""
Shared pytest fixtures for all eval suites.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Opt out of DeepEval cloud telemetry before any deepeval import.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

GOLDEN_DIR = Path(__file__).parent / "golden"

requires_real_llm = pytest.mark.skipif(
    os.getenv("LLM_PROVIDER") == "fake",
    reason="Eval suite requires a real LLM — set LLM_PROVIDER != fake",
)


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def labels() -> list[dict]:
    data = json.loads((GOLDEN_DIR / "labels_v1.json").read_text())
    return data["labels"]


@pytest.fixture(scope="session")
def resume_texts() -> dict[str, str]:
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in (GOLDEN_DIR / "resumes").glob("*.txt")
    }


@pytest.fixture(scope="session")
def jd_texts() -> dict[str, str]:
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in (GOLDEN_DIR / "jds").glob("*.txt")
    }


@pytest.fixture(scope="session")
def llm():
    from app.llm import get_llm
    return get_llm()


@pytest.fixture(scope="session")
def judge_llm():
    """Judge LLM (larger model) used in hallucination and relevance evals."""
    import os
    from app.llm import LLMClient
    return LLMClient(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL")),
    )
