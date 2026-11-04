"""Unit tests for app/analyzer.py: batching (speed), grounding and suggestion filtering."""
import json
import os
import threading

os.environ.setdefault("LLM_PROVIDER", "fake")

from app.analyzer import (  # noqa: E402
    SCORE_CHUNK_SIZE,
    analyze,
    gap_requirements,
    is_grounded,
    score_requirements,
    suggest_rewrites,
)
from app.schemas import Requirement, RequirementType, ScoreLabel  # noqa: E402

RESUME = (
    "Built a PySpark batch ETL pipeline ingesting 16.9M sensor readings.\n"
    "Designed and built the GitHub Actions CI/CD pipeline for cross-platform releases.\n"
    "Wrote SQL data quality checks to detect duplicate records and missing values."
)


def _reqs(n: int) -> list[Requirement]:
    return [Requirement(id=f"req_{i}", text=f"Skill {i}", type=RequirementType.must_have)
            for i in range(1, n + 1)]


class CountingLLM:
    """Records every call so tests can assert how many round trips the analyser makes."""

    def __init__(self, evidence: str = "GitHub Actions CI/CD pipeline") -> None:
        self.calls: list[list[dict]] = []
        self.evidence = evidence
        self._lock = threading.Lock()

    def chat(self, messages, json_mode=False, **_):
        with self._lock:
            self.calls.append(messages)
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "job-description parser" in system:
            return json.dumps({"requirements": [r.model_dump(mode="json") for r in _reqs(5)]})
        if "resume screener" in system:
            ids = [r["id"] for r in json.loads(user.split("Requirements:\n", 1)[1])]
            return json.dumps({"scores": [
                {"id": i, "label": "partial", "evidence": self.evidence, "confidence": 0.7}
                for i in ids
            ]})
        if '"suggestions"' in system:
            return json.dumps({"suggestions": [
                {"original": "Wrote SQL data quality checks to detect duplicate records",
                 "bullet": "Automated SQL data-quality tests for duplicates and nulls"},
                {"original": "Led a team of 40 engineers at Google",  # not in resume
                 "bullet": "Invented bullet"},
            ]})
        raise AssertionError(f"unexpected prompt: {system[:60]}")


# ── Speed: number of LLM calls ────────────────────────────────────────────────

def test_scoring_uses_one_call_for_a_normal_jd():
    llm = CountingLLM()
    scores = score_requirements(RESUME, _reqs(SCORE_CHUNK_SIZE), llm)
    assert len(llm.calls) == 1
    assert [s.requirement_id for s in scores] == [f"req_{i}" for i in range(1, 13)]


def test_long_jd_is_split_into_chunks():
    llm = CountingLLM()
    scores = score_requirements(RESUME, _reqs(SCORE_CHUNK_SIZE * 2 + 1), llm)
    assert len(llm.calls) == 3
    assert len(scores) == SCORE_CHUNK_SIZE * 2 + 1  # order and count preserved


def test_full_analysis_without_suggestions_makes_two_calls():
    llm = CountingLLM()
    result = analyze(RESUME, "We need Python, SQL and CI/CD experience.", llm,
                     include_suggestions=False)
    assert len(llm.calls) == 2  # extract + one batched scoring call
    assert result.suggestions == []
    assert {"extract_ms", "score_ms", "total_ms"} <= result.timings_ms.keys()


def test_full_analysis_with_suggestions_makes_three_calls():
    llm = CountingLLM()
    analyze(RESUME, "We need Python, SQL and CI/CD experience.", llm)
    assert len(llm.calls) == 3


# ── Grounding ─────────────────────────────────────────────────────────────────

def test_is_grounded_accepts_verbatim_and_whitespace_changes():
    assert is_grounded("GitHub Actions  CI/CD\npipeline", RESUME)


def test_is_grounded_rejects_invented_quote():
    assert not is_grounded("Managed Kubernetes clusters on AWS", RESUME)
    assert not is_grounded(None, RESUME)


def test_ungrounded_evidence_is_dropped():
    llm = CountingLLM(evidence="10 years of Appium experience")
    scores = score_requirements(RESUME, _reqs(2), llm)
    assert all(s.evidence is None for s in scores)


def test_grounded_evidence_is_kept():
    scores = score_requirements(RESUME, _reqs(1), CountingLLM())
    assert scores[0].evidence == "GitHub Actions CI/CD pipeline"


# ── Suggestions ───────────────────────────────────────────────────────────────

def test_suggestions_only_rewrite_bullets_that_exist():
    out = suggest_rewrites(RESUME, _reqs(2), CountingLLM())
    assert len(out) == 1
    assert out[0].original.startswith("Wrote SQL data quality checks")


def test_no_gaps_means_no_suggestion_call():
    llm = CountingLLM()
    assert suggest_rewrites(RESUME, [], llm) == []
    assert llm.calls == []


def test_gap_requirements_puts_must_haves_first():
    reqs = [
        Requirement(id="a", text="Nice", type=RequirementType.nice_to_have),
        Requirement(id="b", text="Must", type=RequirementType.must_have),
        Requirement(id="c", text="Met", type=RequirementType.must_have),
    ]
    scores = score_requirements(RESUME, reqs, CountingLLM())
    scores[2].label = ScoreLabel.met
    assert [r.id for r in gap_requirements(reqs, scores)] == ["b", "a"]
