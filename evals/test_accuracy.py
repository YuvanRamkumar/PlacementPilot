"""
Scoring accuracy eval: LLM label must agree with golden label on ≥ 80% of cases.

Strategy: for each golden label, call score_requirement() directly (bypassing
requirement extraction) so we test scoring logic in isolation.
"""
from __future__ import annotations

import pytest

from app.analyzer import score_requirement
from app.schemas import Requirement, RequirementType, ScoreLabel
from evals.conftest import requires_real_llm
from evals.runner import EvalResult, EvalRunner

THRESHOLD = 0.80
ADJACENT: dict[tuple, float] = {
    (ScoreLabel.met, ScoreLabel.partial): 0.5,
    (ScoreLabel.partial, ScoreLabel.met): 0.5,
    (ScoreLabel.partial, ScoreLabel.missing): 0.5,
    (ScoreLabel.missing, ScoreLabel.partial): 0.5,
}


def _agreement(predicted: str, golden: str) -> float:
    """1.0 for exact match, 0.5 for adjacent labels, 0.0 otherwise."""
    if predicted == golden:
        return 1.0
    return ADJACENT.get((ScoreLabel(predicted), ScoreLabel(golden)), 0.0)


@requires_real_llm
def test_scoring_accuracy(labels, resume_texts, jd_texts, llm):
    runner = EvalRunner("accuracy", threshold=THRESHOLD)

    for lbl in labels:
        resume_text = resume_texts.get(lbl["resume_id"], "")
        if not resume_text:
            continue

        req = Requirement(
            id=f"{lbl['jd_id']}_{lbl['requirement_text'][:20]}",
            text=lbl["requirement_text"],
            type=RequirementType(lbl["requirement_type"]),
        )

        try:
            scored = score_requirement(resume_text, req, llm)
            agreement = _agreement(scored.label.value, lbl["golden_label"])
            case_name = f"{lbl['resume_id']}|{lbl['jd_id']}|{lbl['requirement_text'][:40]}"
            runner.add(
                case_name,
                passed=agreement >= 1.0,
                score=agreement,
                detail=f"predicted={scored.label.value} golden={lbl['golden_label']}",
            )
        except Exception as exc:
            runner.add(
                f"{lbl['resume_id']}|{lbl['jd_id']}|{lbl['requirement_text'][:40]}",
                passed=False,
                score=0.0,
                detail=f"error: {exc}",
            )

    result = runner.finish()
    EvalRunner.write_report({"accuracy": result})

    assert result.score >= THRESHOLD, (
        f"Accuracy {result.score:.1%} below threshold {THRESHOLD:.1%}. "
        f"Passed {result.to_dict()['n_passed']}/{result.to_dict()['n_cases']} cases."
    )
