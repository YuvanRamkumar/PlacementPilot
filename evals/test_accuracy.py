"""
Scoring accuracy eval: LLM label must agree with golden label on ≥ 80% of cases.

Two scores are reported every run:
  strict  — only exact label matches count (1.0); any mismatch = 0.
  lenient — met↔partial confusion counts as half-correct (0.5); met↔missing = 0.

The build gate uses the strict score (threshold 80%).

Strategy: for each golden label, call score_requirement() directly (bypassing
requirement extraction) so we test scoring logic in isolation.
"""
from __future__ import annotations


from app.analyzer import score_requirement
from app.schemas import Requirement, RequirementType
from evals.conftest import requires_real_llm
from evals.runner import EvalRunner

THRESHOLD = 0.80


def _strict(predicted: str, golden: str) -> float:
    """1.0 for exact match, 0.0 for any mismatch."""
    return 1.0 if predicted == golden else 0.0


def _lenient(predicted: str, golden: str) -> float:
    """1.0 exact, 0.5 for met↔partial confusion, 0.0 for met↔missing or partial↔missing."""
    if predicted == golden:
        return 1.0
    if {predicted, golden} == {"met", "partial"}:
        return 0.5
    return 0.0


@requires_real_llm
def test_scoring_accuracy(labels, resume_texts, jd_texts, llm):
    runner = EvalRunner("accuracy", threshold=THRESHOLD)

    strict_scores: list[float] = []
    lenient_scores: list[float] = []

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
            s = _strict(scored.label.value, lbl["golden_label"])
            le = _lenient(scored.label.value, lbl["golden_label"])
            strict_scores.append(s)
            lenient_scores.append(le)
            case_name = f"{lbl['resume_id']}|{lbl['jd_id']}|{lbl['requirement_text'][:40]}"
            runner.add(
                case_name,
                passed=s >= 1.0,
                score=s,
                detail=(
                    f"predicted={scored.label.value} golden={lbl['golden_label']} "
                    f"strict={s:.1f} lenient={le:.1f}"
                ),
            )
        except Exception as exc:
            strict_scores.append(0.0)
            lenient_scores.append(0.0)
            runner.add(
                f"{lbl['resume_id']}|{lbl['jd_id']}|{lbl['requirement_text'][:40]}",
                passed=False,
                score=0.0,
                detail=f"error: {exc}",
            )

    n = len(strict_scores)
    strict_agg = sum(strict_scores) / n if n else 0.0
    lenient_agg = sum(lenient_scores) / n if n else 0.0

    result = runner.finish()
    EvalRunner.write_report({"accuracy": result})

    print(
        f"\nAccuracy — strict: {strict_agg:.1%}  lenient: {lenient_agg:.1%}"
        f"  (n={n}, threshold={THRESHOLD:.0%})"
    )

    assert strict_agg >= THRESHOLD, (
        f"Strict accuracy {strict_agg:.1%} below threshold {THRESHOLD:.1%}. "
        f"Lenient accuracy was {lenient_agg:.1%}."
    )
