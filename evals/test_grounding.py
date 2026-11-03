"""
Evidence grounding eval: for every LLM-produced evidence quote,
verify it can be found verbatim (or near-verbatim) in the resume.

This is a pure code check — no LLM judge needed.
Pass threshold: 100% of evidence strings score ≥ 0.90 fuzzy partial ratio.
"""
from __future__ import annotations

import unicodedata

from rapidfuzz import fuzz

from app.analyzer import score_requirement
from app.schemas import Requirement, RequirementType, ScoreLabel
from evals.conftest import requires_real_llm
from evals.runner import EvalRunner

FUZZY_THRESHOLD = 90.0


def _normalise(text: str) -> str:
    """Collapse Unicode dash/quote variants to ASCII so fuzzy match isn't tripped up."""
    text = unicodedata.normalize("NFKD", text)
    # Replace common dash variants with ASCII hyphen.
    for ch in "–—―−":  # en-dash, em-dash, horizontal bar, minus
        text = text.replace(ch, "-")
    return text


@requires_real_llm
def test_evidence_grounding(labels, resume_texts, llm):
    runner = EvalRunner("grounding", threshold=1.0)

    for lbl in labels:
        resume_text = resume_texts.get(lbl["resume_id"], "")
        if not resume_text:
            continue

        req = Requirement(
            id=f"grounding_{lbl['requirement_text'][:20]}",
            text=lbl["requirement_text"],
            type=RequirementType(lbl["requirement_type"]),
        )

        try:
            scored = score_requirement(resume_text, req, llm)
        except Exception as exc:
            runner.add(
                f"{lbl['resume_id']}|{lbl['requirement_text'][:40]}",
                passed=False,
                score=0.0,
                detail=f"scoring error: {exc}",
            )
            continue

        # Only check evidence when the LLM says met/partial.
        if scored.label == ScoreLabel.missing or scored.evidence is None:
            runner.add(
                f"{lbl['resume_id']}|{lbl['requirement_text'][:40]}",
                passed=True,
                score=1.0,
                detail="no evidence expected (missing)",
            )
            continue

        ratio = fuzz.partial_ratio(_normalise(scored.evidence), _normalise(resume_text))
        passed = ratio >= FUZZY_THRESHOLD
        runner.add(
            f"{lbl['resume_id']}|{lbl['requirement_text'][:40]}",
            passed=passed,
            score=ratio / 100.0,
            detail=f"ratio={ratio:.1f} evidence={scored.evidence!r}",
        )

    result = runner.finish()
    EvalRunner.write_report({"grounding": result})

    failures = [c for c in result.cases if not c.passed]
    assert not failures, (
        f"Grounding failures ({len(failures)}):\n"
        + "\n".join(f"  {c.name}: {c.detail}" for c in failures)
    )
