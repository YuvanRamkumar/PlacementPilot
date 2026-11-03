"""
Hallucination eval: rewritten bullets must not introduce facts absent from the resume.
The judge model (large LLM) scores each bullet 0–10; ≥5 = no hallucination.
Pass threshold: hallucination rate ≤ 5%.
"""
from __future__ import annotations

import json
import os
import re


os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from app.analyzer import rewrite_bullet
from evals.conftest import requires_real_llm
from evals.runner import EvalRunner

HALLUCINATION_THRESHOLD = 0.05  # ≤ 5% hallucination rate

_PAIRS = [
    ("data_engineer", "data_engineer"),
    ("ml_student", "ai_quality_engineer"),
    ("web_developer", "backend_developer"),
]

_BULLETS = {
    "data_engineer": "Built ETL pipelines in Python and Apache Spark processing 2 TB/day across 15 sources",
    "ml_student": "Built a prototype tool to assess LLM outputs using evaluation metrics",
    "web_developer": "Built and maintained REST APIs using Node.js and TypeScript serving 50,000+ monthly active users",
}


def _judge_score(judge_llm, prompt: str) -> float:
    """Call the judge LLM; parse a 0-10 integer and return as 0.0–1.0. Returns 0.0 on failure."""
    full_prompt = (
        prompt
        + '\n\nRespond with JSON only, no other text: {"score": <integer 0-10>, "reason": "<one sentence>"}'
    )
    try:
        response = judge_llm.chat([{"role": "user", "content": full_prompt}])
        text = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
        try:
            return float(json.loads(text)["score"]) / 10.0
        except Exception:
            m = re.search(r'"score"\s*:\s*(10|[0-9])', text)
            if m:
                return float(m.group(1)) / 10.0
    except Exception:
        pass
    return 0.0


@requires_real_llm
def test_hallucination_rate(resume_texts, jd_texts, llm, judge_llm):
    runner = EvalRunner("hallucination", threshold=1.0 - HALLUCINATION_THRESHOLD)

    for resume_id, jd_id in _PAIRS:
        resume_text = resume_texts.get(resume_id, "")
        jd_text = jd_texts.get(jd_id, "")
        bullet = _BULLETS.get(resume_id, resume_text[:120])
        if not resume_text or not jd_text:
            continue

        try:
            rewritten = rewrite_bullet(resume_text, bullet, jd_text, llm)
            prompt = (
                f"Resume (ground truth):\n{resume_text[:3000]}\n\n"
                f"Rewritten resume bullet:\n{rewritten.bullet}\n\n"
                f"Does the rewritten bullet contain any fact, skill, number, or claim "
                f"that is NOT present in the resume? "
                f"Score 10 = no hallucination (every claim is in the resume). "
                f"Score 0 = clear hallucination (invented facts not in the resume)."
            )
            score = _judge_score(judge_llm, prompt)
            runner.add(
                f"{resume_id}|{jd_id}",
                passed=score >= 0.5,
                score=score,
                detail=f"bullet={rewritten.bullet[:80]}",
            )
        except Exception as exc:
            runner.add(f"{resume_id}|{jd_id}", passed=False, score=0.0, detail=str(exc))

    result = runner.finish()
    EvalRunner.write_report({"hallucination": result})

    hallucination_rate = 1.0 - result.score
    print(f"\nHallucination rate: {hallucination_rate:.1%}  (threshold ≤ {HALLUCINATION_THRESHOLD:.0%})")
    assert hallucination_rate <= HALLUCINATION_THRESHOLD, (
        f"Hallucination rate {hallucination_rate:.1%} exceeds {HALLUCINATION_THRESHOLD:.1%} threshold."
    )
