"""
Relevance eval: rewritten bullets must be relevant to the job description.
The judge model scores each bullet 0–10.
Pass threshold: average relevance ≥ 0.7.
"""
from __future__ import annotations

import json
import os
import re


os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from app.analyzer import rewrite_bullet
from evals.conftest import requires_real_llm
from evals.runner import EvalRunner

RELEVANCE_THRESHOLD = 0.7

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
def test_suggestion_relevance(resume_texts, jd_texts, llm, judge_llm):
    runner = EvalRunner("relevance", threshold=RELEVANCE_THRESHOLD)

    for resume_id, jd_id in _PAIRS:
        resume_text = resume_texts.get(resume_id, "")
        jd_text = jd_texts.get(jd_id, "")
        bullet = _BULLETS.get(resume_id, resume_text[:120])
        if not resume_text or not jd_text:
            continue

        try:
            rewritten = rewrite_bullet(resume_text, bullet, jd_text, llm)
            prompt = (
                f"Job description (excerpt):\n{jd_text[:2000]}\n\n"
                f"Rewritten resume bullet:\n{rewritten.bullet}\n\n"
                f"How relevant is this resume bullet to the job description? "
                f"Score 10 = highly relevant (directly addresses a required skill or responsibility). "
                f"Score 0 = completely irrelevant to the role."
            )
            score = _judge_score(judge_llm, prompt)
            runner.add(
                f"{resume_id}|{jd_id}",
                passed=score >= RELEVANCE_THRESHOLD,
                score=score,
                detail=f"bullet={rewritten.bullet[:80]}",
            )
        except Exception as exc:
            runner.add(f"{resume_id}|{jd_id}", passed=False, score=0.0, detail=str(exc))

    result = runner.finish()
    EvalRunner.write_report({"relevance": result})

    print(f"\nAverage relevance: {result.score:.3f}  (threshold ≥ {RELEVANCE_THRESHOLD})")
    assert result.score >= RELEVANCE_THRESHOLD, (
        f"Average relevance {result.score:.3f} below threshold {RELEVANCE_THRESHOLD}."
    )
