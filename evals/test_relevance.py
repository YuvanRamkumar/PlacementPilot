"""
Relevance eval: rewritten bullets must be relevant to the job description.
Uses DeepEval GEval with the judge model.
Pass threshold: average relevance score ≥ 0.7.
"""
from __future__ import annotations

import os
import pytest

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


def _make_deepeval_judge(judge_llm):
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError:
        pytest.skip("deepeval not installed")

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self, client):
            self._client = client

        def load_model(self):
            return self._client

        def generate(self, prompt: str, schema=None) -> str:
            return self._client.chat([{"role": "user", "content": prompt}])

        async def a_generate(self, prompt: str, schema=None) -> str:
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return self._client.model

    return GroqJudge(judge_llm)


@requires_real_llm
def test_suggestion_relevance(resume_texts, jd_texts, llm, judge_llm):
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError:
        pytest.skip("deepeval not installed")

    runner = EvalRunner("relevance", threshold=RELEVANCE_THRESHOLD)
    deepeval_judge = _make_deepeval_judge(judge_llm)

    relevance_metric = GEval(
        name="Relevance",
        criteria=(
            "The 'actual output' (rewritten resume bullet) should address skills, "
            "experience, or responsibilities described in the 'input' (job description). "
            "A score of 1 means the bullet is highly relevant to the JD. "
            "A score of 0 means it has no connection to the JD."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=deepeval_judge,
        threshold=RELEVANCE_THRESHOLD,
    )

    for resume_id, jd_id in _PAIRS:
        resume_text = resume_texts.get(resume_id, "")
        jd_text = jd_texts.get(jd_id, "")
        bullet = _BULLETS.get(resume_id, resume_text[:120])
        if not resume_text or not jd_text:
            continue

        try:
            rewritten = rewrite_bullet(resume_text, bullet, jd_text, llm)
            test_case = LLMTestCase(
                input=jd_text,
                actual_output=rewritten.bullet,
            )
            relevance_metric.measure(test_case)
            score = relevance_metric.score
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

    assert result.score >= RELEVANCE_THRESHOLD, (
        f"Average relevance {result.score:.3f} below threshold {RELEVANCE_THRESHOLD}."
    )
