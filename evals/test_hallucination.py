"""
Hallucination eval: rewritten bullets must not introduce facts absent from the resume.
Uses DeepEval GEval with the judge (large) model as the LLM.
Pass threshold: hallucination rate ≤ 5% (at most 1 in 20 bullets may hallucinate).
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from app.analyzer import rewrite_bullet
from evals.conftest import requires_real_llm
from evals.runner import EvalRunner

HALLUCINATION_THRESHOLD = 0.05  # ≤ 5% hallucination rate

# Resume+JD pairs to test rewriting on.
_PAIRS = [
    ("data_engineer", "data_engineer"),
    ("ml_student", "ai_quality_engineer"),
    ("web_developer", "backend_developer"),
]

# Representative bullets to rewrite — chosen as the first substantive line.
_BULLETS = {
    "data_engineer": "Built ETL pipelines in Python and Apache Spark processing 2 TB/day across 15 sources",
    "ml_student": "Built a prototype tool to assess LLM outputs using evaluation metrics",
    "web_developer": "Built and maintained REST APIs using Node.js and TypeScript serving 50,000+ monthly active users",
}


def _make_deepeval_judge(judge_llm):
    """Wrap our LLMClient as a DeepEvalBaseLLM."""
    import json
    import re

    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError:
        pytest.skip("deepeval not installed")

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self, client):
            self._client = client

        def load_model(self):
            return self._client

        def generate(self, prompt: str, schema=None):
            # When schema is provided, GEval expects a Pydantic instance back, not a string.
            # Append an explicit JSON-only instruction so Groq returns parseable output.
            actual_prompt = prompt if schema is None else prompt + "\n\nRespond with valid JSON only."
            response = self._client.chat([{"role": "user", "content": actual_prompt}])
            if schema is None:
                return response
            text = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.MULTILINE)
            text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
            try:
                return schema(**json.loads(text))
            except Exception:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    try:
                        return schema(**json.loads(match.group()))
                    except Exception:
                        pass
            return schema.model_construct()

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return self._client.model

    return GroqJudge(judge_llm)


@requires_real_llm
def test_hallucination_rate(resume_texts, jd_texts, llm, judge_llm):
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError:
        pytest.skip("deepeval not installed")

    runner = EvalRunner("hallucination", threshold=1.0 - HALLUCINATION_THRESHOLD)
    deepeval_judge = _make_deepeval_judge(judge_llm)

    hallucination_metric = GEval(
        name="NoHallucination",
        criteria=(
            "The 'actual output' (rewritten resume bullet) must only contain facts, "
            "skills, numbers, and claims that are explicitly present in the 'context' "
            "(the original resume). A score of 1 means no hallucination. "
            "A score of 0 means the output invented facts not in the resume."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
        model=deepeval_judge,
        threshold=0.5,
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
                input=bullet,
                actual_output=rewritten.bullet,
                context=[resume_text],
            )
            hallucination_metric.measure(test_case)
            score = hallucination_metric.score  # 1.0 = no hallucination
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
    assert hallucination_rate <= HALLUCINATION_THRESHOLD, (
        f"Hallucination rate {hallucination_rate:.1%} exceeds {HALLUCINATION_THRESHOLD:.1%} threshold."
    )
