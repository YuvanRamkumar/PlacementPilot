"""
Core analysis logic: extract requirements, score them, compute match %, rewrite bullets.
Each step calls the LLM with a structured prompt and validates the JSON response.
"""
from __future__ import annotations

import json
import re

from app.guardrails import scan_and_wrap
from app.llm import LLMClient, FakeLLM
from app.pii import redact
from app.schemas import (
    ExtractedRequirements,
    Requirement,
    RequirementType,
    ScoredRequirement,
    ScoreLabel,
    RewrittenBullet,
    AnalysisResult,
)

LLMType = LLMClient | FakeLLM

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def _parse_json(raw: str) -> dict | list:
    """Strip markdown fences if present, then parse JSON."""
    m = _JSON_FENCE.search(raw)
    if m:
        raw = m.group(1)
    return json.loads(raw.strip())


def _llm_json(llm: LLMType, messages: list[dict], retry: int = 1) -> dict | list:
    """Call LLM and parse JSON, retrying once on failure."""
    raw = llm.chat(messages)
    try:
        return _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        if retry > 0:
            # Ask the model to fix its output.
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your response was not valid JSON. Reply with only valid JSON, no prose."},
            ]
            raw2 = llm.chat(messages)
            return _parse_json(raw2)
        raise


# ── Step 1: Extract requirements ──────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a job-description parser. Extract every requirement from the job description.
Return ONLY a JSON array. Each element must have:
  "id": string (req_1, req_2, …),
  "text": string (the requirement, one sentence),
  "type": one of "must_have" | "nice_to_have" | "responsibility"
No other keys. No prose outside the JSON array."""


def extract_requirements(jd_text: str, llm: LLMType) -> list[Requirement]:
    redacted = redact(jd_text).redacted_text
    wrapped = scan_and_wrap(redacted, "jd")
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": wrapped},
    ]
    raw = _llm_json(llm, messages)
    parsed = ExtractedRequirements(requirements=raw)  # type: ignore[arg-type]
    return parsed.requirements


# ── Step 2: Score each requirement ───────────────────────────────────────────

_SCORE_SYSTEM = """\
You are a resume screener. Given a resume and one job requirement, determine if the
resume meets the requirement.
Return ONLY a JSON object with:
  "label": "met" | "partial" | "missing",
  "evidence": string (verbatim quote from the resume, max 120 chars) or null,
  "confidence": number 0.0–1.0
The evidence MUST be copied character-for-character from the resume text.
If label is "missing", evidence must be null."""


def score_requirement(
    resume_text: str,
    requirement: Requirement,
    llm: LLMType,
) -> ScoredRequirement:
    r_result = redact(resume_text)
    wrapped_resume = scan_and_wrap(r_result.redacted_text, "resume")
    messages = [
        {"role": "system", "content": _SCORE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{wrapped_resume}\n\n"
                f"Requirement ({requirement.type.value}): {requirement.text}"
            ),
        },
    ]
    raw = _llm_json(llm, messages)
    return ScoredRequirement(
        requirement_id=requirement.id,
        label=raw["label"],
        evidence=raw.get("evidence"),
        confidence=float(raw.get("confidence", 0.5)),
    )


# ── Step 3: Compute match % ───────────────────────────────────────────────────

def compute_match_pct(requirements: list[Requirement], scores: list[ScoredRequirement]) -> float:
    """(met*1 + partial*0.5) / must_have_count * 100. Returns 0 if no must-haves."""
    must_have_ids = {r.id for r in requirements if r.type == RequirementType.must_have}
    if not must_have_ids:
        return 0.0
    score_map = {s.requirement_id: s.label for s in scores}
    total = 0.0
    for rid in must_have_ids:
        label = score_map.get(rid, ScoreLabel.missing)
        if label == ScoreLabel.met:
            total += 1.0
        elif label == ScoreLabel.partial:
            total += 0.5
    return round(total / len(must_have_ids) * 100, 1)


def compute_keyword_coverage(resume_text: str, requirements: list[Requirement]) -> float:
    """Fraction of requirement keywords (lowercased tokens) present in the resume."""
    resume_lower = resume_text.lower()
    keywords = set()
    for req in requirements:
        for word in re.findall(r"[a-z]{3,}", req.text.lower()):
            keywords.add(word)
    if not keywords:
        return 0.0
    found = sum(1 for kw in keywords if kw in resume_lower)
    return round(found / len(keywords) * 100, 1)


# ── Step 4: Rewrite bullets ───────────────────────────────────────────────────

_REWRITE_SYSTEM = """\
You are a resume writing coach. Rewrite the given resume bullet to better match the
job description. Use ONLY facts present in the resume — do not invent skills,
numbers, or experiences. Make it concise, action-verb-led, and ATS-friendly.
Return ONLY a JSON object: {"bullet": "<rewritten bullet>"}"""


def rewrite_bullet(
    resume_text: str,
    bullet: str,
    jd_text: str,
    llm: LLMType,
) -> RewrittenBullet:
    r_result = redact(resume_text)
    wrapped_resume = scan_and_wrap(r_result.redacted_text, "resume")
    wrapped_jd = scan_and_wrap(redact(jd_text).redacted_text, "jd")
    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{wrapped_resume}\n\n{wrapped_jd}\n\n"
                f"Bullet to rewrite: {bullet}"
            ),
        },
    ]
    raw = _llm_json(llm, messages)
    return RewrittenBullet(original=bullet, bullet=raw["bullet"])


# ── Full analysis pipeline ────────────────────────────────────────────────────

def analyze(resume_text: str, jd_text: str, llm: LLMType) -> AnalysisResult:
    """Run all steps end-to-end and return a validated AnalysisResult."""
    requirements = extract_requirements(jd_text, llm)

    scores: list[ScoredRequirement] = []
    for req in requirements:
        scored = score_requirement(resume_text, req, llm)
        scores.append(scored)

    match_pct = compute_match_pct(requirements, scores)
    keyword_pct = compute_keyword_coverage(resume_text, requirements)

    # Generate rewrites for top missing/partial must-haves (up to 3).
    suggestions: list[RewrittenBullet] = []
    missing_ids = {
        s.requirement_id
        for s in scores
        if s.label in (ScoreLabel.missing, ScoreLabel.partial)
    }
    must_have_missing = [
        r for r in requirements
        if r.id in missing_ids and r.type == RequirementType.must_have
    ][:3]

    # Use first real sentence from the resume as a representative bullet.
    first_bullet = next(
        (line.strip() for line in resume_text.splitlines() if len(line.strip()) > 20),
        resume_text[:100],
    )
    for req in must_have_missing:
        try:
            rwt = rewrite_bullet(resume_text, first_bullet, req.text, llm)
            suggestions.append(rwt)
        except Exception:
            pass  # Don't let rewrite failure break the whole analysis.

    return AnalysisResult(
        requirements=requirements,
        scores=scores,
        match_pct=match_pct,
        keyword_coverage_pct=keyword_pct,
        suggestions=suggestions,
    )
