"""
Pydantic models for all LLM inputs/outputs and API request/response bodies.
All LLM responses are validated here; invalid JSON is caught before it propagates.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Requirement extraction ────────────────────────────────────────────────────

class RequirementType(str, Enum):
    must_have = "must_have"
    nice_to_have = "nice_to_have"
    responsibility = "responsibility"


class Requirement(BaseModel):
    id: str
    text: str
    type: RequirementType


class ExtractedRequirements(BaseModel):
    requirements: list[Requirement]


# ── Scoring ───────────────────────────────────────────────────────────────────

class ScoreLabel(str, Enum):
    met = "met"
    partial = "partial"
    missing = "missing"


class ScoredRequirement(BaseModel):
    requirement_id: str
    label: ScoreLabel
    evidence: Optional[str] = None  # verbatim quote from resume, or null
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("evidence")
    @classmethod
    def evidence_not_empty_string(cls, v: Optional[str]) -> Optional[str]:
        if v == "":
            return None
        return v


# ── Rewrite ───────────────────────────────────────────────────────────────────

class RewrittenBullet(BaseModel):
    original: str
    bullet: str


# ── Analysis result (full /analyze response) ─────────────────────────────────

class AnalysisResult(BaseModel):
    requirements: list[Requirement]
    scores: list[ScoredRequirement]
    match_pct: float = Field(ge=0.0, le=100.0)
    keyword_coverage_pct: float = Field(ge=0.0, le=100.0)
    suggestions: list[RewrittenBullet] = []


# ── API request bodies ────────────────────────────────────────────────────────

class AnalyzeTextRequest(BaseModel):
    resume_text: str = Field(min_length=10)
    jd_text: str = Field(min_length=10)


class RewriteRequest(BaseModel):
    resume_text: str = Field(min_length=10)
    bullet: str = Field(min_length=5)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]
    resume_text: Optional[str] = None
    jd_text: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
