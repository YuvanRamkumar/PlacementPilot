"""
FastAPI application.
Endpoints:
  GET  /health
  POST /analyze      — file upload (resume + JD) or JSON text
  POST /rewrite      — rewrite a single bullet
  POST /chat         — follow-up conversation
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles

from app.analyzer import analyze, rewrite_bullet, suggest_rewrites
from app.guardrails import InjectionDetectedError, OffTopicError, check_chat_message, scan_and_wrap
from app.llm import get_llm
from app.logging_ import log_analysis
from app.parsing import FileTooLargeError, UnsupportedFileTypeError, extract_text_from_bytes
from app.pii import redact
from app.schemas import (
    AnalysisResult,
    AnalyzeTextRequest,
    ChatRequest,
    ChatResponse,
    RewriteRequest,
    RewrittenBullet,
    ScoreLabel,
    SuggestRequest,
    SuggestResponse,
)

app = FastAPI(title="PlacementPilot", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": os.getenv("LLM_PROVIDER", "groq")}


# ── Analyze (file upload) ─────────────────────────────────────────────────────

@app.post("/analyze/upload", response_model=AnalysisResult)
async def analyze_upload(
    response: Response,
    resume: UploadFile = File(...),
    jd: UploadFile = File(...),
    include_suggestions: bool = Form(True),
) -> AnalysisResult:
    try:
        resume_text = extract_text_from_bytes(await resume.read(), resume.filename or "resume")
        jd_text = extract_text_from_bytes(await jd.read(), jd.filename or "jd")
    except (FileTooLargeError, UnsupportedFileTypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    result = _run_analysis(resume_text, jd_text, include_suggestions, response)
    # Hand the parsed text back so the page can use it for /chat and /suggest.
    result.resume_text = resume_text
    result.jd_text = jd_text
    return result


# ── Analyze (JSON text) ───────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalysisResult)
def analyze_text(req: AnalyzeTextRequest, response: Response) -> AnalysisResult:
    return _run_analysis(req.resume_text, req.jd_text, req.include_suggestions, response)


def _run_analysis(resume_text: str, jd_text: str, include_suggestions: bool = True,
                  response: Response | None = None) -> AnalysisResult:
    try:
        llm = get_llm()
        result = analyze(resume_text, jd_text, llm, include_suggestions=include_suggestions)
    except (InjectionDetectedError, OffTopicError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log_analysis(endpoint="/analyze", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    num_met = sum(1 for s in result.scores if s.label == ScoreLabel.met)
    num_partial = sum(1 for s in result.scores if s.label == ScoreLabel.partial)
    num_missing = sum(1 for s in result.scores if s.label == ScoreLabel.missing)

    log_analysis(
        endpoint="/analyze",
        match_pct=result.match_pct,
        keyword_coverage_pct=result.keyword_coverage_pct,
        num_requirements=len(result.requirements),
        num_met=num_met,
        num_partial=num_partial,
        num_missing=num_missing,
    )
    if response is not None:
        # Server-Timing shows each stage in the browser's DevTools Network tab.
        response.headers["Server-Timing"] = ", ".join(
            f"{name.removesuffix('_ms')};dur={ms}" for name, ms in result.timings_ms.items()
        )
    return result


# ── Suggest (fetched by the page after the scores are shown) ─────────────────

@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest) -> SuggestResponse:
    try:
        llm = get_llm()
        return SuggestResponse(suggestions=suggest_rewrites(req.resume_text, req.requirements, llm))
    except (InjectionDetectedError, OffTopicError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Suggestions failed. Try again.")


# ── Rewrite ───────────────────────────────────────────────────────────────────

@app.post("/rewrite", response_model=RewrittenBullet)
def rewrite(req: RewriteRequest) -> RewrittenBullet:
    try:
        llm = get_llm()
        return rewrite_bullet(req.resume_text, req.bullet, req.resume_text, llm)
    except (InjectionDetectedError, OffTopicError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Rewrite failed.")


# ── Chat ──────────────────────────────────────────────────────────────────────

_CHAT_SYSTEM = """\
You are a helpful career coach embedded in PlacementPilot.
Help the user improve their resume against the job description.
Stay focused on the resume/JD context provided.
If the user asks something unrelated to job searching or career advice, politely decline."""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        check_chat_message(req.message)
    except (InjectionDetectedError, OffTopicError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    messages: list[dict] = [{"role": "system", "content": _CHAT_SYSTEM}]

    # Inject context if provided.
    if req.resume_text:
        r = redact(req.resume_text)
        messages.append({
            "role": "user",
            "content": f"Here is the candidate's resume:\n{scan_and_wrap(r.redacted_text, 'resume')}",
        })
        messages.append({"role": "assistant", "content": "Understood, I have the resume."})
    if req.jd_text:
        r = redact(req.jd_text)
        messages.append({
            "role": "user",
            "content": f"Here is the job description:\n{scan_and_wrap(r.redacted_text, 'jd')}",
        })
        messages.append({"role": "assistant", "content": "Understood, I have the job description."})

    # Append conversation history (last 10 turns max).
    for turn in req.history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": req.message})

    try:
        llm = get_llm()
        raw = llm.chat(messages)
    except Exception:
        raise HTTPException(status_code=500, detail="Chat failed. Please try again.")

    # raw may be plain text or JSON {"reply": "..."}
    try:
        import json
        parsed = json.loads(raw)
        reply = parsed.get("reply", raw)
    except Exception:
        reply = raw

    log_analysis(endpoint="/chat")
    return ChatResponse(reply=reply)


# ── Static files ──────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
