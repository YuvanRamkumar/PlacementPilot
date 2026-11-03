"""
Guardrails: prompt-injection detection, document delimiting, off-topic refusal.

Strategy
--------
- Wrap every document in explicit XML-style delimiters so the model sees
  clear boundaries between instructions and user content.
- Scan for known injection patterns (ignore previous instructions, etc.)
  and flag them rather than silently passing them through.
- Reject chat messages that are clearly off-topic (harmful / unrelated).
"""
from __future__ import annotations

import re

# Patterns that strongly indicate prompt-injection attempts inside documents.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|results?|output)", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you|i)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(different|new)\s+(ai|assistant|model)", re.I),
    re.compile(r"score\s+(this\s+resume\s+)?100\s*%", re.I),
    re.compile(r"print\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?instructions?", re.I),
    re.compile(r"(do\s+)?jailbreak", re.I),
    # Hidden system-override markers sometimes embedded in documents.
    re.compile(r"\[SYSTEM\s*:", re.I),
    re.compile(r"set\s+(the\s+)?(final\s+)?match_pct\s+to\s+100", re.I),
]

# Very broad off-topic / harmful topics — only obvious cases.
_OFFTOPIC_PATTERNS = [
    re.compile(r"\bhow\s+(to|do\s+i|can\s+i)\s+(make|build|create)\s+(a\s+)?(bomb|weapon|malware|virus)\b", re.I),
    re.compile(r"\b(suicide|self[\s\-]harm)\b", re.I),
]

RESUME_DELIMITER = "<<RESUME>>"
RESUME_END = "<</RESUME>>"
JD_DELIMITER = "<<JOB_DESCRIPTION>>"
JD_END = "<</JOB_DESCRIPTION>>"


class InjectionDetectedError(ValueError):
    pass


class OffTopicError(ValueError):
    pass


def wrap_resume(text: str) -> str:
    return f"{RESUME_DELIMITER}\n{text}\n{RESUME_END}"


def wrap_jd(text: str) -> str:
    return f"{JD_DELIMITER}\n{text}\n{JD_END}"


def check_document(text: str, source: str = "document") -> None:
    """Raise InjectionDetectedError if injection patterns are found."""
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            raise InjectionDetectedError(
                f"Potential prompt-injection detected in {source}: matched {pat.pattern!r}"
            )


def check_chat_message(message: str) -> None:
    """Raise OffTopicError for clearly harmful / off-topic chat messages."""
    for pat in _OFFTOPIC_PATTERNS:
        if pat.search(message):
            raise OffTopicError("Message appears off-topic or harmful and cannot be processed.")
    # Also reject injection attempts in live chat.
    for pat in _INJECTION_PATTERNS:
        if pat.search(message):
            raise InjectionDetectedError("Injection attempt detected in chat message.")


def scan_and_wrap(text: str, kind: str) -> str:
    """
    Check a document for injection, then wrap it in safe delimiters.
    kind: 'resume' | 'jd'
    """
    check_document(text, source=kind)
    if kind == "resume":
        return wrap_resume(text)
    return wrap_jd(text)
