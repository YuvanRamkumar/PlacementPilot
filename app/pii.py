"""
Redact PII (email, phone, URLs) from text before any LLM call.
Returns the redacted text and a restore map so the original can be recovered
for display — PII never leaves this process.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(\+?1[\s\-.]?)?"
    r"(\(?\d{3}\)?[\s\-.]?)"
    r"(\d{3}[\s\-.]?)"
    r"(\d{4})"
    r"(?!\d)"
)
_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+"
    r"|www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s\"'<>]*"
)


@dataclass
class RedactionResult:
    redacted_text: str
    restore_map: dict[str, str] = field(default_factory=dict)


def redact(text: str) -> RedactionResult:
    """Replace PII tokens with placeholders; return result + restore map."""
    restore: dict[str, str] = {}
    counter = [0]

    def _replace(match: re.Match, prefix: str) -> str:
        original = match.group(0)
        # Reuse the same placeholder if the same value appears twice.
        for placeholder, value in restore.items():
            if value == original:
                return placeholder
        counter[0] += 1
        placeholder = f"[{prefix}_{counter[0]}]"
        restore[placeholder] = original
        return placeholder

    out = _URL_RE.sub(lambda m: _replace(m, "URL"), text)
    out = _EMAIL_RE.sub(lambda m: _replace(m, "EMAIL"), out)
    out = _PHONE_RE.sub(lambda m: _replace(m, "PHONE"), out)

    return RedactionResult(redacted_text=out, restore_map=restore)


def restore(text: str, restore_map: dict[str, str]) -> str:
    """Reverse the redaction, putting original values back."""
    for placeholder, original in restore_map.items():
        text = text.replace(placeholder, original)
    return text
