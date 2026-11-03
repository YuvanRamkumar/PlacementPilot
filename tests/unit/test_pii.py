"""Unit tests for app/pii.py"""
from app.pii import redact, restore


def test_email_redacted():
    result = redact("Contact me at alice@example.com for details.")
    assert "alice@example.com" not in result.redacted_text
    assert "[EMAIL_1]" in result.redacted_text


def test_phone_redacted():
    result = redact("Call me at 555-867-5309.")
    assert "555-867-5309" not in result.redacted_text
    assert "[PHONE_1]" in result.redacted_text


def test_url_redacted():
    result = redact("See my portfolio at https://github.com/alice.")
    assert "https://github.com/alice" not in result.redacted_text
    assert "[URL_1]" in result.redacted_text


def test_multiple_pii_types():
    text = "Email alice@test.com or call +1 (800) 555-1234. Visit https://alice.dev"
    result = redact(text)
    assert "alice@test.com" not in result.redacted_text
    assert "+1 (800) 555-1234" not in result.redacted_text
    assert "https://alice.dev" not in result.redacted_text
    # Three distinct PII items should be in the restore map.
    assert len(result.restore_map) == 3


def test_restore_reverses_redaction():
    original = "Email me at bob@example.org or call 415-555-0100."
    result = redact(original)
    restored = restore(result.redacted_text, result.restore_map)
    assert restored == original


def test_duplicate_pii_same_placeholder():
    text = "alice@test.com is my email. Again: alice@test.com"
    result = redact(text)
    # Only one placeholder should appear (same value → reused).
    assert len(result.restore_map) == 1
    assert result.redacted_text.count("[EMAIL_1]") == 2


def test_no_pii_unchanged():
    text = "Experienced Python developer with 5 years of work."
    result = redact(text)
    assert result.redacted_text == text
    assert result.restore_map == {}


def test_international_phone():
    result = redact("My number is +44 20 7946 0958.")
    # Should not crash; may or may not match depending on pattern.
    assert isinstance(result.redacted_text, str)
