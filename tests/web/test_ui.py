"""
Playwright UI tests — page-object pattern.
Covers: page load, tab switching, validation, analysis happy path, chat.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.web.pages.analyze_page import AnalyzePage
from tests.web.pages.chat_page import ChatPage

_RESUME = (
    "Experienced Python developer with 5 years building data pipelines and REST APIs. "
    "Proficient in Apache Spark, SQL, and cloud data engineering on AWS."
)
_JD = (
    "We are looking for a Python developer with 3+ years of experience in data engineering, "
    "Apache Spark, ETL pipelines, and SQL databases."
)


@pytest.fixture
def analyze_page(page: Page, base_url: str) -> AnalyzePage:
    ap = AnalyzePage(page, base_url)
    ap.navigate()
    return ap


# ── Page load ─────────────────────────────────────────────────────────────────

def test_page_title(page: Page, base_url: str):
    page.goto(base_url)
    expect(page).to_have_title("PlacementPilot")


def test_paste_tab_active_by_default(analyze_page: AnalyzePage):
    assert analyze_page.tab_is_active("tab-paste"), "Paste tab should be active on load"
    expect(analyze_page.page.locator("#mode-paste")).to_be_visible()
    expect(analyze_page.page.locator("#mode-file")).to_be_hidden()


def test_results_hidden_on_load(analyze_page: AnalyzePage):
    assert not analyze_page.is_results_visible, "Results section should be hidden before analysis"


# ── Tab switching ─────────────────────────────────────────────────────────────

def test_switch_to_file_tab(analyze_page: AnalyzePage):
    analyze_page.switch_to_file_tab()
    assert analyze_page.tab_is_active("tab-file")
    expect(analyze_page.page.locator("#mode-file")).to_be_visible()
    expect(analyze_page.page.locator("#mode-paste")).to_be_hidden()


def test_switch_back_to_paste_tab(analyze_page: AnalyzePage):
    analyze_page.switch_to_file_tab()
    analyze_page.switch_to_paste_tab()
    assert analyze_page.tab_is_active("tab-paste")
    expect(analyze_page.page.locator("#mode-paste")).to_be_visible()


# ── Client-side validation ────────────────────────────────────────────────────

def test_empty_submit_shows_error(analyze_page: AnalyzePage):
    analyze_page.click_analyze()
    expect(analyze_page.page.locator("#error-box")).to_be_visible()
    assert "Please" in analyze_page.error_text


def test_error_cleared_on_next_submit(analyze_page: AnalyzePage):
    # Trigger an error first.
    analyze_page.click_analyze()
    expect(analyze_page.page.locator("#error-box")).to_be_visible()
    # Now provide valid input and submit.
    analyze_page.paste_texts(_RESUME, _JD)
    analyze_page.click_analyze()
    analyze_page.wait_for_results()
    # Error box should be cleared.
    expect(analyze_page.page.locator("#error-box")).to_be_hidden()


# ── Analysis happy path ───────────────────────────────────────────────────────

def test_results_appear_after_analysis(analyze_page: AnalyzePage):
    analyze_page.paste_texts(_RESUME, _JD)
    analyze_page.click_analyze()
    analyze_page.wait_for_results()
    assert analyze_page.is_results_visible


def test_match_stats_show_percentages(analyze_page: AnalyzePage):
    analyze_page.paste_texts(_RESUME, _JD)
    analyze_page.click_analyze()
    analyze_page.wait_for_results()
    assert "%" in analyze_page.match_pct_text
    assert "%" in analyze_page.keyword_pct_text


def test_requirements_table_populated(analyze_page: AnalyzePage):
    analyze_page.paste_texts(_RESUME, _JD)
    analyze_page.click_analyze()
    analyze_page.wait_for_results()
    assert analyze_page.requirement_row_count > 0


def test_score_badges_present(analyze_page: AnalyzePage):
    analyze_page.paste_texts(_RESUME, _JD)
    analyze_page.click_analyze()
    analyze_page.wait_for_results()
    # At least one score badge (met / partial / missing) should be visible.
    badge_count = analyze_page.page.locator(".badge-met, .badge-partial, .badge-missing").count()
    assert badge_count > 0


# ── Chat ──────────────────────────────────────────────────────────────────────

def test_chat_reply_appears_after_analysis(analyze_page: AnalyzePage):
    analyze_page.paste_texts(_RESUME, _JD)
    analyze_page.click_analyze()
    analyze_page.wait_for_results()

    chat = ChatPage(analyze_page.page)
    chat.send_message("What skills should I highlight?")

    assert chat.total_messages >= 2, "Expected at least one user + one assistant message"
    assert len(chat.last_assistant_reply) > 0
