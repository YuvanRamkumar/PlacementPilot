"""Page object for the main analysis form."""
from __future__ import annotations

from playwright.sync_api import Page


class AnalyzePage:
    def __init__(self, page: Page, base_url: str) -> None:
        self._page = page
        self._base_url = base_url

    # ── Navigation ────────────────────────────────────────────────────────────

    def navigate(self) -> None:
        self._page.goto(self._base_url)
        self._page.wait_for_selector("#analyze-btn")

    # ── Actions ───────────────────────────────────────────────────────────────

    def paste_texts(self, resume: str, jd: str) -> None:
        self._page.fill("#resume-text", resume)
        self._page.fill("#jd-text", jd)

    def click_analyze(self) -> None:
        self._page.click("#analyze-btn")

    def switch_to_file_tab(self) -> None:
        self._page.click("#tab-file")

    def switch_to_paste_tab(self) -> None:
        self._page.click("#tab-paste")

    def wait_for_results(self, timeout: int = 15_000) -> None:
        self._page.wait_for_selector("#results", state="visible", timeout=timeout)
        self._page.wait_for_selector("#analyze-btn:not([disabled])", timeout=timeout)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def page(self) -> Page:
        return self._page

    @property
    def is_results_visible(self) -> bool:
        return self._page.is_visible("#results")

    @property
    def is_error_visible(self) -> bool:
        return self._page.is_visible("#error-box")

    @property
    def error_text(self) -> str:
        return self._page.inner_text("#error-box")

    @property
    def match_pct_text(self) -> str:
        return self._page.inner_text("#stat-match")

    @property
    def keyword_pct_text(self) -> str:
        return self._page.inner_text("#stat-kw")

    @property
    def requirement_row_count(self) -> int:
        return self._page.locator("#req-table tr").count()

    def tab_is_active(self, tab_id: str) -> bool:
        cls = self._page.get_attribute(f"#{tab_id}", "class") or ""
        return "active" in cls
