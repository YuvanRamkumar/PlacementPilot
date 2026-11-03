"""Page object for the follow-up chat section."""
from __future__ import annotations

from playwright.sync_api import Page


class ChatPage:
    def __init__(self, page: Page) -> None:
        self._page = page

    def send_message(self, text: str, timeout: int = 10_000) -> None:
        self._page.fill("#chat-input", text)
        before = self._page.locator(".msg.assistant").count()
        self._page.click("button:has-text('Send')")
        # Wait until a new assistant message appears.
        self._page.wait_for_function(
            f"document.querySelectorAll('.msg.assistant').length > {before}",
            timeout=timeout,
        )

    @property
    def total_messages(self) -> int:
        return self._page.locator(".msg").count()

    @property
    def last_assistant_reply(self) -> str:
        msgs = self._page.locator(".msg.assistant").all()
        return msgs[-1].inner_text() if msgs else ""
