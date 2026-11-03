"""
Shared fixtures for Playwright UI tests.
Starts the FastAPI app in a background thread and exposes its URL.
"""
import os
import threading
import time

import httpx
import pytest
import uvicorn

os.environ.setdefault("LLM_PROVIDER", "fake")

_HOST = "127.0.0.1"
_PORT = 8001
_BASE = f"http://{_HOST}:{_PORT}"


@pytest.fixture(scope="session", autouse=True)
def live_server():
    """Run the FastAPI app in a background daemon thread for the whole session."""
    from app.main import app

    config = uvicorn.Config(app, host=_HOST, port=_PORT, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Poll until the server responds (up to 5 s).
    for _ in range(50):
        try:
            httpx.get(f"{_BASE}/health", timeout=0.5)
            break
        except Exception:
            time.sleep(0.1)

    yield

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def base_url() -> str:
    return _BASE
