"""Shared pytest fixtures.

Critically: the project root has a real `.env` file with the developer's live Gemini session
cookies. `Settings.load()` defaults to reading `<project root>/.env` whenever no explicit
`env_file` is passed. To guarantee tests never read it (and never depend on whatever ambient
`GEMINI_*` variables happen to be set in the shell that runs pytest), every test in this suite
runs with the environment scrubbed via the autouse fixture below. Tests that exercise
`Settings.load()` directly must also pass an explicit `env_file=` pointing at a path under
`tmp_path` (see test_config.py) rather than relying on the default.
"""

from __future__ import annotations

import pytest

_GEMINI_ENV_VARS = (
    "GEMINI_1PSID",
    "GEMINI_1PSIDTS",
    "GEMINI_OUTPUT_DIR",
    "GEMINI_COOKIE_PATH",
    "GEMINI_DEFAULT_MODEL",
    "GEMINI_TIMEOUT",
    "GEMINI_PROXY",
    "GEMINI_REFRESH_INTERVAL",
)


@pytest.fixture(autouse=True)
def _isolate_gemini_env(monkeypatch):
    """Scrub all GEMINI_* env vars before every test, regardless of the ambient shell."""
    for var in _GEMINI_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
