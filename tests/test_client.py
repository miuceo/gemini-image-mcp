"""Tests for gemini_image_mcp.client.map_library_error.

Verifies every real `gemini_webapi.exceptions` class maps to the right project-level
exception, and that unknown/arbitrary exceptions are wrapped safely rather than leaking a
raw third-party (or bare) exception type out of the client module's public API.
"""

from __future__ import annotations

import pytest
from gemini_webapi.exceptions import (
    APIError,
    AuthError,
    GeminiError,
    ImageGenerationError,
    ModelInvalidError,
    TemporarilyBlockedError,
)
from gemini_webapi.exceptions import TimeoutError as LibTimeoutError
from gemini_webapi.exceptions import UsageLimitExceededError

from gemini_image_mcp.client import (
    GeminiAuthError,
    GeminiGenerationError,
    GeminiImageError,
    GeminiUsageLimitError,
    map_library_error,
)


def test_auth_error_maps_to_gemini_auth_error():
    result = map_library_error(AuthError())
    assert isinstance(result, GeminiAuthError)
    assert "cookies" in str(result).lower()


def test_usage_limit_exceeded_maps_to_usage_limit_error():
    result = map_library_error(UsageLimitExceededError("quota gone"))
    assert isinstance(result, GeminiUsageLimitError)
    assert "quota gone" in str(result)


def test_usage_limit_exceeded_with_no_message_gets_a_default():
    result = map_library_error(UsageLimitExceededError())
    assert isinstance(result, GeminiUsageLimitError)
    assert str(result)


def test_temporarily_blocked_maps_to_usage_limit_error():
    result = map_library_error(TemporarilyBlockedError("429"))
    assert isinstance(result, GeminiUsageLimitError)
    assert "429" in str(result)
    assert "rate limited" in str(result).lower()


@pytest.mark.parametrize(
    "exc",
    [
        LibTimeoutError("timed out"),
        ModelInvalidError("bad model"),
        APIError("api broke"),
        ImageGenerationError("bad image"),  # subclass of APIError
        GeminiError("generic failure"),
    ],
)
def test_generation_family_maps_to_generation_error(exc):
    result = map_library_error(exc)
    assert isinstance(result, GeminiGenerationError)
    assert str(exc) in str(result)


def test_unknown_exception_is_wrapped_safely():
    result = map_library_error(ValueError("something else broke"))
    assert isinstance(result, GeminiGenerationError)
    assert "ValueError" in str(result)
    assert "something else broke" in str(result)


def test_bare_exception_is_wrapped_safely():
    result = map_library_error(Exception())
    assert isinstance(result, GeminiGenerationError)
    assert "Exception" in str(result)


def test_all_mapped_results_are_gemini_image_error():
    for exc in (
        AuthError(),
        UsageLimitExceededError("x"),
        TemporarilyBlockedError("x"),
        LibTimeoutError("x"),
        ModelInvalidError("x"),
        APIError("x"),
        GeminiError("x"),
        RuntimeError("x"),
    ):
        assert isinstance(map_library_error(exc), GeminiImageError)


# --- cookie source selection ---------------------------------------------------------


from pathlib import Path

from gemini_image_mcp import client as client_mod
from gemini_image_mcp.config import Settings


def _settings(**overrides) -> Settings:
    base = dict(
        secure_1psid=None,
        secure_1psidts=None,
        output_dir=Path("."),
        cookie_path=Path("."),
        default_model=None,
        timeout=5,
        proxy=None,
        refresh_interval=240.0,
        cookie_source="auto",
        firefox_cookie_file=None,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def fake_init(monkeypatch):
    calls = []

    async def _fake(settings, psid, psidts):
        calls.append((psid, psidts))
        if psid in (None, "stale"):
            raise GeminiAuthError("nope")
        return object()

    monkeypatch.setattr(client_mod, "_init_client", _fake)
    monkeypatch.setattr(client_mod, "load_firefox_cookies", lambda f=None: ("ff", "ffts"))
    monkeypatch.setattr(client_mod, "_client", None)
    return calls


async def test_auto_without_env_cookies_reads_firefox(fake_init):
    await client_mod.get_client(_settings())
    assert fake_init == [("ff", "ffts")]


async def test_auto_prefers_env_cookies(fake_init):
    await client_mod.get_client(_settings(secure_1psid="env", secure_1psidts="envts"))
    assert fake_init == [("env", "envts")]


async def test_auto_falls_back_to_firefox_when_env_cookies_stale(fake_init):
    await client_mod.get_client(_settings(secure_1psid="stale"))
    assert fake_init == [("stale", None), ("ff", "ffts")]


async def test_env_source_never_reads_firefox(fake_init):
    with pytest.raises(GeminiAuthError):
        await client_mod.get_client(_settings(secure_1psid="stale", cookie_source="env"))
    assert fake_init == [("stale", None)]
