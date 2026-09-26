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
        browser_cookie_file=None,
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
    monkeypatch.setattr(client_mod, "load_browser_cookies", _fake_browsers({"firefox": "ff"}))
    monkeypatch.setattr(client_mod, "auto_browsers", lambda: ("firefox", "edge", "chrome"))
    monkeypatch.setattr(client_mod, "_client", None)
    return calls


def _fake_browsers(logins: dict[str, str], missing: tuple[str, ...] = ()):
    """A `load_browser_cookies` stand-in: `logins` maps browser -> 1PSID value."""

    def _load(browser, cookie_file=None):
        if browser in missing:
            raise client_mod.BrowserCookieError(f"{browser}: no store", missing_store=True)
        if browser not in logins:
            raise client_mod.BrowserCookieError(f"{browser}: not signed in")
        return logins[browser], f"{logins[browser]}ts"

    return _load


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


async def test_auto_uses_next_browser_with_a_login(fake_init, monkeypatch):
    monkeypatch.setattr(client_mod, "load_browser_cookies", _fake_browsers({"edge": "ed"}))

    await client_mod.get_client(_settings())

    assert fake_init == [("ed", "edts")]
    assert client_mod.cookie_source_used() == "Edge"


async def test_auto_skips_browser_whose_login_is_rejected(fake_init, monkeypatch):
    monkeypatch.setattr(
        client_mod, "load_browser_cookies", _fake_browsers({"firefox": "stale", "edge": "ed"})
    )

    await client_mod.get_client(_settings())

    assert fake_init == [("stale", "stalets"), ("ed", "edts")]


async def test_explicit_browser_reads_only_that_browser(fake_init, monkeypatch):
    monkeypatch.setattr(client_mod, "load_browser_cookies", _fake_browsers({"firefox": "ff"}))

    with pytest.raises(GeminiAuthError, match="edge: not signed in"):
        await client_mod.get_client(_settings(cookie_source="edge"))
    assert fake_init == []


async def test_auto_error_lists_reasons_but_not_missing_browsers(fake_init, monkeypatch):
    monkeypatch.setattr(
        client_mod, "load_browser_cookies", _fake_browsers({}, missing=("chrome",))
    )

    with pytest.raises(GeminiAuthError) as excinfo:
        await client_mod.get_client(_settings())

    message = str(excinfo.value)
    assert "firefox: not signed in" in message and "edge: not signed in" in message
    assert "chrome" not in message


async def test_unknown_cookie_source_is_rejected(fake_init):
    with pytest.raises(GeminiAuthError, match="Unknown GEMINI_COOKIE_SOURCE"):
        await client_mod.get_client(_settings(cookie_source="netscape"))


class _Cookie:
    def __init__(self, name, value, domain=".google.com"):
        self.name, self.value, self.domain = name, value, domain


def _fake_cookie3(monkeypatch, loader):
    import sys
    import types

    module = types.ModuleType("browser_cookie3")
    module.edge = loader
    monkeypatch.setitem(sys.modules, "browser_cookie3", module)


def test_load_browser_cookies_reads_gemini_cookies(monkeypatch):
    jar = [
        _Cookie("__Secure-1PSID", "p"),
        _Cookie("__Secure-1PSIDTS", "t"),
        _Cookie("__Secure-1PSID", "other", domain=".example.com"),
    ]
    _fake_cookie3(monkeypatch, lambda cookie_file=None, domain_name="": jar)

    assert client_mod.load_browser_cookies("edge") == ("p", "t")


def test_load_browser_cookies_not_signed_in(monkeypatch):
    _fake_cookie3(monkeypatch, lambda cookie_file=None, domain_name="": [])

    with pytest.raises(client_mod.BrowserCookieError, match="not signed in"):
        client_mod.load_browser_cookies("edge")


def test_load_browser_cookies_explains_app_bound_encryption(monkeypatch):
    class RequiresAdminError(Exception):
        pass

    def _raise(cookie_file=None, domain_name=""):
        raise RequiresAdminError("This operation requires admin.")

    _fake_cookie3(monkeypatch, _raise)

    with pytest.raises(client_mod.BrowserCookieError, match="administrator") as excinfo:
        client_mod.load_browser_cookies("edge")
    assert not excinfo.value.missing_store


def test_load_browser_cookies_marks_missing_store(monkeypatch):
    class BrowserCookieError(Exception):
        pass

    def _raise(cookie_file=None, domain_name=""):
        raise BrowserCookieError("Failed to find cookies")

    _fake_cookie3(monkeypatch, _raise)

    with pytest.raises(client_mod.BrowserCookieError) as excinfo:
        client_mod.load_browser_cookies("edge")
    assert excinfo.value.missing_store


def test_cookie_file_applies_to_chosen_browser_only():
    auto = _settings(browser_cookie_file="x.sqlite")
    edge = _settings(cookie_source="edge", browser_cookie_file="x.db")

    assert client_mod.cookie_file_for("firefox", auto) == "x.sqlite"
    assert client_mod.cookie_file_for("edge", auto) is None
    assert client_mod.cookie_file_for("edge", edge) == "x.db"
    assert client_mod.cookie_file_for("firefox", edge) is None
