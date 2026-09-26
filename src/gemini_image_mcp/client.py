"""Lazy, process-wide `GeminiClient` lifecycle management.

This module owns constructing and initializing the third-party `gemini_webapi.GeminiClient`
exactly once per process, and translating its exceptions into this project's own exception
hierarchy so callers (MCP tools, a future Telegram bot, scripts) never need to import
`gemini_webapi.exceptions` directly.
"""

from __future__ import annotations

import asyncio
import sys

from gemini_webapi import GeminiClient
from gemini_webapi.exceptions import (
    APIError,
    AuthError,
    GeminiError,
    ModelInvalidError,
    TemporarilyBlockedError,
    TimeoutError as LibTimeoutError,
    UsageLimitExceededError,
)

from .config import Settings, get_settings


class GeminiImageError(Exception):
    """Base exception for all errors raised by gemini_image_mcp."""


class GeminiAuthError(GeminiImageError):
    """Raised when authentication with the Gemini web app fails.

    Almost always means the session cookies are missing, expired, or invalid.
    """


class GeminiUsageLimitError(GeminiImageError):
    """Raised when the Gemini account has hit a usage/quota limit."""


class GeminiGenerationError(GeminiImageError):
    """Raised when content/image generation fails or returns no usable output."""


_AUTH_ERROR_MESSAGE = (
    "Gemini authentication failed. Your session cookies are missing, expired, or invalid.\n"
    "You need to supply two cookies from a logged-in gemini.google.com session:\n"
    "  - __Secure-1PSID  (set as the GEMINI_1PSID environment variable)\n"
    "  - __Secure-1PSIDTS (set as the GEMINI_1PSIDTS environment variable)\n"
    "Alternatively, sign in to gemini.google.com in Firefox (or Edge) and leave GEMINI_1PSID\n"
    "unset; the cookies are then read from the browser automatically.\n"
    "See the 'Getting your cookies' section of README.md for step-by-step instructions."
)


# GEMINI_COOKIE_SOURCE value -> display name. Each key is also a `browser_cookie3` loader.
BROWSERS: dict[str, str] = {
    "firefox": "Firefox",
    "librewolf": "LibreWolf",
    "edge": "Edge",
    "chrome": "Chrome",
    "brave": "Brave",
    "chromium": "Chromium",
    "vivaldi": "Vivaldi",
    "opera": "Opera",
    "safari": "Safari",
}

_PSID = "__Secure-1PSID"
_PSIDTS = "__Secure-1PSIDTS"


class BrowserCookieError(GeminiAuthError):
    """Raised when a browser's Gemini cookies can't be read; the message says why."""

    def __init__(self, message: str, *, missing_store: bool = False) -> None:
        super().__init__(message)
        # True when the browser simply isn't installed / has no cookie store - not worth
        # reporting when merely probing browsers in "auto" mode.
        self.missing_store = missing_store


def auto_browsers() -> tuple[str, ...]:
    """Browsers tried, in order, when GEMINI_COOKIE_SOURCE is "auto" and no .env cookies work.

    On macOS, reading Chromium-family cookies pops up a Keychain password prompt, which a
    background server must never trigger unasked, so only Firefox-family browsers are probed
    there (pick a Chromium browser explicitly to use it).
    """
    if sys.platform == "darwin":
        return ("firefox", "librewolf")
    return ("firefox", "librewolf", "edge", "chrome", "brave", "chromium", "vivaldi", "opera")


def cookie_file_for(browser: str, settings: Settings) -> str | None:
    """The configured cookie file applies to the chosen browser (Firefox when "auto")."""
    target = "firefox" if settings.cookie_source == "auto" else settings.cookie_source
    return settings.browser_cookie_file if browser == target else None


def load_browser_cookies(browser: str, cookie_file: str | None = None) -> tuple[str, str | None]:
    """Read `__Secure-1PSID` / `__Secure-1PSIDTS` from a local browser's cookie store.

    Parameters
    ----------
    browser: `str`
        A key of `BROWSERS`, e.g. "firefox" or "edge".
    cookie_file: `str | None`, optional
        Path to a specific profile's cookie database. Defaults to the browser's default profile.

    Returns
    -------
    `tuple[str, str | None]`
        The 1PSID value (always present) and the 1PSIDTS value, if the browser has one.

    Raises
    ------
    `BrowserCookieError`
        If the cookie store can't be read or holds no Gemini login, with the reason.

    """
    label = BROWSERS[browser]
    try:
        import browser_cookie3
    except ImportError:
        raise BrowserCookieError("browser-cookie3 is not installed.") from None

    try:
        jar = getattr(browser_cookie3, browser)(cookie_file=cookie_file, domain_name="google.com")
    except Exception as exc:  # noqa: BLE001 - every failure becomes a readable reason
        if type(exc).__name__ == "RequiresAdminError":
            raise BrowserCookieError(
                f"{label}: its cookies use app-bound encryption on Windows and can only be "
                "read with administrator rights (don't run this server as admin)."
            ) from None
        missing = type(exc).__name__ == "BrowserCookieError" or isinstance(
            exc, FileNotFoundError
        )
        raise BrowserCookieError(
            f"{label}: cookie store not found or unreadable ({type(exc).__name__}).",
            missing_store=missing,
        ) from None

    values = {cookie.name: cookie.value for cookie in jar if cookie.domain == ".google.com"}
    psid = values.get(_PSID)
    if not psid:
        raise BrowserCookieError(f"{label}: not signed in to gemini.google.com.")
    return psid, values.get(_PSIDTS)


def map_library_error(exc: Exception) -> GeminiImageError:
    """Translate a `gemini_webapi` exception into this project's exception hierarchy.

    Parameters
    ----------
    exc: `Exception`
        The exception raised by `gemini_webapi`.

    Returns
    -------
    `GeminiImageError`
        The corresponding project-level exception, ready to be raised in its place.

    """
    if isinstance(exc, AuthError):
        return GeminiAuthError(_AUTH_ERROR_MESSAGE)
    if isinstance(exc, UsageLimitExceededError):
        return GeminiUsageLimitError(str(exc) or "Gemini usage limit exceeded.")
    if isinstance(exc, TemporarilyBlockedError):
        return GeminiUsageLimitError(
            f"Temporarily blocked by Gemini (rate limited): {exc}"
        )
    if isinstance(exc, (LibTimeoutError, ModelInvalidError, APIError, GeminiError)):
        return GeminiGenerationError(str(exc) or f"{type(exc).__name__} from gemini_webapi.")
    # Unknown exception type: wrap it rather than letting a raw third-party exception
    # (or bare Exception) leak out of this layer's public API.
    return GeminiGenerationError(f"{type(exc).__name__}: {exc}")


_client: GeminiClient | None = None
_client_lock = asyncio.Lock()
# Where the current client's cookies came from, for diagnostics (never the values).
_cookie_source_used: str | None = None


def cookie_source_used() -> str | None:
    """Where the live client's cookies came from ("environment", "Firefox", ...), if any."""
    return _cookie_source_used if _client is not None else None


async def get_client(settings: Settings | None = None) -> GeminiClient:
    """Return the process-wide `GeminiClient`, constructing and initializing it on first call.

    Safe to call concurrently: an `asyncio.Lock` guards against double-initialization when
    multiple MCP tool calls race to obtain the client at the same time.

    Parameters
    ----------
    settings: `Settings | None`, optional
        Settings to use when constructing the client. Defaults to the process-wide settings.

    Raises
    ------
    `GeminiAuthError`
        If cookies are missing/invalid and authentication fails.
    `GeminiImageError`
        For any other failure while constructing or initializing the client.

    """
    global _client, _cookie_source_used

    if _client is not None:
        return _client

    async with _client_lock:
        # Re-check after acquiring the lock: another caller may have finished init already.
        if _client is not None:
            return _client

        settings = settings or get_settings()

        source_setting = settings.cookie_source
        if source_setting == "env" or (source_setting == "auto" and settings.secure_1psid):
            try:
                client = await _init_client(
                    settings, settings.secure_1psid, settings.secure_1psidts
                )
                source = "environment"
            except GeminiAuthError:
                if source_setting != "auto":
                    raise
                # Cookies in the environment went stale: fall back to a browser's login.
                client, label = await _init_from_browsers(settings, auto_browsers())
                source = f"{label} (the GEMINI_1PSID cookies were stale)"
        elif source_setting == "auto":
            client, source = await _init_from_browsers(settings, auto_browsers())
        elif source_setting in BROWSERS:
            client, source = await _init_from_browsers(settings, (source_setting,))
        else:
            raise GeminiAuthError(
                f"Unknown GEMINI_COOKIE_SOURCE '{source_setting}'. Use auto, env, or one of: "
                + ", ".join(BROWSERS)
                + "."
            )

        _client = client
        _cookie_source_used = source
        return _client


async def reset_client() -> None:
    """Close and drop the cached client so the next `get_client()` re-reads cookies."""
    global _client

    async with _client_lock:
        if _client is not None:
            try:
                await _client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup of a dead session
                pass
            _client = None


async def _init_from_browsers(
    settings: Settings, browsers: tuple[str, ...]
) -> tuple[GeminiClient, str]:
    """Initialize from the first browser whose Gemini login works; return it and its name.

    A browser whose login Gemini rejects (e.g. signed out long ago) doesn't stop the search:
    another browser may still hold a fresh session.
    """
    probing = len(browsers) > 1
    problems: list[str] = []
    for browser in browsers:
        label = BROWSERS[browser]
        try:
            psid, psidts = load_browser_cookies(browser, cookie_file_for(browser, settings))
        except BrowserCookieError as exc:
            if not (probing and exc.missing_store):
                problems.append(str(exc))
            continue
        try:
            return await _init_client(settings, psid, psidts), label
        except GeminiAuthError:
            problems.append(f"{label}: Gemini rejected its login (signed out or expired).")

    detail = "\n".join(f"  - {p}" for p in problems) or "  - No supported browser was found."
    raise GeminiAuthError(f"{_AUTH_ERROR_MESSAGE}\n\nBrowsers checked:\n{detail}")


async def _init_client(
    settings: Settings, secure_1psid: str | None, secure_1psidts: str | None
) -> GeminiClient:
    """Construct and initialize a `GeminiClient` with the given cookie values."""
    if not secure_1psid:
        raise GeminiAuthError(_AUTH_ERROR_MESSAGE)

    client = GeminiClient(
        secure_1psid=secure_1psid,
        secure_1psidts=secure_1psidts,
        proxy=settings.proxy,
    )
    try:
        await client.init(
            timeout=settings.timeout,
            auto_close=False,
            auto_refresh=True,
            refresh_interval=settings.refresh_interval,
        )
    except Exception as exc:  # noqa: BLE001 - translated deliberately at this boundary
        raise map_library_error(exc) from exc
    return client
