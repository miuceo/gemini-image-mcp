"""Lazy, process-wide `GeminiClient` lifecycle management.

This module owns constructing and initializing the third-party `gemini_webapi.GeminiClient`
exactly once per process, and translating its exceptions into this project's own exception
hierarchy so callers (MCP tools, a future Telegram bot, scripts) never need to import
`gemini_webapi.exceptions` directly.
"""

from __future__ import annotations

import asyncio

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
    "See the 'Getting your cookies' section of README.md for step-by-step instructions."
)


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
    global _client

    if _client is not None:
        return _client

    async with _client_lock:
        # Re-check after acquiring the lock: another caller may have finished init already.
        if _client is not None:
            return _client

        settings = settings or get_settings()

        if not settings.secure_1psid:
            raise GeminiAuthError(_AUTH_ERROR_MESSAGE)

        client = GeminiClient(
            secure_1psid=settings.secure_1psid,
            secure_1psidts=settings.secure_1psidts,
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

        _client = client
        return _client
