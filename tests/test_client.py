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
