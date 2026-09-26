"""Setup diagnostics: `gemini-image-mcp doctor` and the `doctor` MCP tool.

Checks configuration, cookies, Gemini sign-in, the gallery and GitHub publishing, and
reports each as ok / warn / fail / skip. Never includes cookie or token values. Like `core`,
this module MUST NOT import anything from `mcp`.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Literal

from .client import (
    BROWSERS,
    BrowserCookieError,
    GeminiImageError,
    auto_browsers,
    cookie_file_for,
    cookie_source_used,
    load_browser_cookies,
)
from .config import Settings, get_settings

Status = Literal["ok", "warn", "fail", "skip"]


@dataclass
class Check:
    name: str
    status: Status
    detail: str


def _check_output_dir(settings: Settings) -> Check:
    try:
        with tempfile.TemporaryFile(dir=settings.output_dir):
            pass
    except OSError as exc:
        return Check("Output directory", "fail", f"{settings.output_dir} is not writable: {exc}")
    return Check("Output directory", "ok", str(settings.output_dir))


def _check_cookie_cache(settings: Settings) -> Check:
    from .core import _is_within

    if _is_within(settings.cookie_path, settings.output_dir):
        return Check(
            "Cookie cache",
            "warn",
            f"{settings.cookie_path} is inside the output directory, which you may share or "
            "publish. Point GEMINI_COOKIE_PATH somewhere private (or unset it).",
        )
    return Check("Cookie cache", "ok", str(settings.cookie_path))


def _check_cookies(settings: Settings) -> Check:
    source = settings.cookie_source
    if source not in ("auto", "env", *BROWSERS):
        return Check(
            "Cookies",
            "fail",
            f"Unknown GEMINI_COOKIE_SOURCE '{source}'. Use auto, env, or one of: "
            + ", ".join(BROWSERS)
            + ".",
        )
    has_env = bool(settings.secure_1psid)
    if source == "env" or (source == "auto" and has_env):
        if not has_env:
            return Check("Cookies", "fail", "GEMINI_COOKIE_SOURCE=env but GEMINI_1PSID is empty.")
        if not settings.secure_1psidts:
            return Check("Cookies", "warn", "GEMINI_1PSID is set but GEMINI_1PSIDTS is empty.")
        return Check("Cookies", "ok", "Found in the environment / .env.")
    browsers = auto_browsers() if source == "auto" else (source,)
    problems: list[str] = []
    for browser in browsers:
        try:
            load_browser_cookies(browser, cookie_file_for(browser, settings))
        except BrowserCookieError as exc:
            if not (source == "auto" and exc.missing_store):
                problems.append(str(exc))
            continue
        return Check("Cookies", "ok", f"Found a Gemini login in {BROWSERS[browser]}.")
    return Check(
        "Cookies",
        "fail",
        "No Gemini cookies found. "
        + " ".join(problems)
        + (" " if problems else "")
        + "Sign in to gemini.google.com in Firefox or Edge, or set GEMINI_1PSID / "
        "GEMINI_1PSIDTS in .env.",
    )


async def _check_gemini(settings: Settings) -> Check:
    from .core import list_models

    try:
        models = await list_models(settings=settings)
    except GeminiImageError as exc:
        return Check("Gemini sign-in", "fail", str(exc))
    return Check(
        "Gemini sign-in",
        "ok",
        f"Signed in (cookies from {cookie_source_used() or 'unknown'}); "
        f"{len(models)} model(s) available.",
    )


def _check_gallery(settings: Settings) -> Check:
    from . import gallery

    count = len(gallery.load_records(settings))
    if (settings.output_dir / "manifest.corrupt.json").exists():
        return Check(
            "Gallery",
            "warn",
            f"{count} image(s). A corrupt manifest was set aside as manifest.corrupt.json.",
        )
    return Check("Gallery", "ok", f"{count} image(s) in {settings.output_dir / 'gallery.html'}")


async def _check_github(settings: Settings, online: bool) -> Check:
    from . import github

    if not settings.github_enabled:
        return Check("GitHub publishing", "skip", "Not configured (GEMINI_GITHUB_REPO is empty).")
    mode = "auto-publish on" if settings.github_auto_publish else "publish on request"
    source = github.token_source(settings)
    if source is None:
        return Check(
            "GitHub publishing",
            "fail",
            "No token: set GEMINI_GITHUB_TOKEN (a fine-grained token for this one repo).",
        )
    notes = [f"repo {settings.github_repo}", mode, f"token from {source}"]
    status: Status = "ok"
    if source.startswith("GitHub CLI"):
        status = "warn"
        notes.append("the gh token usually reaches ALL your repos; prefer a fine-grained token")
    if online:
        try:
            info = await github.repo_info(settings)
        except github.GitHubPublishError as exc:
            return Check("GitHub publishing", "fail", str(exc))
        if info.get("permissions", {}).get("push") is False:
            return Check("GitHub publishing", "fail", "The token cannot push to that repository.")
        if not info.get("private"):
            notes.append("repository is PUBLIC")
            if settings.github_include_prompt:
                status = "warn"
                notes.append("prompts go into public file names/commits (GEMINI_GITHUB_INCLUDE_PROMPT)")
    return Check("GitHub publishing", status, "; ".join(notes) + ".")


async def run_checks(settings: Settings | None = None, *, online: bool = True) -> list[Check]:
    """Run every check. `online=False` skips the network calls to Gemini and GitHub."""
    settings = settings or get_settings()
    checks = [
        _check_output_dir(settings),
        _check_cookie_cache(settings),
        _check_cookies(settings),
    ]
    if online:
        checks.append(await _check_gemini(settings))
    else:
        checks.append(Check("Gemini sign-in", "skip", "Offline mode."))
    checks.append(_check_gallery(settings))
    checks.append(await _check_github(settings, online))
    return checks


def format_report(checks: list[Check]) -> str:
    labels = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}
    lines = [f"{labels[c.status]} {c.name}: {c.detail}" for c in checks]
    failed = sum(c.status == "fail" for c in checks)
    warned = sum(c.status == "warn" for c in checks)
    lines.append("")
    lines.append(
        "All good." if not (failed or warned) else f"{failed} problem(s), {warned} warning(s)."
    )
    return "\n".join(lines)
