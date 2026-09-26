"""Configuration loading for gemini_image_mcp.

Reads settings from the environment (optionally via a `.env` file) into a
frozen `Settings` dataclass. All environment variables are `GEMINI_`-prefixed.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root: three levels up from this file (src/gemini_image_mcp/config.py -> project root).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_APP_NAME = "gemini-image-mcp"


def _user_state_dir() -> Path:
    """Per-user directory for private state (the cookie cache), outside the output tree.

    The cookie cache holds a live Google session, so it must not live next to the images and
    gallery, which users routinely share, zip up, or publish.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / _APP_NAME


def _env_flag(name: str) -> bool:
    """A boolean environment variable; unset or anything but 1/true/yes/on means False."""
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _migrate_legacy_cookie_cache(legacy_dir: Path, cookie_dir: Path) -> None:
    """Move cookie cache files out of the old default location, `<output_dir>/.cookies`.

    Best effort: a failure here only means the next run re-reads cookies from their source.
    """
    if not legacy_dir.is_dir() or legacy_dir == cookie_dir:
        return
    for cached in legacy_dir.glob(".cached_cookies_*.json"):
        try:
            if (cookie_dir / cached.name).exists():
                cached.unlink()
            else:
                shutil.move(str(cached), str(cookie_dir / cached.name))
        except OSError:
            pass
    try:
        legacy_dir.rmdir()
    except OSError:
        pass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the Gemini image generation client.

    All fields are populated from `GEMINI_`-prefixed environment variables.
    Use `Settings.load()` to construct an instance.
    """

    secure_1psid: str | None
    secure_1psidts: str | None
    output_dir: Path
    cookie_path: Path
    default_model: str | None
    timeout: int
    proxy: str | None
    refresh_interval: float
    cookie_source: str = "auto"
    # A specific profile's cookie database for the browser named by `cookie_source`.
    browser_cookie_file: str | None = None
    # Optional GitHub publishing: when `github_repo` ("owner/name") is set, new images are
    # uploaded there and the gallery records their public raw URL.
    github_repo: str | None = None
    github_token: str | None = None
    github_branch: str | None = None
    github_path: str = "images"
    # Off by default: a configured repo is often public, so publishing is opt-in per call
    # unless the user explicitly turns this on.
    github_auto_publish: bool = False
    # Prompts can be private; only put them in public file names/commit messages on request.
    github_include_prompt: bool = False

    @property
    def github_enabled(self) -> bool:
        return bool(self.github_repo)

    @classmethod
    def load(cls, env_file: str | Path | None = None) -> "Settings":
        """Load settings from the environment, optionally reading a `.env` file first.

        Parameters
        ----------
        env_file: `str | Path | None`, optional
            Path to a `.env` file to load. Defaults to a `.env` file in the project root,
            if one exists. Does not override variables already present in the environment.

        """
        dotenv_path = Path(env_file) if env_file else _PROJECT_ROOT / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)

        output_dir = Path(os.environ.get("GEMINI_OUTPUT_DIR") or (_PROJECT_ROOT / "output")).resolve()
        explicit_cookie_path = os.environ.get("GEMINI_COOKIE_PATH")
        cookie_path = Path(explicit_cookie_path or (_user_state_dir() / "cookies")).resolve()

        # gemini-webapi reads this env var itself to persist refreshed cookies across
        # restarts, so it must be exported before a GeminiClient is constructed.
        os.environ["GEMINI_COOKIE_PATH"] = str(cookie_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        # gemini_webapi treats GEMINI_COOKIE_PATH as a directory (it writes cache files named
        # `.cached_cookies_<1psid>.json` inside it), but never creates the directory itself -
        # it must exist before the first `save_cookies()` call or refreshed cookies silently
        # fail to persist.
        cookie_path.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            # Owner-only, like ~/.ssh: the cache files are equivalent to a signed-in session.
            # (On Windows the per-user LOCALAPPDATA default is already private to the user.)
            try:
                cookie_path.chmod(0o700)
            except OSError:
                pass
        if not explicit_cookie_path:
            _migrate_legacy_cookie_cache(output_dir / ".cookies", cookie_path)

        timeout_raw = os.environ.get("GEMINI_TIMEOUT")
        try:
            timeout = int(timeout_raw) if timeout_raw else 120
        except ValueError:
            timeout = 120

        # gemini-webapi's own `start_auto_refresh()` clamps this to a 60s floor internally
        # (`refresh_interval = max(refresh_interval, 60)`), so anything below 60 is silently
        # raised back up to 60 by the library - we don't need to duplicate that clamp here.
        #
        # The library default (600s) rarely fires at all: MCP server processes are spawned
        # per Claude Code session and frequently live under 10 minutes, and the background
        # refresh loop sleeps the *full* interval before its *first* rotation attempt. 240s
        # (4 minutes) instead gives a typical sub-10-minute session two or more rotation
        # opportunities, while staying comfortably above both the library's 60s floor and
        # `rotate_1psidts`'s own 60s anti-429 debounce (it skips rotation if the cookie cache
        # file was written less than 60s ago) - so it won't trigger extra rate-limit risk.
        refresh_interval_raw = os.environ.get("GEMINI_REFRESH_INTERVAL")
        try:
            refresh_interval = float(refresh_interval_raw) if refresh_interval_raw else 240.0
        except ValueError:
            refresh_interval = 240.0

        return cls(
            secure_1psid=os.environ.get("GEMINI_1PSID") or None,
            secure_1psidts=os.environ.get("GEMINI_1PSIDTS") or None,
            output_dir=output_dir,
            cookie_path=cookie_path,
            default_model=os.environ.get("GEMINI_DEFAULT_MODEL") or None,
            timeout=timeout,
            proxy=os.environ.get("GEMINI_PROXY") or None,
            refresh_interval=refresh_interval,
            # "auto": use GEMINI_1PSID from the environment if set, else probe browsers.
            # "env": environment only. A browser name ("firefox", "edge", ...): only that
            # browser's cookie store.
            cookie_source=(os.environ.get("GEMINI_COOKIE_SOURCE") or "auto").strip().lower(),
            browser_cookie_file=os.environ.get("GEMINI_BROWSER_COOKIE_FILE")
            # Older name, from when only Firefox was supported.
            or os.environ.get("GEMINI_FIREFOX_COOKIE_FILE")
            or None,
            github_repo=(os.environ.get("GEMINI_GITHUB_REPO") or "").strip().strip("/") or None,
            github_token=os.environ.get("GEMINI_GITHUB_TOKEN") or None,
            github_branch=os.environ.get("GEMINI_GITHUB_BRANCH") or None,
            github_path=(os.environ.get("GEMINI_GITHUB_PATH") or "images").strip().strip("/"),
            github_auto_publish=_env_flag("GEMINI_GITHUB_AUTO_PUBLISH"),
            github_include_prompt=_env_flag("GEMINI_GITHUB_INCLUDE_PROMPT"),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide cached `Settings` instance, loading it on first use."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
