"""Optional publishing of generated images to a GitHub repository.

Uploads files through the GitHub REST "contents" API (one commit per image), so no local
clone or `git` binary is needed. Enabled only when `Settings.github_repo` is set. Like
`core`, this module MUST NOT import anything from `mcp`, and must never write to stdout.

Stdlib-only (`urllib`) on purpose: the calls are few and simple, and blocking I/O is pushed
onto a worker thread by the async entry point.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import Settings

API_ROOT = "https://api.github.com"
RAW_ROOT = "https://raw.githubusercontent.com"
_TIMEOUT = 60
# Concurrent commits to the same branch can race and come back 409/422; retry a few times.
_CONFLICT_RETRIES = 3


class GitHubPublishError(Exception):
    """Raised when an image could not be published to GitHub."""


# Cached per process: (repo) -> default branch, and the resolved token.
_branch_cache: dict[str, str] = {}
_token_cache: str | None = None


def _resolve_token(settings: Settings) -> str:
    """Token from settings, then GITHUB_TOKEN, then the GitHub CLI's stored login."""
    global _token_cache
    if settings.github_token:
        return settings.github_token
    env_token = os.environ.get("GITHUB_TOKEN")
    if env_token:
        return env_token
    if _token_cache:
        return _token_cache

    gh = shutil.which("gh")
    if gh:
        try:
            # capture_output keeps gh off our stdout, which carries the MCP protocol.
            result = subprocess.run(
                [gh, "auth", "token"], capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and result.stdout.strip():
            _token_cache = result.stdout.strip()
            return _token_cache

    raise GitHubPublishError(
        "No GitHub token found. Set GEMINI_GITHUB_TOKEN (or GITHUB_TOKEN), or sign in with "
        "the GitHub CLI (`gh auth login`)."
    )


def _request(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "gemini-image-mcp")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        text = resp.read().decode("utf-8")
    return json.loads(text) if text else {}


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(exc.read().decode("utf-8")).get("message", "")
    except Exception:  # noqa: BLE001 - best-effort detail only
        detail = ""
    hint = ""
    if exc.code in (401, 403):
        hint = " Check that the token is valid and has write (Contents) access to the repo."
    elif exc.code == 404:
        hint = " Check GEMINI_GITHUB_REPO, and that the token can see that repository."
    return f"GitHub API returned {exc.code}{': ' + detail if detail else ''}.{hint}"


def _default_branch(repo: str, token: str) -> str:
    if repo not in _branch_cache:
        info = _request("GET", f"{API_ROOT}/repos/{repo}", token)
        _branch_cache[repo] = info.get("default_branch") or "main"
    return _branch_cache[repo]


def slugify(text: str, max_len: int = 48) -> str:
    """A short, URL-safe filename stem derived from a prompt."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_len:
        cut = slug[: max_len + 1]
        # Prefer ending on a whole word, unless that throws away most of the slug.
        slug = cut.rsplit("-", 1)[0] if cut.rfind("-") > max_len // 2 else cut[:max_len]
    return slug.rstrip("-") or "image"


def remote_path_for(local_path: Path, prompt: str, record_id: str, settings: Settings) -> str:
    """Repo-relative destination path.

    `images/<record id>.jpg` by default, so nothing about the prompt ends up in a possibly
    public repo; `images/watercolor-fox-65157777.jpg` when `github_include_prompt` is on.
    """
    suffix = local_path.suffix.lower()
    if settings.github_include_prompt:
        name = f"{slugify(prompt)}-{record_id[:8]}{suffix}"
    else:
        name = f"{record_id}{suffix}"
    return f"{settings.github_path}/{name}" if settings.github_path else name


def commit_message_for(prompt: str, record_id: str, settings: Settings) -> str:
    """Commit message for an uploaded image; carries the prompt only when opted in."""
    if settings.github_include_prompt:
        return f"Add image: {prompt.strip()[:72]}"
    return f"Add image {record_id[:8]}"


def raw_url(repo: str, branch: str, remote_path: str) -> str:
    quoted = urllib.parse.quote(remote_path)
    return f"{RAW_ROOT}/{repo}/{urllib.parse.quote(branch)}/{quoted}"


def _publish_sync(local_path: Path, remote_path: str, message: str, settings: Settings) -> str:
    repo = settings.github_repo
    if not repo:
        raise GitHubPublishError("GitHub publishing is not configured (GEMINI_GITHUB_REPO).")
    if not local_path.is_file():
        raise GitHubPublishError(f"Image not found: '{local_path}'")

    token = _resolve_token(settings)
    content = base64.b64encode(local_path.read_bytes()).decode("ascii")

    try:
        branch = settings.github_branch or _default_branch(repo, token)
        url = f"{API_ROOT}/repos/{repo}/contents/{urllib.parse.quote(remote_path)}"
        body = {"message": message, "content": content, "branch": branch}

        for attempt in range(_CONFLICT_RETRIES + 1):
            try:
                _request("PUT", url, token, body)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 422:
                    # The path already exists (e.g. re-publishing): update it in place,
                    # which requires the current blob sha.
                    existing = _request(
                        "GET", f"{url}?ref={urllib.parse.quote(branch)}", token
                    )
                    body["sha"] = existing.get("sha")
                elif exc.code != 409 or attempt == _CONFLICT_RETRIES:
                    raise
                if attempt == _CONFLICT_RETRIES:
                    raise
                time.sleep(0.5 * (attempt + 1))
    except urllib.error.HTTPError as exc:
        raise GitHubPublishError(_http_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        raise GitHubPublishError(f"Could not reach GitHub: {exc.reason}") from exc

    return raw_url(repo, branch, remote_path)


async def publish_file(
    local_path: Path, remote_path: str, message: str, settings: Settings
) -> str:
    """Upload `local_path` to `remote_path` in the configured repo; return its raw URL."""
    return await asyncio.to_thread(_publish_sync, local_path, remote_path, message, settings)
