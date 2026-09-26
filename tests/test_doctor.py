"""Tests for gemini_image_mcp.doctor and the `doctor` CLI entry point.

No network calls: Gemini (`core.list_models`) and GitHub (`github.repo_info`,
`github.token_source`) are monkeypatched wherever a check would otherwise reach them.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gemini_image_mcp import core, doctor, github, server
from gemini_image_mcp.client import GeminiAuthError
from gemini_image_mcp.config import Settings


def _settings(tmp_path: Path, **overrides) -> Settings:
    output_dir = tmp_path / "output"
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    base = Settings(
        secure_1psid="fake-psid",
        secure_1psidts="fake-psidts",
        output_dir=output_dir,
        cookie_path=tmp_path / "private" / "cookies",
        default_model=None,
        timeout=120,
        proxy=None,
        refresh_interval=240.0,
    )
    return replace(base, **overrides)


def _by_name(checks: list[doctor.Check]) -> dict[str, doctor.Check]:
    return {c.name: c for c in checks}


async def test_offline_healthy_setup(tmp_path):
    checks = _by_name(await doctor.run_checks(_settings(tmp_path), online=False))

    assert checks["Output directory"].status == "ok"
    assert checks["Cookie cache"].status == "ok"
    assert checks["Cookies"].status == "ok"
    assert checks["Gemini sign-in"].status == "skip"
    assert checks["Gallery"].status == "ok"
    assert checks["GitHub publishing"].status == "skip"


async def test_report_never_contains_secret_values(tmp_path, monkeypatch):
    settings = _settings(
        tmp_path, secure_1psid="SECRET-PSID", github_repo="me/pics", github_token="SECRET-TOK"
    )
    monkeypatch.setattr(
        github, "repo_info", AsyncMock(return_value={"private": True, "permissions": {"push": True}})
    )
    monkeypatch.setattr(core, "list_models", AsyncMock(return_value=[]))

    report = doctor.format_report(await doctor.run_checks(settings))

    assert "SECRET" not in report


async def test_cookie_cache_inside_output_dir_warns(tmp_path):
    settings = _settings(tmp_path)
    settings = replace(settings, cookie_path=settings.output_dir / ".cookies")

    check = doctor._check_cookie_cache(settings)

    assert check.status == "warn"


def test_env_cookie_source_without_psid_fails(tmp_path):
    check = doctor._check_cookies(_settings(tmp_path, secure_1psid=None, cookie_source="env"))
    assert check.status == "fail"


def test_missing_psidts_warns(tmp_path):
    check = doctor._check_cookies(_settings(tmp_path, secure_1psidts=None))
    assert check.status == "warn"


def test_unknown_cookie_source_fails(tmp_path):
    check = doctor._check_cookies(_settings(tmp_path, cookie_source="chrome"))
    assert check.status == "fail"
    assert "chrome" in check.detail


def test_firefox_cookies_found_or_missing(tmp_path, monkeypatch):
    settings = _settings(tmp_path, secure_1psid=None)

    monkeypatch.setattr(doctor, "load_firefox_cookies", lambda _file: ("psid", "psidts"))
    assert doctor._check_cookies(settings).status == "ok"

    monkeypatch.setattr(doctor, "load_firefox_cookies", lambda _file: (None, None))
    missing = doctor._check_cookies(settings)
    assert missing.status == "fail"
    assert "Firefox" in missing.detail


async def test_gemini_sign_in_ok_and_fail(tmp_path, monkeypatch):
    settings = _settings(tmp_path)

    monkeypatch.setattr(
        core, "list_models", AsyncMock(return_value=[{"display_name": "a", "model_name": "a"}])
    )
    monkeypatch.setattr(doctor, "cookie_source_used", lambda: "Firefox")
    ok = await doctor._check_gemini(settings)
    assert ok.status == "ok"
    assert "Firefox" in ok.detail and "1 model" in ok.detail

    monkeypatch.setattr(core, "list_models", AsyncMock(side_effect=GeminiAuthError("expired")))
    failed = await doctor._check_gemini(settings)
    assert failed.status == "fail"
    assert "expired" in failed.detail


def test_corrupt_manifest_warns(tmp_path):
    settings = _settings(tmp_path)
    (settings.output_dir / "manifest.corrupt.json").write_text("{", encoding="utf-8")

    assert doctor._check_gallery(settings).status == "warn"


async def test_github_without_token_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(github, "token_source", lambda _settings: None)

    check = await doctor._check_github(_settings(tmp_path, github_repo="me/pics"), online=False)

    assert check.status == "fail"


async def test_github_gh_cli_token_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(github, "token_source", lambda _settings: "GitHub CLI (gh auth token)")

    check = await doctor._check_github(_settings(tmp_path, github_repo="me/pics"), online=False)

    assert check.status == "warn"
    assert "fine-grained" in check.detail


@pytest.mark.parametrize(
    ("info", "include_prompt", "status"),
    [
        ({"private": True, "permissions": {"push": True}}, True, "ok"),
        ({"private": False, "permissions": {"push": True}}, False, "ok"),
        ({"private": False, "permissions": {"push": True}}, True, "warn"),
        ({"private": True, "permissions": {"push": False}}, False, "fail"),
    ],
)
async def test_github_online_repo_checks(tmp_path, monkeypatch, info, include_prompt, status):
    monkeypatch.setattr(github, "repo_info", AsyncMock(return_value=info))
    settings = _settings(
        tmp_path, github_repo="me/pics", github_token="tok", github_include_prompt=include_prompt
    )

    check = await doctor._check_github(settings, online=True)

    assert check.status == status
    if not info["private"]:
        assert "PUBLIC" in check.detail


async def test_github_unreachable_repo_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        github, "repo_info", AsyncMock(side_effect=github.GitHubPublishError("GitHub API returned 404."))
    )
    settings = _settings(tmp_path, github_repo="me/pics", github_token="tok")

    check = await doctor._check_github(settings, online=True)

    assert check.status == "fail"
    assert "404" in check.detail


def test_format_report_summary():
    report = doctor.format_report(
        [
            doctor.Check("A", "ok", "fine"),
            doctor.Check("B", "warn", "hmm"),
            doctor.Check("C", "fail", "broken"),
        ]
    )
    assert "[ OK ] A: fine" in report
    assert "[FAIL] C: broken" in report
    assert report.endswith("1 problem(s), 1 warning(s).")

    assert doctor.format_report([doctor.Check("A", "ok", "fine")]).endswith("All good.")


async def test_cli_exit_code_reflects_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(server, "get_settings", lambda: _settings(tmp_path))
    assert await server._doctor_cli(online=False) == 0
    assert "All good." in capsys.readouterr().out

    monkeypatch.setattr(
        server, "get_settings", lambda: _settings(tmp_path, secure_1psid=None, cookie_source="env")
    )
    assert await server._doctor_cli(online=False) == 1
    assert "[FAIL] Cookies" in capsys.readouterr().out
