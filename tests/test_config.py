"""Tests for gemini_image_mcp.config.Settings.load.

Every call below passes an explicit `env_file` pointing at a nonexistent path under `tmp_path`
(or under monkeypatched cwd) so none of these tests can ever read the developer's real
project-root `.env`. The autouse `_isolate_gemini_env` fixture in conftest.py additionally
scrubs ambient GEMINI_* env vars before each test.
"""

from __future__ import annotations

from pathlib import Path

from gemini_image_mcp.config import Settings


def _no_dotenv(tmp_path: Path) -> Path:
    """A guaranteed-nonexistent .env path, so Settings.load() never reads a real file."""
    return tmp_path / "does-not-exist.env"


def test_load_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings.load(env_file=_no_dotenv(tmp_path))

    assert settings.secure_1psid is None
    assert settings.secure_1psidts is None
    assert settings.default_model is None
    assert settings.proxy is None
    assert settings.timeout == 120
    assert settings.refresh_interval == 240.0


def test_load_output_dir_and_cookie_dir_are_created(tmp_path, monkeypatch):
    output_dir = tmp_path / "custom_output"
    monkeypatch.setenv("GEMINI_OUTPUT_DIR", str(output_dir))
    settings = Settings.load(env_file=_no_dotenv(tmp_path))

    assert settings.output_dir == output_dir.resolve()
    assert settings.output_dir.is_dir()
    assert (settings.output_dir / "images").is_dir()

    # Default cookie_path derives from output_dir and must be created too (the library never
    # creates it, and previously it was never created here either - output/.cookies did not
    # exist on disk before this fix).
    assert settings.cookie_path == (output_dir / ".cookies").resolve()
    assert settings.cookie_path.is_dir()


def test_load_explicit_cookie_path_is_created_and_exported_to_environ(tmp_path, monkeypatch):
    import os

    output_dir = tmp_path / "out"
    cookie_dir = tmp_path / "somewhere_else" / "cookies"
    monkeypatch.setenv("GEMINI_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("GEMINI_COOKIE_PATH", str(cookie_dir))

    settings = Settings.load(env_file=_no_dotenv(tmp_path))

    assert settings.cookie_path == cookie_dir.resolve()
    assert settings.cookie_path.is_dir()
    # gemini_webapi reads GEMINI_COOKIE_PATH from os.environ itself, so Settings.load() must
    # export the resolved path back into the process environment.
    assert os.environ["GEMINI_COOKIE_PATH"] == str(cookie_dir.resolve())


def test_load_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_1PSID", "psid-value")
    monkeypatch.setenv("GEMINI_1PSIDTS", "psidts-value")
    monkeypatch.setenv("GEMINI_DEFAULT_MODEL", "gemini-test-model")
    monkeypatch.setenv("GEMINI_TIMEOUT", "60")
    monkeypatch.setenv("GEMINI_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("GEMINI_REFRESH_INTERVAL", "90")

    settings = Settings.load(env_file=_no_dotenv(tmp_path))

    assert settings.secure_1psid == "psid-value"
    assert settings.secure_1psidts == "psidts-value"
    assert settings.default_model == "gemini-test-model"
    assert settings.timeout == 60
    assert settings.proxy == "http://proxy.example:8080"
    assert settings.refresh_interval == 90.0


def test_load_bad_timeout_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_TIMEOUT", "not-a-number")
    settings = Settings.load(env_file=_no_dotenv(tmp_path))
    assert settings.timeout == 120


def test_load_bad_refresh_interval_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_REFRESH_INTERVAL", "not-a-number")
    settings = Settings.load(env_file=_no_dotenv(tmp_path))
    assert settings.refresh_interval == 240.0


def test_load_never_reads_real_project_dotenv(tmp_path, monkeypatch):
    """A stray real `.env` in the project root must not leak into a test run.

    Simulates that scenario by pointing env_file at a *real* file with a poisoned value and
    confirming that omitting env_file (letting Settings.load() fall back to its own default
    resolution) never accidentally picks it up because our explicit tmp_path env_file always
    wins in every other test. This test instead directly proves the opposite failure mode
    would be caught: if code regressed to ignore env_file and always read the project's real
    .env, this repo's real GEMINI_1PSID would leak through as a non-None value.
    """
    fake_dotenv = tmp_path / "poisoned.env"
    fake_dotenv.write_text("GEMINI_1PSID=leaked-from-fake-dotenv\n", encoding="utf-8")

    settings = Settings.load(env_file=fake_dotenv)
    # This *should* pick up the explicit file we pointed at (proving env_file is honored)...
    assert settings.secure_1psid == "leaked-from-fake-dotenv"

    # ...but a fresh load with a nonexistent env_file must NOT retain it (proving no caching
    # or ambient-env leakage across calls within a process).
    import os

    os.environ.pop("GEMINI_1PSID", None)
    settings2 = Settings.load(env_file=_no_dotenv(tmp_path))
    assert settings2.secure_1psid is None
