"""Tests for optional GitHub publishing (gemini_image_mcp.github + core wiring).

No network calls: `github._request` is monkeypatched with a fake that records calls and
simulates the GitHub contents API.
"""

from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from gemini_webapi.types import GeneratedImage

from gemini_image_mcp import core, gallery, github
from gemini_image_mcp.client import GeminiGenerationError
from gemini_image_mcp.config import Settings

JPEG_MAGIC = bytes.fromhex("ffd8ffe0") + b"\x00" * 12


def _settings(tmp_path: Path, **overrides) -> Settings:
    output_dir = tmp_path / "output"
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    base = Settings(
        secure_1psid="fake-psid",
        secure_1psidts="fake-psidts",
        output_dir=output_dir,
        cookie_path=output_dir / ".cookies",
        default_model=None,
        timeout=120,
        proxy=None,
        refresh_interval=240.0,
        github_repo="someone/pics",
        github_token="tok",
        github_branch="main",
        github_auto_publish=True,
    )
    return replace(base, **overrides)


def _http_error(code: int, message: str = "") -> urllib.error.HTTPError:
    body = io.BytesIO(json.dumps({"message": message}).encode("utf-8"))
    return urllib.error.HTTPError("https://api.github.com/x", code, message, {}, body)


class FakeGitHub:
    def __init__(self, fail_first_put_with: int | None = None):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.fail_first_put_with = fail_first_put_with

    def __call__(self, method, url, token, body=None):
        self.calls.append((method, url, body))
        if method == "PUT" and self.fail_first_put_with:
            code, self.fail_first_put_with = self.fail_first_put_with, None
            raise _http_error(code, "sha wasn't supplied")
        if method == "GET" and "/contents/" in url:
            return {"sha": "abc123"}
        if method == "GET":
            return {"default_branch": "trunk"}
        return {"content": {"path": url}}

    @property
    def puts(self):
        return [c for c in self.calls if c[0] == "PUT"]


@pytest.fixture
def fake_github(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(github, "_request", fake)
    github._branch_cache.clear()
    return fake


@pytest.fixture
def fake_client(monkeypatch):
    async def _fake_save(self, path, filename=None, verbose=False, **kwargs):
        dest = Path(path) / filename
        dest.write_bytes(JPEG_MAGIC)
        return str(dest)

    monkeypatch.setattr(GeneratedImage, "save", _fake_save)
    client = Mock()
    client.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            images=[GeneratedImage(url="https://example.com/i.png", title="[Image]", alt="")],
            text="ok",
        )
    )
    monkeypatch.setattr(core, "get_client", AsyncMock(return_value=client))
    return client


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_slugify():
    assert github.slugify("A Watercolor Fox, sleeping!") == "a-watercolor-fox-sleeping"
    assert github.slugify("!!!") == "image"
    assert len(github.slugify("word " * 50)) <= 48
    assert github.slugify(
        "Minimal text-only wordmark logo, dark mode version, for the brand"
    ) == "minimal-text-only-wordmark-logo-dark-mode"
    assert github.slugify("x" * 100) == "x" * 48


def test_remote_path_and_message_omit_prompt_by_default(tmp_path):
    settings = _settings(tmp_path)
    path = github.remote_path_for(Path("x/abc.JPG"), "my secret idea", "0123456789abcdef", settings)
    message = github.commit_message_for("my secret idea", "0123456789abcdef", settings)
    assert path == "images/0123456789abcdef.jpg"
    assert "secret" not in message


def test_remote_path_uses_prefix_slug_and_short_id(tmp_path):
    settings = _settings(tmp_path, github_path="assets/gen", github_include_prompt=True)
    path = github.remote_path_for(Path("x/abc.JPG"), "A fox", "0123456789abcdef", settings)
    assert path == "assets/gen/a-fox-01234567.jpg"


def test_settings_load_reads_github_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("GEMINI_GITHUB_REPO", "/me/imgs/")
    monkeypatch.setenv("GEMINI_GITHUB_AUTO_PUBLISH", "yes")
    monkeypatch.setenv("GEMINI_GITHUB_INCLUDE_PROMPT", "true")
    settings = Settings.load(env_file=tmp_path / "none.env")
    assert settings.github_repo == "me/imgs"
    assert settings.github_enabled
    assert settings.github_auto_publish is True
    assert settings.github_include_prompt is True
    assert settings.github_path == "images"


def test_github_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_OUTPUT_DIR", str(tmp_path / "out"))
    settings = Settings.load(env_file=tmp_path / "none.env")
    assert not settings.github_enabled
    # Publishing and prompt disclosure are both opt-in.
    assert settings.github_auto_publish is False
    assert settings.github_include_prompt is False


# ---------------------------------------------------------------------------
# publish_file
# ---------------------------------------------------------------------------


async def test_publish_file_puts_base64_and_returns_raw_url(tmp_path, fake_github):
    settings = _settings(tmp_path)
    image = tmp_path / "a.jpg"
    image.write_bytes(JPEG_MAGIC)

    url = await github.publish_file(image, "images/a.jpg", "Add a", settings)

    assert url == "https://raw.githubusercontent.com/someone/pics/main/images/a.jpg"
    (method, api_url, body), = fake_github.puts
    assert api_url.endswith("/repos/someone/pics/contents/images/a.jpg")
    assert body["branch"] == "main"
    assert body["message"] == "Add a"
    assert body["content"]


async def test_publish_file_uses_repo_default_branch_when_unset(tmp_path, fake_github):
    settings = _settings(tmp_path, github_branch=None)
    image = tmp_path / "a.jpg"
    image.write_bytes(JPEG_MAGIC)

    url = await github.publish_file(image, "images/a.jpg", "m", settings)

    assert "/trunk/" in url
    assert fake_github.puts[0][2]["branch"] == "trunk"


async def test_publish_file_overwrites_existing_path_with_sha(tmp_path, monkeypatch):
    fake = FakeGitHub(fail_first_put_with=422)
    monkeypatch.setattr(github, "_request", fake)
    settings = _settings(tmp_path)
    image = tmp_path / "a.jpg"
    image.write_bytes(JPEG_MAGIC)

    await github.publish_file(image, "images/a.jpg", "m", settings)

    assert len(fake.puts) == 2
    assert fake.puts[1][2]["sha"] == "abc123"


async def test_publish_file_maps_auth_errors(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise _http_error(401, "Bad credentials")

    monkeypatch.setattr(github, "_request", boom)
    settings = _settings(tmp_path)
    image = tmp_path / "a.jpg"
    image.write_bytes(JPEG_MAGIC)

    with pytest.raises(github.GitHubPublishError, match="401: Bad credentials"):
        await github.publish_file(image, "images/a.jpg", "m", settings)


def test_resolve_token_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")
    assert github._resolve_token(_settings(tmp_path, github_token=None)) == "from-env"


# ---------------------------------------------------------------------------
# core wiring
# ---------------------------------------------------------------------------


async def test_generate_publishes_and_stores_remote_url(tmp_path, fake_client, fake_github):
    settings = _settings(tmp_path)

    records = await core.generate_images("a red fox", settings=settings)

    assert records[0].remote_url == (
        f"https://raw.githubusercontent.com/someone/pics/main/images/{records[0].id}.jpg"
    )
    assert "red fox" not in fake_github.puts[0][2]["message"]
    manifest = json.loads((settings.output_dir / gallery.MANIFEST_NAME).read_text("utf-8"))
    assert manifest[0]["remote_url"] == records[0].remote_url
    assert "publish_error" not in manifest[0]


async def test_generate_respects_publish_false(tmp_path, fake_client, fake_github):
    settings = _settings(tmp_path)
    records = await core.generate_images("a red fox", settings=settings, publish=False)
    assert records[0].remote_url is None
    assert fake_github.calls == []


async def test_auto_publish_off_needs_explicit_opt_in(tmp_path, fake_client, fake_github):
    settings = _settings(tmp_path, github_auto_publish=False)
    records = await core.generate_images("a red fox", settings=settings)
    assert records[0].remote_url is None

    records = await core.generate_images("a red fox", settings=settings, publish=True)
    assert records[0].remote_url


async def test_no_publishing_when_not_configured(tmp_path, fake_client, fake_github):
    settings = _settings(tmp_path, github_repo=None)
    records = await core.generate_images("a fox", settings=settings, publish=True)
    assert records[0].remote_url is None
    assert fake_github.calls == []


async def test_publish_failure_keeps_image_and_reports_error(tmp_path, fake_client, monkeypatch):
    def boom(*args, **kwargs):
        raise _http_error(403, "Resource not accessible")

    monkeypatch.setattr(github, "_request", boom)
    settings = _settings(tmp_path)

    records = await core.generate_images("a fox", settings=settings)

    assert records[0].remote_url is None
    assert "403" in records[0].publish_error
    assert (settings.output_dir / records[0].image_path).exists()
    assert len(gallery.load_records(settings)) == 1


async def test_publish_images_updates_existing_gallery_record(tmp_path, fake_client, fake_github):
    settings = _settings(tmp_path)
    [record] = await core.generate_images("a fox", settings=settings, publish=False)

    results = await core.publish_images([record.image_path], settings=settings)

    assert results[0].remote_url
    [stored] = gallery.load_records(settings)
    assert stored.remote_url == results[0].remote_url

    by_id = await core.publish_images([record.id], settings=settings)
    assert by_id[0].id == record.id


async def test_publish_images_rejects_unknown_path(tmp_path, fake_github):
    settings = _settings(tmp_path)
    with pytest.raises(GeminiGenerationError, match="Image not found"):
        await core.publish_images(["nope.png"], settings=settings)


async def test_publish_images_requires_configuration(tmp_path):
    settings = _settings(tmp_path, github_repo=None)
    with pytest.raises(GeminiGenerationError, match="not configured"):
        await core.publish_images(["x.png"], settings=settings)


async def test_publish_images_refuses_files_outside_images_dir(tmp_path, fake_github):
    settings = _settings(tmp_path)
    outside = tmp_path / "id_rsa.jpg"
    outside.write_bytes(JPEG_MAGIC)

    with pytest.raises(GeminiGenerationError, match="only images inside"):
        await core.publish_images([str(outside)], settings=settings)
    assert fake_github.puts == []


async def test_publish_images_refuses_cookie_cache_inside_output_dir(tmp_path, fake_github):
    settings = _settings(tmp_path)
    cookies = settings.output_dir / ".cookies"
    cookies.mkdir()
    (cookies / ".cached_cookies_x.json").write_text("{}", encoding="utf-8")

    with pytest.raises(GeminiGenerationError, match="only images inside"):
        await core.publish_images([".cookies/.cached_cookies_x.json"], settings=settings)
    assert fake_github.puts == []


async def test_publish_images_refuses_non_image_in_images_dir(tmp_path, fake_github):
    settings = _settings(tmp_path)
    (settings.output_dir / "images" / "notes.jpg").write_text("secret", encoding="utf-8")

    with pytest.raises(GeminiGenerationError, match="not an image"):
        await core.publish_images(["images/notes.jpg"], settings=settings)
    assert fake_github.puts == []


async def test_publish_images_refuses_tampered_manifest_path(tmp_path, fake_github):
    settings = _settings(tmp_path)
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(JPEG_MAGIC)
    tampered = core.GeneratedRecord(
        id="evil", prompt="x", model="", created_at="2026-01-01T00:00:00+00:00",
        image_path="../secret.jpg", kind="generate",
    )
    gallery.append_records([tampered], settings)

    with pytest.raises(GeminiGenerationError, match="only images inside"):
        await core.publish_images(["evil"], settings=settings)
    assert fake_github.puts == []
