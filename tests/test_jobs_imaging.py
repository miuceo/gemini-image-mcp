"""Tests for imaging (shape parsing / exact-size crop), text comparison, jobs and batches.

The Gemini side is faked: `core.generate_images` / `core.check_text` are monkeypatched, so no
network calls are made.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from PIL import Image

from gemini_image_mcp import core, imaging, jobs
from gemini_image_mcp.client import (
    GeminiAuthError,
    GeminiGenerationError,
    GeminiUsageLimitError,
)
from gemini_image_mcp.config import Settings


def _settings(tmp_path: Path, **overrides) -> Settings:
    output_dir = tmp_path / "output"
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    base = dict(
        secure_1psid="fake",
        secure_1psidts="fake",
        output_dir=output_dir,
        cookie_path=output_dir / ".cookies",
        default_model=None,
        timeout=120,
        proxy=None,
        refresh_interval=240.0,
    )
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# imaging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio", "size", "expected"),
    [
        ("3:4", None, (3, 4, None, None)),
        ("16/9", None, (16, 9, None, None)),
        ("1.91:1", None, (191, 100, None, None)),
        (None, "1080x1440", (3, 4, 1080, 1440)),
        ("1:1", "uzum", (3, 4, 1080, 1440)),
        (None, "story", (9, 16, 1080, 1920)),
    ],
)
def test_parse_shape(ratio, size, expected):
    shape = imaging.parse_shape(ratio, size)
    assert (shape.ratio_w, shape.ratio_h, shape.width, shape.height) == expected


def test_parse_shape_none_and_errors():
    assert imaging.parse_shape(None, "") is None
    with pytest.raises(GeminiGenerationError):
        imaging.parse_shape("tall", None)
    with pytest.raises(GeminiGenerationError):
        imaging.parse_shape(None, "99999x10")


@pytest.mark.parametrize("src_size", [(1024, 1024), (1600, 900), (700, 2000)])
def test_fit_to_shape_exact_size(tmp_path, src_size):
    src = tmp_path / "src.jpg"
    Image.new("RGB", src_size, "red").save(src, format="JPEG")
    dest = tmp_path / "out.jpg"

    final = imaging.fit_to_shape(src, dest, imaging.parse_shape(None, "1080x1440"))

    assert final == (1080, 1440)
    with Image.open(dest) as im:
        assert im.size == (1080, 1440)
        assert im.format == "JPEG"


def test_fit_to_shape_ratio_only_crops_without_resizing(tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (1000, 1000), "blue").save(src)
    dest = tmp_path / "out.png"

    assert imaging.fit_to_shape(src, dest, imaging.parse_shape("3:4", None)) == (750, 1000)


def test_thumbnail_is_small_jpeg(tmp_path):
    src = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "green").save(src)
    data = imaging.thumbnail_jpeg(src, max_side=400)
    assert data[:3] == bytes.fromhex("ffd8ff")
    from io import BytesIO

    with Image.open(BytesIO(data)) as im:
        assert max(im.size) == 400


def test_apply_shape_keeps_raw_original(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (1024, 1024), "white").save(img, format="JPEG")

    sizes = core._apply_shape([img], imaging.parse_shape(None, "uzum"))

    assert sizes == ["1080x1440"]
    assert (tmp_path / "raw" / "a.jpg").is_file()
    assert imaging.image_size(tmp_path / "raw" / "a.jpg") == (1024, 1024)


# ---------------------------------------------------------------------------
# text comparison
# ---------------------------------------------------------------------------


def test_compare_text_uzbek_apostrophes_equal():
    result = core.compare_text("O‘zbekiston bo'ylab yetkazib berish", "O'zbekiston boʻylab yetkazib berish")
    assert result["ok"] is True
    assert result["missing_words"] == []


def test_compare_text_flags_misspelling():
    result = core.compare_text("Chegirma 50% bepul yetkazib", "Chegrima 50% bepul yetkazib")
    assert result["ok"] is False
    assert result["missing_words"] == ["chegirma"]
    assert result["misspelled_as"] == {"chegirma": "chegrima"}


def test_compare_text_cyrillic():
    assert core.compare_text("Скидка 30%", "СКИДКА 30%")["ok"] is True


def test_safe_project():
    assert core.safe_project("Uzum cards / autumn") == "Uzum-cards-autumn"
    assert core.safe_project("../../etc") == "etc"
    assert core.safe_project("  ") is None


def test_record_roundtrip_with_metadata():
    rec = core.GeneratedRecord(
        id="x", prompt="p", model="m", created_at="t", image_path="images/x.jpg",
        kind="generate", project="shop", tags=["uz"], size="1080x1440", shape="1080x1440",
        text_check={"ok": True},
    )
    assert core.GeneratedRecord.from_dict(rec.to_dict()) == rec
    plain = core.GeneratedRecord(
        id="y", prompt="p", model="m", created_at="t", image_path="images/y.jpg", kind="generate"
    )
    assert "tags" not in plain.to_dict() and "project" not in plain.to_dict()


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def _fake_record(settings: Settings, name: str = "img") -> core.GeneratedRecord:
    path = settings.output_dir / "images" / f"{name}.png"
    Image.new("RGB", (64, 64), "red").save(path)
    return core.GeneratedRecord(
        id=name, prompt="p", model="m", created_at="t",
        image_path=f"images/{name}.png", kind="generate",
    )


@pytest.fixture
def fake_generate(monkeypatch):
    calls: list[dict] = []
    behaviour: list = []  # exceptions to raise, in order, before succeeding

    async def _gen(prompt, *, settings, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        await asyncio.sleep(0.01)
        if behaviour:
            exc = behaviour.pop(0)
            if exc is not None:
                raise exc
        return [_fake_record(settings, f"r{len(calls)}")]

    monkeypatch.setattr(core, "generate_images", _gen)
    return calls, behaviour


async def test_job_succeeds_and_is_persisted(tmp_path, fake_generate):
    settings = _settings(tmp_path)
    job, created = await jobs.start_job("generate", {"prompt": "a fox"}, settings)
    assert created

    done = await jobs.wait_job(settings, job["id"], 5)

    assert done["status"] == "succeeded"
    assert done["result"]["paths"][0].endswith("r1.png")
    assert (settings.output_dir / "jobs" / f"{job['id']}.json").is_file()


async def test_idempotency_key_returns_existing_job(tmp_path, fake_generate):
    calls, _ = fake_generate
    settings = _settings(tmp_path)
    first, created1 = await jobs.start_job("generate", {"prompt": "a"}, settings, idempotency_key="k1")
    await jobs.wait_job(settings, first["id"], 5)
    second, created2 = await jobs.start_job("generate", {"prompt": "a"}, settings, idempotency_key="k1")

    assert created1 and not created2
    assert second["id"] == first["id"]
    assert len(calls) == 1


async def test_failed_job_reports_error_and_can_rerun_with_same_key(tmp_path, fake_generate):
    calls, behaviour = fake_generate
    behaviour.append(GeminiGenerationError("no image"))
    settings = _settings(tmp_path)

    job, _ = await jobs.start_job("generate", {"prompt": "a"}, settings, idempotency_key="k2")
    failed = await jobs.wait_job(settings, job["id"], 5)
    assert failed["status"] == "failed" and "no image" in failed["error"]

    again, created = await jobs.start_job("generate", {"prompt": "a"}, settings, idempotency_key="k2")
    assert created and again["id"] == job["id"]
    assert (await jobs.wait_job(settings, job["id"], 5))["status"] == "succeeded"


async def test_job_timeout(tmp_path, monkeypatch):
    async def _slow(prompt, *, settings, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(core, "generate_images", _slow)
    settings = _settings(tmp_path, job_timeout=0.2)
    job, _ = await jobs.start_job("generate", {"prompt": "a"}, settings)
    done = await jobs.wait_job(settings, job["id"], 5)
    assert done["status"] == "failed" and "Timed out" in done["error"]


def test_stale_running_job_is_reported_interrupted(tmp_path):
    settings = _settings(tmp_path)
    jobs._write_json(
        jobs._jobs_dir(settings) / "abc123.json",
        {"id": "abc123", "status": "running", "heartbeat": time.time() - 999},
    )
    assert jobs.load_job(settings, "abc123")["status"] == "interrupted"


def test_job_ids_are_validated(tmp_path):
    with pytest.raises(Exception):
        jobs.load_job(_settings(tmp_path), "../../secret")


async def test_text_check_attached_to_job(tmp_path, fake_generate, monkeypatch):
    async def _check(path, expected, *, settings, **kw):
        return core.compare_text(expected, "Chegrima")

    monkeypatch.setattr(core, "check_text", _check)
    settings = _settings(tmp_path)
    job, _ = await jobs.start_job("generate", {"prompt": "a"}, settings, expected_text="Chegirma")
    done = await jobs.wait_job(settings, job["id"], 5)
    assert done["result"]["records"][0]["text_check"]["ok"] is False


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------


async def _wait_batch(settings, batch_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        b = jobs.load_batch(settings, batch_id)
        if b["status"] not in jobs.ACTIVE:
            return b
        await asyncio.sleep(0.02)
    raise AssertionError("batch did not finish")


async def test_batch_runs_items_with_defaults(tmp_path, fake_generate):
    calls, _ = fake_generate
    settings = _settings(tmp_path)
    batch = await jobs.start_batch(
        [{"id": "a", "prompt": "one"}, {"id": "b", "prompt": "two"}],
        settings,
        defaults={"project": "shop", "tags": ["uz"], "size": "uzum"},
        delay_seconds=0,
    )
    done = await _wait_batch(settings, batch["id"])

    assert done["status"] == "completed"
    assert [i["status"] for i in done["items"]] == ["done", "done"]
    assert calls[0]["project"] == "shop" and calls[0]["size"] == "uzum"
    assert "uz" in calls[0]["tags"] and f"batch:{batch['id']}" in calls[0]["tags"]


async def test_batch_retries_then_fails_item_and_continues(tmp_path, fake_generate, monkeypatch):
    calls, behaviour = fake_generate
    behaviour.extend([GeminiGenerationError("x"), GeminiGenerationError("x")])
    monkeypatch.setattr(jobs, "_sleep", _no_sleep)
    settings = _settings(tmp_path)
    batch = await jobs.start_batch(
        [{"id": "a", "prompt": "one"}, {"id": "b", "prompt": "two"}],
        settings, max_retries=1, delay_seconds=0,
    )
    done = await _wait_batch(settings, batch["id"])

    assert [i["status"] for i in done["items"]] == ["failed", "done"]
    assert done["status"] == "completed_with_errors"


async def test_batch_pauses_on_auth_error_and_resumes(tmp_path, fake_generate):
    calls, behaviour = fake_generate
    behaviour.extend([None, GeminiAuthError("cookies expired")])
    settings = _settings(tmp_path)
    batch = await jobs.start_batch(
        [{"id": "a", "prompt": "1"}, {"id": "b", "prompt": "2"}, {"id": "c", "prompt": "3"}],
        settings, delay_seconds=0,
    )
    paused = await _wait_batch(settings, batch["id"])
    assert paused["status"] == "paused"
    assert [i["status"] for i in paused["items"]] == ["done", "pending", "pending"]

    await jobs.resume_batch(settings, batch["id"])
    done = await _wait_batch(settings, batch["id"])
    assert done["status"] == "completed"
    assert len(calls) == 4  # item "a" was not regenerated


async def test_batch_backs_off_on_rate_limit(tmp_path, fake_generate, monkeypatch):
    calls, behaviour = fake_generate
    behaviour.append(GeminiUsageLimitError("slow down"))
    waits: list[float] = []

    async def _record_sleep(seconds, *a, **k):
        waits.append(seconds)

    monkeypatch.setattr(jobs, "_sleep", _record_sleep)
    settings = _settings(tmp_path)
    batch = await jobs.start_batch([{"id": "a", "prompt": "1"}], settings, delay_seconds=0)
    done = await _wait_batch(settings, batch["id"])

    assert done["items"][0]["status"] == "done"
    assert 60 in waits


async def test_batch_rejects_bad_items(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(Exception, match="prompt"):
        await jobs.start_batch([{"id": "a"}], settings)
    with pytest.raises(Exception, match="Duplicate"):
        await jobs.start_batch([{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}], settings)
    with pytest.raises(Exception, match="size"):
        await jobs.start_batch([{"prompt": "x", "size": "huge"}], settings)


_real_sleep = asyncio.sleep


async def _no_sleep(seconds, *a, **k):
    await _real_sleep(0)
