"""Tests for gemini_image_mcp.core.generate_images / edit_images with the Gemini client mocked.

No network calls are made: `core.get_client` is monkeypatched to return a fake client, and
`GeneratedImage.save` is monkeypatched to write a small real PNG to disk (so downstream
suffix-sniffing / path logic runs for real) instead of hitting the network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from gemini_webapi.types import GeneratedImage

from gemini_image_mcp import core
from gemini_image_mcp.client import GeminiGenerationError
from gemini_image_mcp.config import Settings

PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 8


def _settings(tmp_path: Path) -> Settings:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    return Settings(
        secure_1psid="fake-psid",
        secure_1psidts="fake-psidts",
        output_dir=output_dir,
        cookie_path=output_dir / ".cookies",
        default_model=None,
        timeout=120,
        proxy=None,
        refresh_interval=240.0,
    )


async def _fake_save(self, path, filename=None, verbose=False, **kwargs) -> str:
    dest = Path(path) / filename
    dest.write_bytes(PNG_MAGIC)
    return str(dest)


@pytest.fixture
def fake_save(monkeypatch):
    monkeypatch.setattr(GeneratedImage, "save", _fake_save)


@pytest.fixture
def fake_client(monkeypatch):
    client = Mock()
    client.generate_content = AsyncMock()
    client.start_chat = Mock()
    client.list_models = Mock(return_value=[])
    monkeypatch.setattr(core, "get_client", AsyncMock(return_value=client))
    return client


def _gen_image(url: str = "https://example.com/image.png") -> GeneratedImage:
    return GeneratedImage(url=url, title="[Image]", alt="")


# ---------------------------------------------------------------------------
# generate_images
# ---------------------------------------------------------------------------


async def test_generate_images_builds_records_correctly(tmp_path, fake_client, fake_save):
    settings = _settings(tmp_path)
    fake_client.generate_content.return_value = SimpleNamespace(
        images=[_gen_image()], text="here you go"
    )

    records = await core.generate_images("a fox in the snow", settings=settings)

    assert len(records) == 1
    record = records[0]
    assert record.kind == "generate"
    assert record.prompt == "a fox in the snow"  # original prompt, not the shaped one
    assert record.model == "default"
    assert record.source_images == []
    # image_path is relative (posix-style) to output_dir, not absolute
    assert not Path(record.image_path).is_absolute()
    assert record.image_path.startswith("images/")
    assert (settings.output_dir / record.image_path).exists()
    # id matches the saved file's stem
    assert record.id == Path(record.image_path).stem

    # the shaped (directive-prepended) prompt is what actually gets sent to the client
    fake_client.generate_content.assert_awaited_once()
    sent_prompt = fake_client.generate_content.await_args.args[0]
    assert sent_prompt == "Generate an image: a fox in the snow"


async def test_generate_images_persists_to_manifest_and_gallery(tmp_path, fake_client, fake_save):
    settings = _settings(tmp_path)
    fake_client.generate_content.return_value = SimpleNamespace(
        images=[_gen_image()], text=""
    )

    await core.generate_images("draw a dragon", settings=settings)

    assert (settings.output_dir / "manifest.json").exists()
    assert (settings.output_dir / "gallery.html").exists()


async def test_generate_images_uses_explicit_model_over_default(tmp_path, fake_client, fake_save):
    settings = _settings(tmp_path)
    fake_client.generate_content.return_value = SimpleNamespace(
        images=[_gen_image()], text=""
    )

    records = await core.generate_images("draw a dragon", model="gemini-explicit", settings=settings)

    assert records[0].model == "gemini-explicit"
    assert fake_client.generate_content.await_args.kwargs["model"] == "gemini-explicit"


async def test_generate_images_no_generated_images_raises_with_model_text(
    tmp_path, fake_client, fake_save
):
    settings = _settings(tmp_path)
    fake_client.generate_content.return_value = SimpleNamespace(
        images=[], text="I can't generate that kind of image."
    )

    with pytest.raises(GeminiGenerationError) as excinfo:
        await core.generate_images("draw something blocked", settings=settings)

    assert "I can't generate that kind of image." in str(excinfo.value)


# ---------------------------------------------------------------------------
# edit_images
# ---------------------------------------------------------------------------


async def test_edit_images_missing_source_raises_clear_error(tmp_path, fake_client, fake_save):
    settings = _settings(tmp_path)

    with pytest.raises(GeminiGenerationError) as excinfo:
        await core.edit_images(
            "make it nighttime", [str(tmp_path / "does_not_exist.png")], settings=settings
        )

    assert "does_not_exist.png" in str(excinfo.value)
    fake_client.start_chat.assert_not_called()


async def test_edit_images_builds_records_and_copies_external_source(
    tmp_path, fake_client, fake_save
):
    settings = _settings(tmp_path)
    source = tmp_path / "my_photo.png"
    source.write_bytes(PNG_MAGIC)

    fake_chat = Mock()
    fake_chat.send_message = AsyncMock(
        return_value=SimpleNamespace(images=[_gen_image()], text="")
    )
    fake_client.start_chat.return_value = fake_chat

    records = await core.edit_images("make it nighttime", [str(source)], settings=settings)

    assert len(records) == 1
    record = records[0]
    assert record.kind == "edit"
    assert len(record.source_images) == 1
    # the external source image was copied into output_dir/images and referenced relatively
    assert record.source_images[0].startswith("images/")
    assert (settings.output_dir / record.source_images[0]).exists()
    assert not Path(record.image_path).is_absolute()


async def test_edit_images_no_edited_images_raises_with_model_text(
    tmp_path, fake_client, fake_save
):
    settings = _settings(tmp_path)
    source = tmp_path / "my_photo.png"
    source.write_bytes(PNG_MAGIC)

    fake_chat = Mock()
    fake_chat.send_message = AsyncMock(
        return_value=SimpleNamespace(images=[], text="refused: unsafe content")
    )
    fake_client.start_chat.return_value = fake_chat

    with pytest.raises(GeminiGenerationError) as excinfo:
        await core.edit_images("make it weird", [str(source)], settings=settings)

    assert "refused: unsafe content" in str(excinfo.value)


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


async def test_list_models_maps_fields(tmp_path, fake_client):
    settings = _settings(tmp_path)
    fake_client.list_models.return_value = [
        SimpleNamespace(display_name="Gemini Test", model_name="gemini-test"),
    ]

    models = await core.list_models(settings=settings)

    assert models == [{"display_name": "Gemini Test", "model_name": "gemini-test"}]
