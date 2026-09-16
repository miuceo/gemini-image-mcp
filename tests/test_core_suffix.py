"""Tests for gemini_image_mcp.core's file-format sniffing and prompt shaping helpers."""

from __future__ import annotations

import pytest

from gemini_image_mcp.core import _fix_suffix, _shape_generate_prompt, _sniff_suffix

PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 8
JPEG_MAGIC = bytes.fromhex("ffd8ff") + b"\x00" * 13
GIF87_MAGIC = b"GIF87a" + b"\x00" * 10
GIF89_MAGIC = b"GIF89a" + b"\x00" * 10
WEBP_MAGIC = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 4
UNKNOWN_MAGIC = b"\x00" * 16


@pytest.mark.parametrize(
    "magic, expected_suffix",
    [
        (PNG_MAGIC, ".png"),
        (JPEG_MAGIC, ".jpg"),
        (GIF87_MAGIC, ".gif"),
        (GIF89_MAGIC, ".gif"),
        (WEBP_MAGIC, ".webp"),
    ],
)
def test_sniff_suffix_detects_real_format(tmp_path, magic, expected_suffix):
    path = tmp_path / "image.png"  # deliberately wrong extension for jpeg/gif/webp cases
    path.write_bytes(magic)
    assert _sniff_suffix(path) == expected_suffix


def test_sniff_suffix_falls_back_to_existing_suffix_for_unknown_format(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(UNKNOWN_MAGIC)
    assert _sniff_suffix(path) == ".png"


def test_sniff_suffix_falls_back_to_png_when_no_suffix_and_unknown_format(tmp_path):
    path = tmp_path / "image"
    path.write_bytes(UNKNOWN_MAGIC)
    assert _sniff_suffix(path) == ".png"


def test_fix_suffix_renames_jpeg_reported_as_png(tmp_path):
    """The real-world case this matters for: Gemini returns JPEG bytes despite us asking for .png."""
    path = tmp_path / "abc123.png"
    path.write_bytes(JPEG_MAGIC)

    fixed = _fix_suffix(path)

    assert fixed.suffix == ".jpg"
    assert fixed.exists()
    assert not path.exists()
    assert fixed.read_bytes() == JPEG_MAGIC


def test_fix_suffix_is_a_noop_when_extension_already_correct(tmp_path):
    path = tmp_path / "abc123.png"
    path.write_bytes(PNG_MAGIC)

    fixed = _fix_suffix(path)

    assert fixed == path
    assert path.exists()


def test_fix_suffix_renames_gif(tmp_path):
    path = tmp_path / "abc123.png"
    path.write_bytes(GIF89_MAGIC)

    fixed = _fix_suffix(path)

    assert fixed.suffix == ".gif"
    assert fixed.exists()
    assert not path.exists()


def test_fix_suffix_renames_webp(tmp_path):
    path = tmp_path / "abc123.png"
    path.write_bytes(WEBP_MAGIC)

    fixed = _fix_suffix(path)

    assert fixed.suffix == ".webp"
    assert fixed.exists()


# ---------------------------------------------------------------------------
# _shape_generate_prompt
# ---------------------------------------------------------------------------


def test_shape_generate_prompt_prepends_directive_when_absent():
    result = _shape_generate_prompt("a cat wearing a hat")
    assert result == "Generate an image: a cat wearing a hat"


@pytest.mark.parametrize(
    "prompt",
    [
        "Generate an image of a cat",
        "generate a picture of a dog",
        "create an image of a sunset",
        "draw a dragon",
        "Paint a landscape",
        "imagine a spaceship",
        "a picture of a mountain",
        "an illustration of a robot",
        "sketch a portrait",
    ],
)
def test_shape_generate_prompt_not_duplicated_when_already_a_generation_request(prompt):
    result = _shape_generate_prompt(prompt)
    assert result == prompt
