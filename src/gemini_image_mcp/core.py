"""Generation core for gemini_image_mcp.

This module contains the actual image generation/editing logic and MUST NOT import
anything from `mcp` - it is also the entrypoint a future Telegram bot will import directly.
"""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from gemini_webapi.types import GeneratedImage

from .client import GeminiGenerationError, get_client, map_library_error
from .config import Settings, get_settings

# Verbs that already read as an explicit request to generate/create/draw an image. If the
# prompt doesn't obviously ask for one, we prepend a directive so gemini-webapi's backend
# actually returns `GeneratedImage` results instead of just text or web search images.
_GENERATE_VERBS = (
    "generate",
    "create an image",
    "create a picture",
    "draw",
    "paint",
    "render an image",
    "make an image",
    "imagine",
    "picture of",
    "illustration of",
    "sketch",
)

_GENERATION_DIRECTIVE = "Generate an image: "


def _shape_generate_prompt(prompt: str) -> str:
    """Prepend a generation directive unless the prompt already reads as a generation request."""
    lowered = prompt.strip().lower()
    if any(verb in lowered for verb in _GENERATE_VERBS):
        return prompt
    return f"{_GENERATION_DIRECTIVE}{prompt}"


@dataclass
class GeneratedRecord:
    """A single generated or edited image, as recorded in the gallery."""

    id: str
    prompt: str
    model: str
    created_at: str
    image_path: str
    kind: Literal["generate", "edit"]
    source_images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "model": self.model,
            "created_at": self.created_at,
            "image_path": self.image_path,
            "kind": self.kind,
            "source_images": list(self.source_images),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedRecord":
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            model=data["model"],
            created_at=data["created_at"],
            image_path=data["image_path"],
            kind=data["kind"],
            source_images=list(data.get("source_images", [])),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Gemini returns whatever format it likes (often JPEG) regardless of the filename we ask
# for, so sniff the magic bytes and correct the extension rather than shipping a .png that
# is really a JPEG - downstream consumers (Telegram, image tooling) trust the extension.
_MAGIC_SUFFIXES: tuple[tuple[bytes, str], ...] = (
    (bytes.fromhex("89504e470d0a1a0a"), ".png"),
    (bytes.fromhex("ffd8ff"), ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)


def _sniff_suffix(path: Path) -> str:
    """Return the correct file suffix for `path` based on its magic bytes."""
    with path.open("rb") as fh:
        header = fh.read(16)
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return ".webp"
    for magic, suffix in _MAGIC_SUFFIXES:
        if header.startswith(magic):
            return suffix
    return path.suffix or ".png"


def _fix_suffix(path: Path) -> Path:
    """Rename `path` to match its actual image format, returning the final path."""
    correct = _sniff_suffix(path)
    if path.suffix.lower() == correct:
        return path
    fixed = path.with_suffix(correct)
    path.replace(fixed)
    return fixed


def _relative_to_output(path: Path, settings: Settings) -> str:
    """Path relative to `output_dir`, tolerating Windows casing/short-name mismatches."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(settings.output_dir).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, settings.output_dir)).as_posix()


async def _save_generated_images(
    generated: list[GeneratedImage],
    settings: Settings,
) -> list[Path]:
    """Save each `GeneratedImage` under `settings.output_dir / "images"` and return real paths.

    `GeneratedImage.save()` may not always honor the exact filename passed in (e.g. it derives
    an extension from the response content type), so we capture its return value as the source
    of truth and verify the file actually landed on disk.
    """
    images_dir = settings.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for image in generated:
        record_id = uuid.uuid4().hex
        filename = f"{record_id}.png"
        try:
            saved_path_str = await image.save(
                path=str(images_dir),
                filename=filename,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - translated at this boundary
            raise map_library_error(exc) from exc

        saved_path = Path(saved_path_str)
        if not saved_path.exists():
            raise GeminiGenerationError(
                f"Gemini reported saving an image to '{saved_path}', but the file does not exist."
            )
        saved_paths.append(_fix_suffix(saved_path))

    return saved_paths


def _model_label(model: str | None, settings: Settings) -> str:
    return model or settings.default_model or "default"


async def generate_images(
    prompt: str,
    *,
    model: str | None = None,
    settings: Settings | None = None,
) -> list[GeneratedRecord]:
    """Generate one or more images from a text prompt using the Gemini web app.

    Parameters
    ----------
    prompt: `str`
        The user's image prompt. A generation directive is prepended automatically unless
        the prompt already reads as an explicit request to generate/draw/create an image.
    model: `str | None`, optional
        Model name, alias, or id to use. Defaults to `settings.default_model`, or the
        account's own default model if that is also unset.
    settings: `Settings | None`, optional
        Configuration to use. Defaults to the process-wide settings.

    Returns
    -------
    `list[GeneratedRecord]`
        One record per generated image, already appended to the gallery.

    Raises
    ------
    `GeminiAuthError`
        If Gemini session cookies are missing or invalid.
    `GeminiGenerationError`
        If no images were generated (the exception message includes the model's text
        response, which usually explains the refusal - region/age restriction, safety block).

    """
    settings = settings or get_settings()
    client = await get_client(settings)

    effective_model = model or settings.default_model
    shaped_prompt = _shape_generate_prompt(prompt)

    try:
        output = await client.generate_content(shaped_prompt, model=effective_model)
    except Exception as exc:  # noqa: BLE001 - translated at this boundary
        raise map_library_error(exc) from exc

    generated_only = [img for img in output.images if isinstance(img, GeneratedImage)]
    if not generated_only:
        raise GeminiGenerationError(
            "Gemini did not return any generated images for this prompt. "
            f"Model response: {output.text!r}"
        )

    saved_paths = await _save_generated_images(generated_only, settings)

    created_at = _now_iso()
    model_label = _model_label(effective_model, settings)
    records = [
        GeneratedRecord(
            id=path.stem,
            prompt=prompt,
            model=model_label,
            created_at=created_at,
            image_path=_relative_to_output(path, settings),
            kind="generate",
            source_images=[],
        )
        for path in saved_paths
    ]

    from . import gallery

    gallery.append_records(records, settings)
    gallery.render(settings)

    return records


async def edit_images(
    prompt: str,
    image_paths: list[str],
    *,
    model: str | None = None,
    settings: Settings | None = None,
) -> list[GeneratedRecord]:
    """Edit one or more existing images using a text prompt, via a Gemini chat session.

    Parameters
    ----------
    prompt: `str`
        Instructions describing the desired edit.
    image_paths: `list[str]`
        Paths to the source images to edit. Each must exist on disk.
    model: `str | None`, optional
        Model name, alias, or id to use. Defaults to `settings.default_model`.
    settings: `Settings | None`, optional
        Configuration to use. Defaults to the process-wide settings.

    Raises
    ------
    `GeminiGenerationError`
        If a source image path does not exist, or if no edited images come back.
    `GeminiAuthError`
        If Gemini session cookies are missing or invalid.

    """
    settings = settings or get_settings()

    resolved_inputs: list[Path] = []
    for raw_path in image_paths:
        candidate = Path(raw_path)
        if not candidate.exists():
            raise GeminiGenerationError(f"Source image not found: '{raw_path}'")
        resolved_inputs.append(candidate)

    images_dir = settings.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copy any source image not already inside output_dir so the gallery can reference it
    # with a relative path, same as generated images.
    source_relative_paths: list[str] = []
    for candidate in resolved_inputs:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(settings.output_dir)
            already_inside = True
        except ValueError:
            already_inside = False

        if already_inside:
            source_relative_paths.append(_relative_to_output(resolved, settings))
        else:
            dest = images_dir / f"{uuid.uuid4().hex}{candidate.suffix or '.png'}"
            shutil.copy2(candidate, dest)
            source_relative_paths.append(_relative_to_output(dest, settings))

    client = await get_client(settings)
    effective_model = model or settings.default_model
    chat = client.start_chat(model=effective_model)

    try:
        output = await chat.send_message(prompt, files=[str(p) for p in resolved_inputs])
    except Exception as exc:  # noqa: BLE001 - translated at this boundary
        raise map_library_error(exc) from exc

    generated_only = [img for img in output.images if isinstance(img, GeneratedImage)]
    if not generated_only:
        raise GeminiGenerationError(
            "Gemini did not return any edited images for this prompt. "
            f"Model response: {output.text!r}"
        )

    saved_paths = await _save_generated_images(generated_only, settings)

    created_at = _now_iso()
    model_label = _model_label(effective_model, settings)
    records = [
        GeneratedRecord(
            id=path.stem,
            prompt=prompt,
            model=model_label,
            created_at=created_at,
            image_path=_relative_to_output(path, settings),
            kind="edit",
            source_images=source_relative_paths,
        )
        for path in saved_paths
    ]

    from . import gallery

    gallery.append_records(records, settings)
    gallery.render(settings)

    return records


async def list_models(settings: Settings | None = None) -> list[dict[str, str]]:
    """List models available to the current Gemini account.

    Returns
    -------
    `list[dict[str, str]]`
        Each entry has `display_name` and `model_name` keys.

    """
    settings = settings or get_settings()
    client = await get_client(settings)

    models = client.list_models() or []
    return [{"display_name": m.display_name, "model_name": m.model_name} for m in models]
