"""Generation core for gemini_image_mcp.

This module contains the actual image generation/editing logic and MUST NOT import
anything from `mcp` - it is also the entrypoint a future Telegram bot will import directly.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from gemini_webapi.types import GeneratedImage

from . import imaging
from .client import (
    GeminiGenerationError,
    GeminiImageError,
    GeminiTimeoutError,
    get_client,
    map_library_error,
    reset_client,
)
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
    # Public URL of the copy published to GitHub, if any.
    remote_url: str | None = None
    # Why publishing failed for this call, if it did. Transient: not written to the manifest.
    publish_error: str | None = field(default=None, compare=False)
    # Organisation: a project name (also the images/ subfolder) and free-form tags.
    project: str | None = None
    tags: list[str] = field(default_factory=list)
    # Final pixel size ("1080x1440") and the shape that was requested ("3:4" / "1080x1440").
    size: str | None = None
    shape: str | None = None
    # Result of an OCR check of rendered text, if one was requested.
    text_check: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "prompt": self.prompt,
            "model": self.model,
            "created_at": self.created_at,
            "image_path": self.image_path,
            "kind": self.kind,
            "source_images": list(self.source_images),
        }
        if self.remote_url:
            data["remote_url"] = self.remote_url
        if self.project:
            data["project"] = self.project
        if self.tags:
            data["tags"] = list(self.tags)
        if self.size:
            data["size"] = self.size
        if self.shape:
            data["shape"] = self.shape
        if self.text_check:
            data["text_check"] = self.text_check
        return data

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
            remote_url=data.get("remote_url") or None,
            project=data.get("project") or None,
            tags=[str(t) for t in data.get("tags", []) if t],
            size=data.get("size") or None,
            shape=data.get("shape") or None,
            text_check=data.get("text_check") or None,
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


def _image_suffix(path: Path) -> str | None:
    """The file suffix matching `path`'s magic bytes, or None if it is not a known image."""
    with path.open("rb") as fh:
        header = fh.read(16)
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return ".webp"
    for magic, suffix in _MAGIC_SUFFIXES:
        if header.startswith(magic):
            return suffix
    return None


def _sniff_suffix(path: Path) -> str:
    """Return the correct file suffix for `path` based on its magic bytes."""
    return _image_suffix(path) or path.suffix or ".png"


# Upper bounds on tool arguments, which may come from a confused or prompt-injected agent.
_MAX_PROMPT_CHARS = 8000
_MAX_EDIT_IMAGES = 10
_MAX_PUBLISH_IMAGES = 50


def _check_prompt(prompt: str) -> None:
    if not prompt or not prompt.strip():
        raise GeminiGenerationError("The prompt is empty.")
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise GeminiGenerationError(
            f"The prompt is {len(prompt)} characters long; the limit is {_MAX_PROMPT_CHARS}."
        )


# Source images are uploaded to Gemini as-is; cap them so a stray path can't ship a huge file.
_MAX_SOURCE_IMAGE_BYTES = 20 * 1024 * 1024


def _is_within(path: Path, directory: Path) -> bool:
    """Whether `path` resolves to a location inside `directory` (case-insensitive on Windows)."""
    target = os.path.normcase(str(path.resolve()))
    root = os.path.normcase(str(directory.resolve()))
    try:
        return os.path.commonpath([target, root]) == root
    except ValueError:  # different drives on Windows
        return False


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


# Downloading one generated image from Google's CDN; generous, but never unbounded.
_SAVE_TIMEOUT = 90

_semaphore: asyncio.Semaphore | None = None


def _generation_slot(settings: Settings) -> asyncio.Semaphore:
    """Process-wide cap on concurrent Gemini generations (GEMINI_MAX_CONCURRENT)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.max_concurrent)
    return _semaphore


def safe_project(project: str | None) -> str | None:
    """A project name usable as a single folder name, or None."""
    if not project or not project.strip():
        return None
    slug = re.sub(r"[^\w.-]+", "-", project.strip(), flags=re.UNICODE).strip(".-")
    return slug[:60] or None


def _clean_tags(tags: list[str] | None) -> list[str]:
    seen: list[str] = []
    for tag in tags or []:
        t = str(tag).strip()[:40]
        if t and t not in seen:
            seen.append(t)
    return seen[:20]


async def _save_generated_images(
    generated: list[GeneratedImage],
    settings: Settings,
    subdir: str | None = None,
) -> list[Path]:
    """Save each `GeneratedImage` under `settings.output_dir / "images"` and return real paths.

    `GeneratedImage.save()` may not always honor the exact filename passed in (e.g. it derives
    an extension from the response content type), so we capture its return value as the source
    of truth and verify the file actually landed on disk.
    """
    images_dir = settings.output_dir / "images"
    if subdir:
        images_dir = images_dir / subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for image in generated:
        record_id = uuid.uuid4().hex
        filename = f"{record_id}.png"
        try:
            saved_path_str = await asyncio.wait_for(
                image.save(
                    path=str(images_dir),
                    filename=filename,
                    verbose=False,
                ),
                timeout=_SAVE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise GeminiTimeoutError(
                f"Downloading a generated image took longer than {_SAVE_TIMEOUT}s."
            ) from None
        except Exception as exc:  # noqa: BLE001 - translated at this boundary
            raise map_library_error(exc) from exc

        saved_path = Path(saved_path_str)
        if not saved_path.exists():
            raise GeminiGenerationError(
                f"Gemini reported saving an image to '{saved_path}', but the file does not exist."
            )
        saved_paths.append(_fix_suffix(saved_path))

    return saved_paths


def _apply_shape(paths: list[Path], shape: imaging.Shape | None) -> list[str | None]:
    """Crop/resize each image to `shape` in place (raw original kept in `raw/`).

    Returns each image's final "WxH" size (None when it can't be read).
    """
    sizes: list[str | None] = []
    for path in paths:
        if shape is not None:
            raw_dir = path.parent / "raw"
            raw_dir.mkdir(exist_ok=True)
            raw_copy = raw_dir / path.name
            shutil.copy2(path, raw_copy)
            try:
                imaging.fit_to_shape(raw_copy, path, shape)
            except Exception as exc:  # noqa: BLE001 - keep the raw image rather than fail
                shutil.copy2(raw_copy, path)
                raise GeminiGenerationError(
                    f"Post-processing to {shape.label} failed ({exc}); the original is at {raw_copy}"
                ) from exc
        dims = imaging.image_size(path)
        sizes.append(f"{dims[0]}x{dims[1]}" if dims else None)
    return sizes


def _looks_signed_out(text: str | None) -> bool:
    """Whether Gemini's text reply says the session is signed out (stale cookies)."""
    return "signed out" in (text or "").lower()


def _model_label(model: str | None, settings: Settings) -> str:
    return model or settings.default_model or "default"


def _should_publish(publish: bool | None, settings: Settings) -> bool:
    if not settings.github_enabled:
        return False
    return settings.github_auto_publish if publish is None else publish


async def _publish_records(records: list[GeneratedRecord], settings: Settings) -> None:
    """Upload each record's image to GitHub, setting `remote_url` or `publish_error`.

    Never raises for a publishing failure: the image is already saved locally, so a GitHub
    problem is reported alongside the result instead of failing the whole generation.
    """
    from . import github

    for record in records:
        local_path = settings.output_dir / record.image_path
        remote_path = github.remote_path_for(local_path, record.prompt, record.id, settings)
        message = github.commit_message_for(record.prompt, record.id, settings)
        try:
            record.remote_url = await github.publish_file(
                local_path, remote_path, message, settings
            )
            record.publish_error = None
        except github.GitHubPublishError as exc:
            record.publish_error = str(exc)


async def generate_images(
    prompt: str,
    *,
    model: str | None = None,
    settings: Settings | None = None,
    publish: bool | None = None,
    aspect_ratio: str | None = None,
    size: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
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
    publish: `bool | None`, optional
        Whether to upload the images to the configured GitHub repo. `None` follows
        `settings.github_auto_publish`; ignored when GitHub publishing is not configured.
    aspect_ratio / size: `str | None`, optional
        Requested shape ("3:4", "1080x1440" or a preset such as "uzum"); the ratio is added
        to the prompt and the result is center-cropped/resized to match exactly.
    project / tags: optional
        Gallery metadata; `project` is also the subfolder under `images/`.

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
    _check_prompt(prompt)
    shape = imaging.parse_shape(aspect_ratio, size)
    subdir = safe_project(project)
    settings = settings or get_settings()
    client = await get_client(settings)

    effective_model = model or settings.default_model
    shaped_prompt = _shape_generate_prompt(prompt)
    if shape is not None:
        shaped_prompt = f"{shaped_prompt}\n\n{shape.prompt_hint()}"

    async with _generation_slot(settings):
        try:
            output = await client.generate_content(shaped_prompt, model=effective_model)
            if not output.images and _looks_signed_out(output.text):
                # Session went stale mid-process: rebuild the client (re-reading the browser's
                # cookies when configured) and retry once.
                await reset_client()
                client = await get_client(settings)
                output = await client.generate_content(shaped_prompt, model=effective_model)
        except GeminiImageError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated at this boundary
            raise map_library_error(exc) from exc

        generated_only = [img for img in output.images if isinstance(img, GeneratedImage)]
        if not generated_only:
            raise GeminiGenerationError(
                "Gemini did not return any generated images for this prompt. "
                f"Model response: {output.text!r}"
            )

        saved_paths = await _save_generated_images(generated_only, settings, subdir)

    sizes = await asyncio.to_thread(_apply_shape, saved_paths, shape)

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
            project=subdir,
            tags=_clean_tags(tags),
            size=final_size,
            shape=shape.label if shape else None,
        )
        for path, final_size in zip(saved_paths, sizes)
    ]

    if _should_publish(publish, settings):
        await _publish_records(records, settings)

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
    publish: bool | None = None,
    aspect_ratio: str | None = None,
    size: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
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
    publish, aspect_ratio, size, project, tags: optional
        Same as for `generate_images`.

    Raises
    ------
    `GeminiGenerationError`
        If a source image path does not exist, or if no edited images come back.
    `GeminiAuthError`
        If Gemini session cookies are missing or invalid.

    """
    _check_prompt(prompt)
    if not image_paths:
        raise GeminiGenerationError("No source images were given to edit.")
    if len(image_paths) > _MAX_EDIT_IMAGES:
        raise GeminiGenerationError(
            f"Too many source images ({len(image_paths)}); the limit is {_MAX_EDIT_IMAGES}."
        )
    shape = imaging.parse_shape(aspect_ratio, size)
    subdir = safe_project(project)
    settings = settings or get_settings()

    # Tool arguments can come from a prompt-injected agent, so only real image files are
    # accepted: anything else (.env, keys, documents) would otherwise be uploaded to Google.
    resolved_inputs: list[Path] = []
    input_suffixes: list[str] = []
    for raw_path in image_paths:
        candidate = Path(raw_path)
        if not candidate.is_file():
            raise GeminiGenerationError(f"Source image not found: '{raw_path}'")
        if candidate.stat().st_size > _MAX_SOURCE_IMAGE_BYTES:
            raise GeminiGenerationError(
                f"Source image is larger than {_MAX_SOURCE_IMAGE_BYTES // (1024 * 1024)} MB: "
                f"'{raw_path}'"
            )
        suffix = _image_suffix(candidate)
        if suffix is None:
            raise GeminiGenerationError(
                f"Not a supported image (PNG, JPEG, WEBP or GIF): '{raw_path}'"
            )
        resolved_inputs.append(candidate)
        input_suffixes.append(suffix)

    client = await get_client(settings)
    effective_model = model or settings.default_model
    message = prompt if shape is None else f"{prompt}\n\n{shape.prompt_hint()}"

    files = [str(p) for p in resolved_inputs]
    async with _generation_slot(settings):
        chat = client.start_chat(model=effective_model)
        try:
            output = await chat.send_message(message, files=files)
            if not output.images and _looks_signed_out(output.text):
                await reset_client()
                client = await get_client(settings)
                chat = client.start_chat(model=effective_model)
                output = await chat.send_message(message, files=files)
        except GeminiImageError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated at this boundary
            raise map_library_error(exc) from exc

        generated_only = [img for img in output.images if isinstance(img, GeneratedImage)]
        if not generated_only:
            raise GeminiGenerationError(
                "Gemini did not return any edited images for this prompt. "
                f"Model response: {output.text!r}"
            )
        saved_paths = await _save_generated_images(generated_only, settings, subdir)

    # Only now, with an edit in hand, copy any source image not already inside output_dir so
    # the gallery can reference it with a relative path (a failed edit leaves no stray copies).
    images_dir = settings.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    source_relative_paths: list[str] = []
    for candidate, suffix in zip(resolved_inputs, input_suffixes):
        resolved = candidate.resolve()
        if _is_within(resolved, settings.output_dir):
            source_relative_paths.append(_relative_to_output(resolved, settings))
        else:
            dest = images_dir / f"{uuid.uuid4().hex}{suffix}"
            shutil.copy2(candidate, dest)
            source_relative_paths.append(_relative_to_output(dest, settings))

    sizes = await asyncio.to_thread(_apply_shape, saved_paths, shape)

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
            project=subdir,
            tags=_clean_tags(tags),
            size=final_size,
            shape=shape.label if shape else None,
        )
        for path, final_size in zip(saved_paths, sizes)
    ]

    if _should_publish(publish, settings):
        await _publish_records(records, settings)

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
    try:
        client = await asyncio.wait_for(get_client(settings), timeout=settings.init_timeout + 5)
    except asyncio.TimeoutError:
        raise GeminiTimeoutError(
            "Timed out signing in to Gemini while listing models. Run `doctor` for details."
        ) from None

    models = client.list_models() or []
    return [{"display_name": m.display_name, "model_name": m.model_name} for m in models]


async def publish_images(
    image_paths: list[str], settings: Settings | None = None
) -> list[GeneratedRecord]:
    """Publish existing images to the configured GitHub repo.

    Each path may be a gallery image (absolute, or relative to `output_dir` like
    `images/<id>.jpg`) or a record id. Gallery records get their `remote_url` saved to the
    manifest; any other existing image file is published under a record built on the fly
    (not added to the gallery). Per-image failures are reported via `publish_error`.

    Only images inside `<output_dir>/images` can be published. Tool arguments may come from a
    prompt-injected agent, and this uploads to a possibly public repository, so any other
    path (the cookie cache, `.env`, SSH keys, ...) is refused outright.

    Raises
    ------
    `GeminiGenerationError`
        If GitHub publishing is not configured, a path matches neither a file nor a record,
        or it points outside the images directory or at a non-image file.

    """
    settings = settings or get_settings()
    if not settings.github_enabled:
        raise GeminiGenerationError(
            "GitHub publishing is not configured. Set GEMINI_GITHUB_REPO=owner/name in .env."
        )
    if len(image_paths) > _MAX_PUBLISH_IMAGES:
        raise GeminiGenerationError(
            f"Too many images ({len(image_paths)}); publish at most {_MAX_PUBLISH_IMAGES} at once."
        )

    from . import gallery

    known = gallery.load_records(settings)
    by_path = {
        (settings.output_dir / r.image_path).resolve(): r for r in known
    }
    by_id = {r.id: r for r in known}

    targets: list[GeneratedRecord] = []
    tracked: list[GeneratedRecord] = []
    for raw in image_paths:
        if raw in by_id:
            record = by_id[raw]
            tracked.append(record)
            targets.append(record)
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = settings.output_dir / candidate
        candidate = candidate.resolve()
        if candidate in by_path:
            record = by_path[candidate]
            tracked.append(record)
        elif candidate.is_file():
            record = GeneratedRecord(
                id=uuid.uuid4().hex,
                prompt=candidate.stem,
                model="",
                created_at=_now_iso(),
                image_path=_relative_to_output(candidate, settings),
                kind="generate",
            )
        else:
            raise GeminiGenerationError(f"Image not found: '{raw}'")
        targets.append(record)

    # Checked for gallery records too: their paths come from manifest.json, which is just a
    # file on disk and could have been edited to point anywhere.
    images_dir = settings.output_dir / "images"
    for raw, record in zip(image_paths, targets):
        local_path = settings.output_dir / record.image_path
        if not _is_within(local_path, images_dir):
            raise GeminiGenerationError(
                f"Refusing to publish '{raw}': only images inside '{images_dir}' can be published."
            )
        if not local_path.is_file() or _image_suffix(local_path) is None:
            raise GeminiGenerationError(f"Refusing to publish '{raw}': not an image file.")

    await _publish_records(targets, settings)

    published = [r for r in tracked if r.remote_url]
    if published:
        gallery.append_records(published, settings)
        gallery.render(settings)
    return targets


# ---------------------------------------------------------------------------
# Rendered-text check ("OCR" via Gemini itself - no extra OCR engine needed)
# ---------------------------------------------------------------------------

_TRANSCRIBE_PROMPT = (
    "Transcribe ALL text that is visible in this image exactly as it is written - same "
    "spelling, letters, diacritics, punctuation and line breaks. Do not translate, correct or "
    "describe anything. Output only the transcribed text. If there is no text, output: NO_TEXT"
)

# Uzbek Latin uses several look-alike apostrophes in o\u2018 g\u2018 and the tutuq belgisi;
# treat them as one so that "o'zbek" and "o\u2018zbek" compare equal.
_APOSTROPHES = "'`\u00b4\u02bb\u02bc\u2018\u2019\u2032\u02b9"


def normalize_text(text: str) -> str:
    """Normalize text for a spelling comparison (case, apostrophes, whitespace, punctuation)."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    for ch in _APOSTROPHES:
        text = text.replace(ch, "'")
    text = re.sub(r"[^\w'%+\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def compare_text(expected: str, found: str) -> dict[str, Any]:
    """Compare the expected string with the transcription; flag missing/misspelled words."""
    exp_norm = normalize_text(expected)
    found_norm = normalize_text(found)
    ratio = difflib.SequenceMatcher(None, exp_norm, found_norm).ratio() if exp_norm else 1.0
    found_words = found_norm.split()
    missing: list[str] = []
    suggestions: dict[str, str] = {}
    for word in exp_norm.split():
        if word in found_words:
            continue
        missing.append(word)
        close = difflib.get_close_matches(word, found_words, n=1, cutoff=0.6)
        if close:
            suggestions[word] = close[0]
    # Every expected word must appear; extra text elsewhere in the image is fine, so the
    # character similarity is reported for information only.
    return {
        "ok": not missing,
        "similarity": round(ratio, 3),
        "expected": expected,
        "found": found.strip(),
        "missing_words": missing,
        "misspelled_as": suggestions,
    }


async def check_text(
    image_path: str | Path,
    expected_text: str,
    *,
    settings: Settings | None = None,
    model: str | None = None,
    timeout: float = 90,
) -> dict[str, Any]:
    """Read the text rendered in an image (via Gemini) and compare it with `expected_text`."""
    settings = settings or get_settings()
    path = Path(image_path)
    if not path.is_file() or _image_suffix(path) is None:
        raise GeminiGenerationError(f"Not an image file: '{image_path}'")
    if not expected_text or not expected_text.strip():
        raise GeminiGenerationError("expected_text is empty.")
    client = await get_client(settings)
    chat = client.start_chat(model=model or settings.default_model)
    try:
        output = await asyncio.wait_for(
            chat.send_message(_TRANSCRIBE_PROMPT, files=[str(path)]), timeout=timeout
        )
    except asyncio.TimeoutError:
        raise GeminiTimeoutError(
            f"Reading text from the image took longer than {timeout:.0f}s."
        ) from None
    except GeminiImageError:
        raise
    except Exception as exc:  # noqa: BLE001 - translated at this boundary
        raise map_library_error(exc) from exc
    # Gemini sometimes appends citation markers such as "[cite: 1]" or "[cite_start]".
    found = re.sub(r"\[cite[^\]]*\]", "", output.text or "").strip()
    if found.upper() == "NO_TEXT":
        found = ""
    return compare_text(expected_text, found)
