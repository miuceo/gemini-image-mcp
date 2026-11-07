"""Image post-processing: aspect ratio / exact pixel size, and small previews.

Gemini's web app has no aspect-ratio or size control, so the requested ratio is written into
the prompt (which the image model usually honours) and the result is then center-cropped to
the exact ratio and, if a pixel size was asked for, resized to it - e.g. an Uzum product card
at exactly 1080x1440. Like `core`, this module MUST NOT import anything from `mcp`.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from .client import GeminiGenerationError

# Named shortcuts for common marketplace / social formats.
SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "uzum": (1080, 1440),
    "uzum-card": (1080, 1440),
    "instagram-post": (1080, 1350),
    "instagram-square": (1080, 1080),
    "story": (1080, 1920),
    "youtube-thumbnail": (1280, 720),
}

_MAX_SIDE = 8192


@dataclass(frozen=True)
class Shape:
    """A requested output shape: a ratio, optionally with an exact pixel size."""

    ratio_w: int
    ratio_h: int
    width: int | None = None
    height: int | None = None

    @property
    def ratio(self) -> float:
        return self.ratio_w / self.ratio_h

    @property
    def label(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return f"{self.ratio_w}:{self.ratio_h}"

    def prompt_hint(self) -> str:
        orientation = (
            "portrait" if self.ratio < 0.98 else "landscape" if self.ratio > 1.02 else "square"
        )
        return (
            f"Compose the image in a {self.ratio_w}:{self.ratio_h} {orientation} aspect ratio, "
            "keeping all important content and any text well inside the frame with safe margins."
        )


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def parse_shape(aspect_ratio: str | None, size: str | None) -> Shape | None:
    """Parse `aspect_ratio` ("3:4") and/or `size` ("1080x1440" or a preset like "uzum").

    The size wins when both are given (its ratio is what will be produced).
    """
    aspect_ratio = (aspect_ratio or "").strip().lower()
    size = (size or "").strip().lower()
    if not aspect_ratio and not size:
        return None

    if size:
        if size in SIZE_PRESETS:
            w, h = SIZE_PRESETS[size]
        else:
            m = re.fullmatch(r"(\d{2,5})\s*[x×*]\s*(\d{2,5})", size)
            if not m:
                raise GeminiGenerationError(
                    f"Invalid size '{size}'. Use WIDTHxHEIGHT (e.g. 1080x1440) or one of: "
                    + ", ".join(SIZE_PRESETS)
                )
            w, h = int(m.group(1)), int(m.group(2))
        if not (16 <= w <= _MAX_SIDE and 16 <= h <= _MAX_SIDE):
            raise GeminiGenerationError(f"Size {w}x{h} is out of range (16..{_MAX_SIDE} px).")
        g = _gcd(w, h)
        return Shape(w // g, h // g, w, h)

    m = re.fullmatch(r"(\d{1,4}(?:\.\d+)?)\s*[:/x]\s*(\d{1,4}(?:\.\d+)?)", aspect_ratio)
    if not m:
        raise GeminiGenerationError(
            f"Invalid aspect_ratio '{aspect_ratio}'. Use W:H, e.g. 3:4, 16:9 or 1:1."
        )
    rw, rh = float(m.group(1)), float(m.group(2))
    if rw <= 0 or rh <= 0 or not (0.1 <= rw / rh <= 10):
        raise GeminiGenerationError(f"Unsupported aspect_ratio '{aspect_ratio}'.")
    # Normalise decimals like 1.91:1 to integers.
    scale = 100 if (rw % 1 or rh % 1) else 1
    iw, ih = int(round(rw * scale)), int(round(rh * scale))
    g = _gcd(iw, ih)
    return Shape(iw // g, ih // g)


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a declared dependency
        raise GeminiGenerationError("Pillow is required for image post-processing.") from exc
    return Image


def image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) of an image file, or None if it can't be read."""
    try:
        Image = _pil()
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001 - unreadable / not an image
        return None


def fit_to_shape(src: Path, dest: Path, shape: Shape) -> tuple[int, int]:
    """Center-crop `src` to `shape`'s ratio (and resize to its exact size) into `dest`.

    Returns the final (width, height). `dest` keeps `src`'s format (JPEG stays JPEG).
    """
    Image = _pil()
    with Image.open(src) as im:
        fmt = im.format or "PNG"
        im.load()
        w, h = im.size
        target = shape.ratio
        if abs(w / h - target) > 0.002:
            if w / h > target:  # too wide: trim the sides
                new_w = max(1, round(h * target))
                left = (w - new_w) // 2
                im = im.crop((left, 0, left + new_w, h))
            else:  # too tall: trim top and bottom
                new_h = max(1, round(w / target))
                top = (h - new_h) // 2
                im = im.crop((0, top, w, top + new_h))
        if shape.width and shape.height and im.size != (shape.width, shape.height):
            im = im.resize((shape.width, shape.height), Image.Resampling.LANCZOS)
        save_kwargs: dict = {}
        if fmt.upper() in ("JPEG", "JPG"):
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            save_kwargs = {"quality": 95, "subsampling": 0, "optimize": True}
        elif fmt.upper() == "WEBP":
            save_kwargs = {"quality": 95}
        im.save(dest, format=fmt, **save_kwargs)
        return im.size


def thumbnail_jpeg(path: Path, max_side: int = 640, quality: int = 75) -> bytes:
    """A small JPEG preview of an image, for returning to the model as image content."""
    Image = _pil()
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
