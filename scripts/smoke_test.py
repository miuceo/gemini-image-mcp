#!/usr/bin/env python
"""Smoke test for gemini_image_mcp.

Usage
-----
    uv run python scripts/smoke_test.py "a watercolor fox"

Generates an image from the given prompt using the configured Gemini web app session
cookies, and prints the saved image paths, record count, and gallery path. Fails loudly
with an actionable message if cookies are not configured.

Does not hardcode any cookie values - it relies entirely on the environment / `.env` file
consumed by `gemini_image_mcp.config.Settings`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running this script directly from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemini_image_mcp.client import GeminiImageError  # noqa: E402
from gemini_image_mcp.config import get_settings  # noqa: E402
from gemini_image_mcp.core import generate_images  # noqa: E402


async def main(prompt: str) -> int:
    settings = get_settings()

    try:
        records = await generate_images(prompt, settings=settings)
    except GeminiImageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {len(records)} image(s) for prompt: {prompt!r}")
    for record in records:
        absolute_path = settings.output_dir / record.image_path
        print(f"  - {absolute_path}")

    gallery_path = settings.output_dir / "gallery.html"
    print(f"Gallery: {gallery_path}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} \"<prompt>\"", file=sys.stderr)
        sys.exit(2)

    sys.exit(asyncio.run(main(sys.argv[1])))
