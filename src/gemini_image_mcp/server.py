"""MCP server exposing Gemini image generation/editing tools.

Wraps `gemini_image_mcp.core` behind three MCP tools (`generate_image`, `edit_image`,
`list_models`) and serves them over stdio via the `mcp` package's `MCPServer`.

Every generated/edited image is appended to a cumulative, self-contained HTML gallery at
`<output_dir>/gallery.html`. That gallery - not base64 image data in the tool response - is
the intended viewing surface, so tool results only ever return file paths and text.

IMPORTANT: MCP stdio servers speak the protocol over stdout. Nothing in this module (or
anything it imports) may `print()` or otherwise write to stdout - see the module-level note
near `main()` for how that risk was checked.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .client import GeminiImageError
from .config import get_settings
from .gallery import ManifestLockTimeout
from .core import GeneratedRecord, edit_images, generate_images, list_models as _list_models

mcp = MCPServer(
    name="gemini-image-mcp",
    instructions=(
        "Generates and edits images by driving the Gemini web app with the user's own "
        "session cookies (no metered API key). Every result is also appended to a "
        "cumulative HTML gallery on disk; the gallery path returned by each tool is the "
        "primary way to view images, since tool results never include raw image data."
    ),
)


def _format_result(records: list[GeneratedRecord], model_used: str, gallery_path) -> str:
    settings = get_settings()
    abs_paths = [str((settings.output_dir / r.image_path).resolve()) for r in records]

    lines = [f"Generated {len(records)} image(s) using model '{model_used}'."]
    for p in abs_paths:
        lines.append(f"  - {p}")
    lines.append(f"Gallery (view all images here): {gallery_path}")
    return "\n".join(lines)


@mcp.tool()
async def generate_image(prompt: str, model: str | None = None) -> str:
    """Generate one or more new images from a text prompt using the Gemini web app.

    The `prompt` should describe the desired image in as much visual detail as helpful
    (subject, style, composition, lighting, etc.) - it is sent to Gemini as an image
    generation request. Newly generated images are saved to disk and appended to the
    cumulative HTML gallery; this tool returns the absolute paths of the new images plus
    the gallery path so the result can be viewed.

    Parameters
    ----------
    prompt: `str`
        A description of the image to generate.
    model: `str | None`, optional
        Model name/alias/id to use (see `list_models`). Defaults to the server's
        configured default model, or the account's own default if unset.

    """
    settings = get_settings()
    try:
        records = await generate_images(prompt, model=model, settings=settings)
    except (GeminiImageError, ManifestLockTimeout) as exc:
        raise ToolError(str(exc)) from exc

    gallery_path = settings.output_dir / "gallery.html"
    model_used = records[0].model if records else (model or settings.default_model or "default")
    return _format_result(records, model_used, gallery_path.resolve())


@mcp.tool()
async def edit_image(prompt: str, image_paths: list[str], model: str | None = None) -> str:
    """Edit one or more existing local image files using a text instruction.

    `image_paths` must be paths to existing image files on disk (e.g. a photo or a
    previously generated image) describing what to edit; `prompt` describes the desired
    change. The edited image(s) Gemini returns are saved to disk and appended to the
    cumulative HTML gallery; this tool returns the absolute paths of the new images plus
    the gallery path so the result can be viewed.

    Parameters
    ----------
    prompt: `str`
        Instructions describing the desired edit.
    image_paths: `list[str]`
        Paths to existing local image files to edit.
    model: `str | None`, optional
        Model name/alias/id to use (see `list_models`). Defaults to the server's
        configured default model, or the account's own default if unset.

    """
    settings = get_settings()
    try:
        records = await edit_images(prompt, image_paths, model=model, settings=settings)
    except (GeminiImageError, ManifestLockTimeout) as exc:
        raise ToolError(str(exc)) from exc

    gallery_path = settings.output_dir / "gallery.html"
    model_used = records[0].model if records else (model or settings.default_model or "default")
    return _format_result(records, model_used, gallery_path.resolve())


@mcp.tool()
async def list_models() -> str:
    """List the Gemini models (image/chat) available to the currently signed-in account.

    Returns each model's display name and internal model name/id, one per line. Use a
    `model_name` value from this list as the `model` argument to `generate_image` or
    `edit_image` to override the default model.

    """
    settings = get_settings()
    try:
        models = await _list_models(settings=settings)
    except (GeminiImageError, ManifestLockTimeout) as exc:
        raise ToolError(str(exc)) from exc

    if not models:
        return "No models were returned for this account."

    lines = ["Available models:"]
    for m in models:
        lines.append(f"  - {m['display_name']} (model_name={m['model_name']})")
    return "\n".join(lines)


def main() -> None:
    """Run the MCP server over stdio.

    Stdio MCP servers use stdout exclusively for the JSON-RPC protocol stream, so nothing
    here may print to stdout. This was verified by:
      - `gemini_webapi` contains no `print()` calls anywhere in its source; it logs via
        `loguru`, whose default sink (and the sink `set_log_level` installs) is `sys.stderr`.
      - `GeminiClient` defaults `verbose=False` and this project never passes `verbose=True`.
      - This module and `gemini_image_mcp.core`/`.client`/`.config`/`.gallery` contain no
        `print()` calls; any diagnostics should use `sys.stderr` or the `logging` module
        configured to a stderr handler, never stdout.
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
