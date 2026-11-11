"""MCP server exposing Gemini image generation/editing tools.

Wraps `gemini_image_mcp.core` / `.jobs` behind MCP tools and serves them over stdio via the
`mcp` package's `MCPServer`.

Every generated/edited image is appended to a cumulative, self-contained HTML gallery at
`<output_dir>/gallery.html`. Tool results return file paths and text, plus small JPEG previews
when `preview=True` is passed, so the model can check its own output.

No tool call blocks for long: generations run as background jobs (see `jobs.py`). The
synchronous `generate_image` / `edit_image` wait at most GEMINI_SYNC_WAIT seconds (default
85, under the ~120 s after which MCP bridges abandon a call) and then return a job id to poll
with `get_job` instead of leaving the caller guessing.

IMPORTANT: MCP stdio servers speak the protocol over stdout. Nothing in this module (or
anything it imports) may `print()` or otherwise write to stdout - see the module-level note
near `main()` for how that risk was checked.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from . import doctor as _doctor
from . import imaging, jobs
from .client import GeminiImageError, cookie_source_used, reset_client
from .config import get_settings
from .core import (
    GeneratedRecord,
    _image_suffix,
    check_text,
    list_models as _list_models,
    publish_images,
)
from .gallery import ManifestLockTimeout, load_records

STARTED_AT = time.time()

mcp = MCPServer(
    name="gemini-image-mcp",
    instructions=(
        "Generates and edits images by driving the Gemini web app with the user's own "
        "session cookies (no metered API key). Every result is also appended to a "
        "cumulative HTML gallery on disk. Pass preview=true to also get small previews of "
        "the images so you can check them (text spelling, layout). Generations are jobs: "
        "if generate_image/edit_image says the job is still running, poll get_job with the "
        "job id instead of starting again; pass an idempotency_key so a retry never makes a "
        "duplicate. For many images use generate_batch. Call ping first to check that the "
        "server is reachable before long work."
    ),
)

_MAX_PREVIEWS = 8


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _text_check_line(check: dict[str, Any] | None) -> str | None:
    if not check:
        return None
    if check.get("ok") is None:
        return f"    Text check: could not run ({check.get('error')})"
    if check.get("ok"):
        return f"    Text check: OK (similarity {check.get('similarity')})"
    bits = [f"similarity {check.get('similarity')}"]
    if check.get("missing_words"):
        bits.append("missing/misspelled: " + ", ".join(check["missing_words"][:12]))
    if check.get("misspelled_as"):
        bits.append(
            "rendered as: "
            + ", ".join(f"{k} -> {v}" for k, v in list(check["misspelled_as"].items())[:8])
        )
    found = (check.get("found") or "").replace("\n", " / ")
    if len(found) > 200:
        found = found[:197] + "..."
    bits.append(f"found text: '{found}'")
    return "    Text check: MISMATCH - " + "; ".join(bits)


def _record_lines(record: dict[str, Any], output_dir: Path) -> list[str]:
    path = (output_dir / record["image_path"]).resolve()
    meta = []
    if record.get("size"):
        meta.append(record["size"])
    if record.get("project"):
        meta.append(f"project={record['project']}")
    if record.get("tags"):
        meta.append("tags=" + ",".join(record["tags"]))
    lines = [f"  - {path}" + (f"  [{' | '.join(meta)}]" if meta else "")]
    if record.get("remote_url"):
        lines.append(f"    URL: {record['remote_url']}")
    tc = _text_check_line(record.get("text_check"))
    if tc:
        lines.append(tc)
    return lines


def _previews(paths: list[Path], max_side: int = 640) -> list[Image]:
    out: list[Image] = []
    for p in paths[:_MAX_PREVIEWS]:
        try:
            out.append(Image(data=imaging.thumbnail_jpeg(p, max_side=max_side), format="jpeg"))
        except Exception:  # noqa: BLE001 - a missing preview never fails the call
            continue
    return out


def _format_job(job: dict[str, Any], preview: bool = False) -> str | list:
    settings = get_settings()
    status = job.get("status")
    jid = job.get("id")
    if status == "succeeded":
        result = job.get("result") or {}
        records = result.get("records") or []
        model = records[0].get("model") if records else "default"
        lines = [f"Generated {len(records)} image(s) using model '{model}' (job {jid})."]
        for r in records:
            lines.extend(_record_lines(r, settings.output_dir))
        for err in result.get("publish_errors") or []:
            lines.append(f"  GitHub publish failed (image kept locally): {err}")
        lines.append(f"Gallery (view all images here): {(settings.output_dir / 'gallery.html').resolve()}")
        text = "\n".join(lines)
        if preview and records:
            imgs = _previews([settings.output_dir / r["image_path"] for r in records])
            if imgs:
                return [text, *imgs]
        return text
    if status in jobs.ACTIVE:
        return (
            f"Job {jid} is still {status} (started {job.get('started_at') or job.get('created_at')}). "
            f"It keeps running in the background: call get_job(job_id='{jid}', wait_seconds=60) "
            "to wait for it. Do NOT start the same generation again."
        )
    return f"Job {jid} {status}: {job.get('error')}"


def _resolve_image(raw: str) -> Path:
    """An absolute path, a path relative to the output dir, or a gallery record id."""
    settings = get_settings()
    candidate = Path(raw)
    if not candidate.is_absolute():
        rel = settings.output_dir / candidate
        if rel.is_file():
            return rel
        for record in load_records(settings):
            if record.id == raw:
                return settings.output_dir / record.image_path
    return candidate


def _shape_params(
    aspect_ratio: str | None, size: str | None, project: str | None, tags: list[str] | None
) -> dict[str, Any]:
    # Validate now so a typo is reported immediately, not from inside a background job.
    try:
        imaging.parse_shape(aspect_ratio, size)
    except GeminiImageError as exc:
        raise ToolError(str(exc)) from exc
    return {"aspect_ratio": aspect_ratio, "size": size, "project": project, "tags": tags}


async def _start(kind: str, params: dict[str, Any], idempotency_key, expected_text):
    settings = get_settings()
    try:
        return await jobs.start_job(
            kind, params, settings, idempotency_key=idempotency_key, expected_text=expected_text
        )
    except GeminiImageError as exc:
        raise ToolError(str(exc)) from exc


async def _run_sync(kind: str, params: dict[str, Any], idempotency_key, expected_text, preview):
    settings = get_settings()
    job, created = await _start(kind, params, idempotency_key, expected_text)
    if not created and job.get("status") == "succeeded":
        out = _format_job(job, preview)
        note = "(Idempotency key matched an earlier job; returning its result instead of regenerating.)\n"
        return [note + out[0], *out[1:]] if isinstance(out, list) else note + out
    job = await jobs.wait_job(settings, job["id"], settings.sync_wait) or job
    if job.get("status") == "failed":
        raise ToolError(f"{job.get('error')} (job {job['id']})")
    return _format_job(job, preview)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_SHAPE_DOC = """
    aspect_ratio: `str | None`, optional
        Output aspect ratio, e.g. "3:4", "1:1", "16:9". Added to the prompt, then the image is
        center-cropped to exactly this ratio.
    size: `str | None`, optional
        Exact output size in pixels, e.g. "1080x1440", or a preset: uzum (1080x1440),
        instagram-post (1080x1350), instagram-square, story (1080x1920), youtube-thumbnail.
        Overrides aspect_ratio. The unprocessed original is kept in a `raw/` subfolder.
    project: `str | None`, optional
        Project name; images are saved under images/<project>/ and tagged in the gallery.
    tags: `list[str] | None`, optional
        Free-form gallery tags (style, product, language, ...).
    expected_text: `str | None`, optional
        Text the image should contain. After generating, the rendered text is read back and
        compared; misspellings (e.g. Uzbek/Russian) are flagged in the result.
    idempotency_key: `str | None`, optional
        Any unique string for this request. Calling again with the same key returns the
        earlier job/result instead of generating a duplicate - use it whenever you might retry.
    preview: `bool`, optional
        Also return small JPEG previews of the result so you can look at it.
"""


def _with_shape_doc(fn):
    fn.__doc__ = (fn.__doc__ or "").rstrip() + "\n" + _SHAPE_DOC
    return fn


@mcp.tool()
@_with_shape_doc
async def generate_image(
    prompt: str,
    model: str | None = None,
    publish: bool | None = None,
    aspect_ratio: str | None = None,
    size: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    expected_text: str | None = None,
    idempotency_key: str | None = None,
    preview: bool = False,
):
    """Generate one or more new images from a text prompt using the Gemini web app.

    Waits up to ~85 s. If Gemini is slower, returns a job id instead - then call `get_job`
    (never re-run the same prompt, or use an `idempotency_key` so a re-run is harmless).

    Parameters
    ----------
    prompt: `str`
        A description of the image to generate (subject, style, composition, lighting, and
        any exact text to render).
    model: `str | None`, optional
        Model name/alias/id to use (see `list_models`). Defaults to the server's default.
    publish: `bool | None`, optional
        Upload the result to the configured GitHub image repo and return its public URL.
        Leave unset to follow the server default (off unless GEMINI_GITHUB_AUTO_PUBLISH=true).
    """
    params = {"prompt": prompt, "model": model, "publish": publish,
              **_shape_params(aspect_ratio, size, project, tags)}
    return await _run_sync("generate", params, idempotency_key, expected_text, preview)



@mcp.tool()
@_with_shape_doc
async def edit_image(
    prompt: str,
    image_paths: list[str],
    model: str | None = None,
    publish: bool | None = None,
    aspect_ratio: str | None = None,
    size: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    expected_text: str | None = None,
    idempotency_key: str | None = None,
    preview: bool = False,
):
    """Edit existing local image files (or use them as references) with a text instruction.

    `image_paths` must be existing image files on disk (PNG/JPEG/WEBP/GIF, up to 10), e.g. a
    product photo downloaded from Telegram or Uzum. Waits up to ~85 s, then returns a job id
    to poll with `get_job` if Gemini is still working.

    Parameters
    ----------
    prompt: `str`
        Instructions describing the desired edit / new image based on the references.
    image_paths: `list[str]`
        Paths to existing local image files.
    model, publish: optional
        Same as for `generate_image`.
    """
    params = {"prompt": prompt, "image_paths": image_paths, "model": model, "publish": publish,
              **_shape_params(aspect_ratio, size, project, tags)}
    return await _run_sync("edit", params, idempotency_key, expected_text, preview)



@mcp.tool()
async def start_generation(
    prompt: str,
    image_paths: list[str] | None = None,
    model: str | None = None,
    publish: bool | None = None,
    aspect_ratio: str | None = None,
    size: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    expected_text: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Start a generation (or, with `image_paths`, an edit) in the background and return at once.

    Returns a job id immediately; poll it with `get_job(job_id, wait_seconds=60)`. Same
    parameters as `generate_image` / `edit_image`. Prefer this when making several images
    or when the connection to this server is flaky.
    """
    shape = _shape_params(aspect_ratio, size, project, tags)
    if image_paths:
        job, created = await _start(
            "edit", {"prompt": prompt, "image_paths": image_paths, "model": model,
                     "publish": publish, **shape}, idempotency_key, expected_text)
    else:
        job, created = await _start(
            "generate", {"prompt": prompt, "model": model, "publish": publish, **shape},
            idempotency_key, expected_text)
    prefix = "Started" if created else "Found existing"
    return (
        f"{prefix} job {job['id']} (status: {job['status']}). "
        f"Call get_job(job_id='{job['id']}', wait_seconds=60) for the result."
    )


@mcp.tool()
async def get_job(job_id: str, wait_seconds: float = 0, preview: bool = False):
    """Get the status/result of a generation job; optionally wait up to 90 s for it to finish.

    Statuses: queued, running, succeeded (with image paths), failed (with the error),
    interrupted (the server stopped mid-job; the image may still be in the gallery).
    """
    settings = get_settings()
    try:
        job = await jobs.wait_job(settings, job_id, max(0.0, min(float(wait_seconds), 90.0)))
    except GeminiImageError as exc:
        raise ToolError(str(exc)) from exc
    if job is None:
        raise ToolError(f"No job with id '{job_id}'.")
    return _format_job(job, preview)


@mcp.tool()
async def list_jobs(limit: int = 10) -> str:
    """List the most recent generation jobs (any server process) with their status."""
    settings = get_settings()
    rows = jobs.list_jobs(settings, max(1, min(int(limit), 50)))
    if not rows:
        return "No jobs yet."
    lines = ["Recent jobs (newest first):"]
    for j in rows:
        prompt = str((j.get("params") or {}).get("prompt", ""))[:70]
        n = len(((j.get("result") or {}).get("paths")) or [])
        extra = f", {n} image(s)" if n else (f", error: {str(j.get('error'))[:80]}" if j.get("error") else "")
        lines.append(f"  - {j['id']} [{j['status']}] {j.get('created_at')}: {prompt!r}{extra}")
    return "\n".join(lines)


def _format_batch(batch: dict[str, Any], detail: bool = True) -> str:
    items = batch.get("items", [])
    counts: dict[str, int] = {}
    for it in items:
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    text_bad = [it["id"] for it in items if it.get("text_check") and it["text_check"].get("ok") is False]
    lines = [f"Batch {batch['id']} - {batch['status']} ({len(items)} items; {summary})"]
    if batch.get("note"):
        lines.append(f"Note: {batch['note']}")
    if text_bad:
        lines.append(f"Text mismatches in items: {', '.join(text_bad)}")
    if detail:
        for it in items:
            line = f"  [{it['status']}] {it['id']}: "
            if it.get("paths"):
                line += ", ".join(it["paths"])
            elif it.get("error"):
                line += f"error: {it['error']}"
            else:
                line += repr(it["prompt"][:60])
            lines.append(line)
            tc = _text_check_line(it.get("text_check"))
            if tc and it["text_check"].get("ok") is not True:
                lines.append(tc)
    if batch["status"] in jobs.ACTIVE:
        lines.append(f"Still running in the background; check again with get_batch('{batch['id']}', wait_seconds=60).")
    elif batch["status"] in ("paused", "interrupted", "cancelled", "completed_with_errors"):
        lines.append(f"Continue with resume_batch('{batch['id']}').")
    return "\n".join(lines)


@mcp.tool()
async def generate_batch(
    items: list[dict[str, Any]],
    project: str | None = None,
    tags: list[str] | None = None,
    aspect_ratio: str | None = None,
    size: str | None = None,
    batch_id: str | None = None,
    max_retries: int = 2,
    text_retries: int = 0,
    delay_seconds: float = 3.0,
    publish: bool | None = None,
) -> str:
    """Queue many generations and run them one by one in the background. Returns at once.

    `items`: list of objects, each {"id": "sku-1", "prompt": "...", "tags": [...],
    "aspect_ratio"/"size", "project", "expected_text", "model", "image_paths"} - only
    `prompt` is required; `image_paths` makes that item an edit using those reference images.
    Top-level `project`, `tags`, `aspect_ratio`, `size` are defaults for every item.

    Handles Gemini rate limits (waits 1, 2, 4 min, then pauses the batch), retries failed
    items up to `max_retries` times, and with `expected_text` + `text_retries` regenerates
    images whose rendered text is misspelled. Progress is saved after every item, so after a
    crash or pause `resume_batch` continues where it stopped (finished items are skipped).
    Poll progress with `get_batch`.
    """
    settings = get_settings()
    try:
        batch = await jobs.start_batch(
            items,
            settings,
            defaults={"project": project, "tags": tags or [], "aspect_ratio": aspect_ratio, "size": size},
            batch_id=batch_id,
            max_retries=max_retries,
            text_retries=text_retries,
            delay_seconds=delay_seconds,
            publish=publish,
        )
    except GeminiImageError as exc:
        raise ToolError(str(exc)) from exc
    return (
        f"Batch {batch['id']} started with {len(batch['items'])} item(s). It runs in the "
        f"background; check progress with get_batch(batch_id='{batch['id']}', wait_seconds=60)."
    )


@mcp.tool()
async def get_batch(batch_id: str = "", wait_seconds: float = 0, details: bool = True) -> str:
    """Show a batch's progress and per-item results (paths, errors, text-check flags).

    Leave `batch_id` empty to list recent batches. `wait_seconds` (max 90) waits for the
    batch to make progress before answering.
    """
    settings = get_settings()
    if not batch_id.strip():
        rows = jobs.list_batches(settings)
        if not rows:
            return "No batches yet."
        return "\n".join(_format_batch(b, detail=False) for b in rows)
    try:
        batch = jobs.load_batch(settings, batch_id)
        deadline = time.monotonic() + max(0.0, min(float(wait_seconds), 90.0))
        while batch and batch.get("status") in jobs.ACTIVE and time.monotonic() < deadline:
            await asyncio.sleep(min(3.0, max(0.1, deadline - time.monotonic())))
            batch = jobs.load_batch(settings, batch_id)
    except GeminiImageError as exc:
        raise ToolError(str(exc)) from exc
    if batch is None:
        raise ToolError(f"No batch with id '{batch_id}'.")
    return _format_batch(batch, detail=details)


@mcp.tool()
async def resume_batch(batch_id: str, retry_failed: bool = True) -> str:
    """Continue a paused, interrupted, cancelled or partly failed batch in the background.

    Finished items are skipped; with `retry_failed` (default) failed items are tried again.
    """
    try:
        batch = await jobs.resume_batch(get_settings(), batch_id, retry_failed=retry_failed)
    except GeminiImageError as exc:
        raise ToolError(str(exc)) from exc
    pending = sum(1 for it in batch["items"] if it["status"] == "pending")
    return f"Batch {batch['id']} resumed: {pending} item(s) to go. Track it with get_batch('{batch['id']}')."


@mcp.tool()
async def cancel_batch(batch_id: str) -> str:
    """Stop a running batch after its current item (it can be resumed later)."""
    try:
        jobs.cancel_batch(get_settings(), batch_id)
    except GeminiImageError as exc:
        raise ToolError(str(exc)) from exc
    return f"Batch {batch_id} will stop after its current item."


@mcp.tool()
async def view_images(images: list[str], max_side: int = 768):
    """Return small previews of images so you can look at them (QA of text, layout, quality).

    Each entry may be an absolute path, a path relative to the output folder, or a gallery
    record id. Up to 8 images; `max_side` is the longest side in pixels (256-1568).
    """
    max_side = max(256, min(int(max_side), 1568))
    out: list[Any] = []
    problems: list[str] = []
    for raw in images[:_MAX_PREVIEWS]:
        path = _resolve_image(raw)
        if not path.is_file() or _image_suffix(path) is None:
            problems.append(f"Not an image file: {raw}")
            continue
        imgs = _previews([path], max_side)
        if not imgs:
            problems.append(f"Could not read: {raw}")
            continue
        size = imaging.image_size(path)
        out.append(f"{path}" + (f" ({size[0]}x{size[1]})" if size else ""))
        out.append(imgs[0])
    if len(images) > _MAX_PREVIEWS:
        problems.append(f"Only the first {_MAX_PREVIEWS} images are shown.")
    if problems:
        out.append("\n".join(problems))
    return out or "No images to show."


@mcp.tool()
async def check_image_text(image: str, expected_text: str) -> str:
    """Read the text rendered in an image and compare it with the text it should contain.

    Flags missing or misspelled words (Uzbek o‘/g‘ apostrophe variants are treated as equal).
    `image` is a path or gallery record id. Uses one Gemini request.
    """
    settings = get_settings()
    path = _resolve_image(image)
    try:
        result = await asyncio.wait_for(check_text(path, expected_text, settings=settings), 100)
    except asyncio.TimeoutError as exc:
        raise ToolError("Reading the image text timed out.") from exc
    except GeminiImageError as exc:
        raise ToolError(str(exc)) from exc
    head = "OK - the text matches." if result["ok"] else "MISMATCH - the text differs."
    line = _text_check_line(result) or ""
    return f"{head}\n{line.strip()}\nFound text:\n{result['found'] or '(none)'}"


@mcp.tool()
async def publish_image(image_paths: list[str]) -> str:
    """Publish already-generated images to the user's configured GitHub image repository.

    Use this for images created earlier (or with publishing turned off) that the user now
    wants hosted. Each entry may be an absolute image path, a path relative to the output
    directory (e.g. `images/<id>.jpg`), or a gallery record id. Returns the public URL of
    each published image; gallery entries are updated to show the link.

    Parameters
    ----------
    image_paths: `list[str]`
        Images to publish.

    """
    settings = get_settings()
    try:
        records = await asyncio.wait_for(publish_images(image_paths, settings=settings), 100)
    except asyncio.TimeoutError as exc:
        raise ToolError("Publishing to GitHub timed out.") from exc
    except (GeminiImageError, ManifestLockTimeout) as exc:
        raise ToolError(str(exc)) from exc

    ok = [r for r in records if r.remote_url]
    lines = [f"Published {len(ok)} of {len(records)} image(s) to {settings.github_repo}."]
    for record in records:
        lines.append(f"  - {(settings.output_dir / record.image_path).resolve()}")
        lines.extend(_publish_lines(record))
    return "\n".join(lines)


def _publish_lines(record: GeneratedRecord) -> list[str]:
    if record.remote_url:
        return [f"    URL: {record.remote_url}"]
    if record.publish_error:
        return [f"    GitHub publish failed (image kept locally): {record.publish_error}"]
    return []


@mcp.tool()
async def list_models() -> str:
    """List the Gemini models (image/chat) available to the currently signed-in account.

    Returns each model's display name and internal model name/id, one per line. Use a
    `model_name` value from this list as the `model` argument to `generate_image` or
    `edit_image` to override the default model. Times out with a clear error instead of hanging.

    """
    settings = get_settings()
    try:
        models = await asyncio.wait_for(_list_models(settings=settings), settings.init_timeout + 10)
    except asyncio.TimeoutError as exc:
        raise ToolError(
            f"Listing models timed out after {settings.init_timeout + 10:.0f}s (Gemini sign-in "
            "did not finish). Run `doctor` to see which step is stuck."
        ) from exc
    except (GeminiImageError, ManifestLockTimeout) as exc:
        raise ToolError(str(exc)) from exc

    if not models:
        return "No models were returned for this account."

    lines = ["Available models:"]
    for m in models:
        lines.append(f"  - {m['display_name']} (model_name={m['model_name']})")
    return "\n".join(lines)


@mcp.tool()
async def ping() -> str:
    """Instant health check (no network): is this server up, signed in, and what is it running?

    Call before starting long or batched work. A reply means the device and server are
    reachable; no reply within a few seconds means the desktop bridge or device is offline.
    """
    settings = get_settings()
    running_jobs = [k for k, t in jobs._tasks.items() if not t.done()]
    batches = [k for k in running_jobs if k.startswith("batch:")]
    source = cookie_source_used()
    return "\n".join([
        f"gemini-image-mcp {__version__} is up (pid {os.getpid()}, uptime {int(time.time() - STARTED_AT)}s).",
        f"Gemini session: {'signed in (cookies from ' + source + ')' if source else 'not signed in yet (signs in on first use)'}",
        f"Running here: {len(running_jobs) - len(batches)} job(s), {len(batches)} batch(es).",
        f"Limits: sync wait {settings.sync_wait:.0f}s, job timeout {settings.job_timeout:.0f}s, "
        f"sign-in timeout {settings.init_timeout:.0f}s, max concurrent {settings.max_concurrent}.",
        f"Output: {settings.output_dir}",
        f"GitHub auto-publish: {'ON' if settings.github_enabled and settings.github_auto_publish else 'off'}",
    ])


@mcp.tool()
async def doctor(test_generation: bool = False) -> str:
    """Check this server's setup: output folder, cookies, Gemini sign-in, gallery, GitHub.

    Use when image generation or publishing fails, or the user asks whether things are set
    up correctly. Returns one ok/warn/fail line per check with a fix hint; never includes
    cookie or token values. `test_generation=True` also makes one tiny test image end to end
    (uses one generation of quota; returns a job id if it takes longer than the wait).

    """
    settings = get_settings()
    checks = await _doctor.run_checks(settings)
    report = _doctor.format_report(checks)
    if not test_generation:
        return report
    signed_in = any(c.name == "Gemini sign-in" and c.status == "ok" for c in checks)
    if not signed_in:
        return report + "\n[SKIP] Test generation: Gemini sign-in is not OK."
    job, _ = await jobs.start_job(
        "generate",
        {"prompt": "a single small plain red circle centered on a white background, flat",
         "publish": False, "project": "_doctor", "tags": ["doctor"]},
        settings,
    )
    job = await jobs.wait_job(settings, job["id"], min(settings.sync_wait, 60)) or job
    if job.get("status") == "succeeded":
        line = f"[ OK ] Test generation: {', '.join(job['result']['paths'])}"
    elif job.get("status") in jobs.ACTIVE:
        line = f"[WAIT] Test generation still running: get_job('{job['id']}')"
    else:
        line = f"[FAIL] Test generation: {job.get('error')}"
    return report + "\n" + line


async def _doctor_cli(online: bool, test_generation: bool = False) -> int:
    try:
        checks = await _doctor.run_checks(
            get_settings(), online=online, test_generation=test_generation
        )
    finally:
        await reset_client()
    print(_doctor.format_report(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0


def main() -> None:
    """Run the MCP server over stdio, or `gemini-image-mcp doctor [--offline] [--test-generation]`.

    The `doctor` subcommand is a plain CLI (not the stdio protocol), so it prints to stdout.

    Stdio MCP servers use stdout exclusively for the JSON-RPC protocol stream, so nothing
    here may print to stdout. This was verified by:
      - `gemini_webapi` contains no `print()` calls anywhere in its source; it logs via
        `loguru`, whose default sink (and the sink `set_log_level` installs) is `sys.stderr`.
      - `GeminiClient` defaults `verbose=False` and this project never passes `verbose=True`.
      - This module and `gemini_image_mcp.core`/`.client`/`.config`/`.gallery`/`.jobs`/
        `.imaging` contain no `print()` calls outside the doctor CLI; any diagnostics should
        use `sys.stderr` or the `logging` module configured to a stderr handler.
    """
    args = sys.argv[1:]
    if args and args[0] == "doctor":
        sys.exit(asyncio.run(_doctor_cli(
            online="--offline" not in args[1:],
            test_generation="--test-generation" in args[1:],
        )))
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
