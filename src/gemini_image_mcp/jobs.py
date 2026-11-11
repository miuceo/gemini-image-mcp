"""Background generation jobs and batches, persisted to disk.

Why this exists: a Gemini web-app generation can take longer than an MCP bridge is willing to
wait for a single tool call (~2 minutes). When the bridge gives up, the caller can't tell
whether the image failed or is still being made, and a blind retry creates duplicates.

So every generation runs as a *job*: it is started in the background, its state is written to
`<output_dir>/jobs/<job_id>.json`, and any tool call - in this process or another server
process pointed at the same output directory - can read it back. An optional idempotency key
maps deterministically to the job id, so retrying with the same key returns the existing job
instead of generating again.

Batches (`<output_dir>/batches/<batch_id>.json`) run many prompts one after another with
rate-limit backoff, retries and per-item status, and can be resumed after a crash.

A running job/batch rewrites a heartbeat timestamp every few seconds; if the heartbeat goes
stale (the server process died), the job is reported as "interrupted".

Like `core`, this module MUST NOT import anything from `mcp`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import core
from .client import (
    GeminiAuthError,
    GeminiImageError,
    GeminiUsageLimitError,
)
from .config import Settings

HEARTBEAT_EVERY = 5.0
STALE_AFTER = 30.0
ACTIVE = ("queued", "running")
MAX_BATCH_ITEMS = 200
_ITEM_KEYS = {"id", "prompt", "tags", "aspect_ratio", "size", "project", "expected_text",
              "model", "image_paths"}

# Strong references: asyncio only keeps weak references to tasks.
_tasks: dict[str, asyncio.Task] = {}
_cancelled_batches: set[str] = set()
# Backoff / retry / pacing waits go through this hook so tests can skip them.
_sleep = asyncio.sleep


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jobs_dir(settings: Settings) -> Path:
    d = settings.output_dir / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _batches_dir(settings: Settings) -> Path:
    d = settings.output_dir / "batches"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, data: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    for _ in range(3):  # tolerate a concurrent os.replace on Windows
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    return None


def _safe_id(raw: str) -> str:
    """Job/batch ids are used as file names; only allow a conservative character set."""
    cleaned = "".join(ch for ch in str(raw) if ch.isalnum() or ch in "-_")
    if not cleaned or cleaned != str(raw) or len(cleaned) > 80:
        raise GeminiImageError(f"Invalid id '{raw}'.")
    return cleaned


def job_id_for_key(key: str) -> str:
    return "k" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _is_stale(data: dict[str, Any], own_tasks: dict[str, asyncio.Task], key: str) -> bool:
    if key in own_tasks and not own_tasks[key].done():
        return False
    return time.time() - float(data.get("heartbeat") or 0) > STALE_AFTER


async def _heartbeat(path: Path, state: dict[str, Any]) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_EVERY)
        state["heartbeat"] = time.time()
        try:
            _write_json(path, state)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def load_job(settings: Settings, job_id: str) -> dict[str, Any] | None:
    job_id = _safe_id(job_id)
    data = _read_json(_jobs_dir(settings) / f"{job_id}.json")
    if data and data.get("status") in ACTIVE and _is_stale(data, _tasks, job_id):
        data["status"] = "interrupted"
        data["error"] = (
            "The server process running this job stopped before it finished. "
            "Check the gallery - the image may exist - or start it again."
        )
    return data


def list_jobs(settings: Settings, limit: int = 20) -> list[dict[str, Any]]:
    files = sorted(_jobs_dir(settings).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[: max(1, limit)]:
        job = load_job(settings, f.stem)
        if job:
            out.append(job)
    return out


async def start_job(
    kind: str,
    params: dict[str, Any],
    settings: Settings,
    *,
    idempotency_key: str | None = None,
    expected_text: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Start a generate/edit job in the background. Returns (job, created).

    With an `idempotency_key`, an existing queued/running/succeeded job for that key is
    returned instead of starting a new one (created=False). Failed or interrupted jobs are
    re-run under the same id.
    """
    if kind not in ("generate", "edit"):
        raise GeminiImageError(f"Unknown job kind '{kind}'.")
    if idempotency_key:
        job_id = job_id_for_key(idempotency_key)
        existing = load_job(settings, job_id)
        if existing and existing.get("status") in (*ACTIVE, "succeeded"):
            return existing, False
    else:
        job_id = uuid.uuid4().hex[:16]

    job: dict[str, Any] = {
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "params": params,
        "expected_text": expected_text or None,
        "idempotency_key": idempotency_key or None,
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "heartbeat": time.time(),
        "pid": os.getpid(),
        "result": None,
        "error": None,
    }
    _write_json(_jobs_dir(settings) / f"{job_id}.json", job)
    task = asyncio.create_task(_run_job(job, settings), name=f"gemini-job-{job_id}")
    _tasks[job_id] = task
    task.add_done_callback(lambda _t, jid=job_id: _tasks.pop(jid, None))
    return job, True


async def _execute(kind: str, params: dict[str, Any], settings: Settings) -> list[core.GeneratedRecord]:
    if kind == "generate":
        return await core.generate_images(settings=settings, **params)
    return await core.edit_images(settings=settings, **params)


async def _attach_text_check(
    records: list[core.GeneratedRecord], expected_text: str, settings: Settings
) -> None:
    """Run the OCR check on each record and save the result into the gallery manifest."""
    from . import gallery

    for record in records:
        try:
            record.text_check = await core.check_text(
                settings.output_dir / record.image_path, expected_text, settings=settings
            )
        except GeminiImageError as exc:
            record.text_check = {"ok": None, "error": str(exc), "expected": expected_text}
    gallery.append_records(records, settings)
    gallery.render(settings)


def _record_result(records: list[core.GeneratedRecord], settings: Settings) -> dict[str, Any]:
    return {
        "records": [r.to_dict() for r in records],
        "paths": [str((settings.output_dir / r.image_path).resolve()) for r in records],
        "publish_errors": [r.publish_error for r in records if r.publish_error],
    }


async def _run_job(job: dict[str, Any], settings: Settings) -> None:
    path = _jobs_dir(settings) / f"{job['id']}.json"
    job["status"] = "running"
    job["started_at"] = _now_iso()
    job["heartbeat"] = time.time()
    _write_json(path, job)
    hb = asyncio.create_task(_heartbeat(path, job))
    try:
        records = await asyncio.wait_for(
            _execute(job["kind"], job["params"], settings), timeout=settings.job_timeout
        )
        if job.get("expected_text"):
            await _attach_text_check(records, job["expected_text"], settings)
        job["result"] = _record_result(records, settings)
        job["status"] = "succeeded"
    except asyncio.TimeoutError:
        job["status"] = "failed"
        job["error"] = f"Timed out after {settings.job_timeout:.0f}s."
    except asyncio.CancelledError:
        job["status"] = "interrupted"
        job["error"] = "Cancelled (the server is shutting down)."
        raise
    except GeminiImageError as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["error_type"] = type(exc).__name__
    except Exception as exc:  # noqa: BLE001 - recorded, never raised into the void
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        hb.cancel()
        job["finished_at"] = _now_iso()
        job["heartbeat"] = time.time()
        _write_json(path, job)


async def wait_job(settings: Settings, job_id: str, timeout: float) -> dict[str, Any] | None:
    """Wait up to `timeout` seconds for a job to finish; return its latest state."""
    job_id = _safe_id(job_id)
    deadline = time.monotonic() + max(0.0, timeout)
    task = _tasks.get(job_id)
    if task is not None:
        await asyncio.wait({task}, timeout=max(0.0, timeout))
        return load_job(settings, job_id)
    # Running in another process (or already finished): poll the file.
    while True:
        job = load_job(settings, job_id)
        if job is None or job.get("status") not in ACTIVE or time.monotonic() >= deadline:
            return job
        await asyncio.sleep(min(2.0, max(0.1, deadline - time.monotonic())))


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def _batch_path(settings: Settings, batch_id: str) -> Path:
    return _batches_dir(settings) / f"{_safe_id(batch_id)}.json"


def load_batch(settings: Settings, batch_id: str) -> dict[str, Any] | None:
    data = _read_json(_batch_path(settings, batch_id))
    if data and data.get("status") in ACTIVE and _is_stale(data, _tasks, f"batch:{batch_id}"):
        data["status"] = "interrupted"
    return data


def list_batches(settings: Settings, limit: int = 10) -> list[dict[str, Any]]:
    files = sorted(_batches_dir(settings).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [b for b in (load_batch(settings, f.stem) for f in files[: max(1, limit)]) if b]


def _normalize_items(items: list[dict[str, Any]], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    if not items:
        raise GeminiImageError("The batch has no items.")
    if len(items) > MAX_BATCH_ITEMS:
        raise GeminiImageError(f"Too many items ({len(items)}); the limit is {MAX_BATCH_ITEMS}.")
    out, seen = [], set()
    for i, raw in enumerate(items, 1):
        if not isinstance(raw, dict) or not str(raw.get("prompt") or "").strip():
            raise GeminiImageError(f"Item {i} needs a non-empty 'prompt'.")
        unknown = set(raw) - _ITEM_KEYS
        if unknown:
            raise GeminiImageError(f"Item {i} has unknown fields: {', '.join(sorted(unknown))}.")
        item_id = str(raw.get("id") or i)
        if item_id in seen:
            raise GeminiImageError(f"Duplicate item id '{item_id}'.")
        seen.add(item_id)
        merged = {**{k: v for k, v in defaults.items() if v not in (None, "", [])}, **raw}
        tags = list(defaults.get("tags") or []) + [t for t in (raw.get("tags") or []) if t not in (defaults.get("tags") or [])]
        out.append({
            "id": item_id,
            "prompt": str(merged["prompt"]),
            "tags": tags,
            "aspect_ratio": merged.get("aspect_ratio"),
            "size": merged.get("size"),
            "project": merged.get("project"),
            "expected_text": merged.get("expected_text"),
            "model": merged.get("model"),
            "image_paths": merged.get("image_paths") or [],
            "status": "pending",
            "attempts": 0,
            "paths": [],
            "record_ids": [],
            "error": None,
            "text_check": None,
        })
    # Validate shapes up front so a typo fails the whole batch immediately, not item 37.
    from . import imaging
    for item in out:
        imaging.parse_shape(item["aspect_ratio"], item["size"])
    return out


async def start_batch(
    items: list[dict[str, Any]],
    settings: Settings,
    *,
    defaults: dict[str, Any] | None = None,
    batch_id: str | None = None,
    max_retries: int = 2,
    text_retries: int = 0,
    delay_seconds: float = 3.0,
    publish: bool | None = None,
) -> dict[str, Any]:
    """Create a batch and start running it in the background."""
    batch_id = _safe_id(batch_id) if batch_id else "b" + uuid.uuid4().hex[:12]
    path = _batch_path(settings, batch_id)
    if path.exists():
        raise GeminiImageError(
            f"Batch '{batch_id}' already exists. Use get_batch / resume_batch, or pick another id."
        )
    batch = {
        "id": batch_id,
        "status": "queued",
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "heartbeat": time.time(),
        "pid": os.getpid(),
        "options": {
            "max_retries": max(0, min(int(max_retries), 5)),
            "text_retries": max(0, min(int(text_retries), 3)),
            "delay_seconds": max(0.0, min(float(delay_seconds), 120.0)),
            "publish": publish,
        },
        "cancel_requested": False,
        "note": None,
        "items": _normalize_items(items, defaults or {}),
    }
    _write_json(path, batch)
    _launch_batch(batch, settings)
    return batch


def _launch_batch(batch: dict[str, Any], settings: Settings) -> None:
    key = f"batch:{batch['id']}"
    task = asyncio.create_task(_run_batch(batch, settings), name=f"gemini-{key}")
    _tasks[key] = task
    task.add_done_callback(lambda _t, k=key: _tasks.pop(k, None))


async def resume_batch(settings: Settings, batch_id: str, *, retry_failed: bool = True) -> dict[str, Any]:
    """Continue a paused/interrupted/finished batch: pending (and optionally failed) items."""
    batch = load_batch(settings, batch_id)
    if batch is None:
        raise GeminiImageError(f"No batch with id '{batch_id}'.")
    if batch.get("status") in ACTIVE:
        raise GeminiImageError(f"Batch '{batch_id}' is still running.")
    for item in batch["items"]:
        if item["status"] == "running" or (retry_failed and item["status"] == "failed"):
            item["status"] = "pending"
            item["attempts"] = 0
            item["error"] = None
    batch["cancel_requested"] = False
    batch["note"] = None
    batch["pid"] = os.getpid()
    batch["status"] = "queued"
    batch["heartbeat"] = time.time()
    _cancelled_batches.discard(batch["id"])
    _write_json(_batch_path(settings, batch["id"]), batch)
    _launch_batch(batch, settings)
    return batch


def cancel_batch(settings: Settings, batch_id: str) -> dict[str, Any]:
    """Ask a running batch to stop after its current item."""
    batch = load_batch(settings, batch_id)
    if batch is None:
        raise GeminiImageError(f"No batch with id '{batch_id}'.")
    _cancelled_batches.add(batch["id"])
    if batch.get("status") in ACTIVE and f"batch:{batch['id']}" not in _tasks:
        # Running in another process: leave a flag in the file for it to pick up.
        batch["cancel_requested"] = True
        _write_json(_batch_path(settings, batch["id"]), batch)
    return batch


def _cancel_requested(settings: Settings, batch: dict[str, Any]) -> bool:
    if batch["id"] in _cancelled_batches:
        return True
    on_disk = _read_json(_batch_path(settings, batch["id"])) or {}
    return bool(on_disk.get("cancel_requested"))


async def _run_batch(batch: dict[str, Any], settings: Settings) -> None:
    path = _batch_path(settings, batch["id"])
    opts = batch["options"]
    batch["status"] = "running"
    batch["started_at"] = batch.get("started_at") or _now_iso()
    _write_json(path, batch)
    hb = asyncio.create_task(_heartbeat(path, batch))
    try:
        first = True
        for item in batch["items"]:
            if item["status"] in ("done", "skipped"):
                continue
            if _cancel_requested(settings, batch):
                batch["status"] = "cancelled"
                batch["note"] = "Cancelled on request; resume_batch continues where it stopped."
                return
            if not first and opts["delay_seconds"]:
                await _sleep(opts["delay_seconds"])
            first = False
            stop = await _run_item(item, batch, settings)
            _write_json(path, batch)
            if stop:
                batch["status"] = "paused"
                batch["note"] = stop
                return
        failed = sum(1 for i in batch["items"] if i["status"] == "failed")
        batch["status"] = "completed_with_errors" if failed else "completed"
    except asyncio.CancelledError:
        batch["status"] = "interrupted"
        raise
    finally:
        hb.cancel()
        batch["finished_at"] = _now_iso()
        batch["heartbeat"] = time.time()
        _cancelled_batches.discard(batch["id"])
        _write_json(path, batch)


async def _run_item(item: dict[str, Any], batch: dict[str, Any], settings: Settings) -> str | None:
    """Generate one batch item with retries. Returns a reason to pause the batch, or None."""
    opts = batch["options"]
    rate_limit_waits = 0
    text_attempts = 0
    item["status"] = "running"
    while True:
        item["attempts"] += 1
        params: dict[str, Any] = {
            "model": item.get("model"),
            "publish": opts.get("publish"),
            "aspect_ratio": item.get("aspect_ratio"),
            "size": item.get("size"),
            "project": item.get("project"),
            "tags": [*item.get("tags", []), f"batch:{batch['id']}", f"item:{item['id']}"],
        }
        try:
            if item.get("image_paths"):
                coro = core.edit_images(item["prompt"], item["image_paths"], settings=settings, **params)
            else:
                coro = core.generate_images(item["prompt"], settings=settings, **params)
            records = await asyncio.wait_for(coro, timeout=settings.job_timeout)
        except GeminiAuthError as exc:
            item["status"] = "pending"
            item["error"] = str(exc).splitlines()[0]
            return "Gemini sign-in failed; fix the cookies (run doctor), then resume_batch."
        except GeminiUsageLimitError as exc:
            # Back off and retry the same item: 60s, 120s, 240s. Then pause the batch.
            if rate_limit_waits >= 3:
                item["status"] = "pending"
                item["error"] = str(exc)
                return "Gemini kept rate-limiting this account; wait a while, then resume_batch."
            wait = 60 * (2 ** rate_limit_waits)
            rate_limit_waits += 1
            item["attempts"] -= 1  # rate limiting isn't the prompt's fault
            item["error"] = f"Rate limited; retrying in {wait}s."
            await _sleep(wait)
            continue
        except (GeminiImageError, asyncio.TimeoutError) as exc:
            message = str(exc) or f"Timed out after {settings.job_timeout:.0f}s."
            if item["attempts"] <= opts["max_retries"]:
                item["error"] = f"Attempt {item['attempts']} failed: {message}"
                await _sleep(5 * item["attempts"])
                continue
            item["status"] = "failed"
            item["error"] = message
            return None

        item["paths"] = [str((settings.output_dir / r.image_path).resolve()) for r in records]
        item["record_ids"] = [r.id for r in records]
        item["error"] = None
        if item.get("expected_text"):
            await _attach_text_check(records, item["expected_text"], settings)
            item["text_check"] = records[0].text_check if records else None
            ok = bool(item["text_check"] and item["text_check"].get("ok"))
            if not ok and text_attempts < opts["text_retries"]:
                text_attempts += 1
                item["attempts"] -= 1
                continue
        item["status"] = "done"
        return None
