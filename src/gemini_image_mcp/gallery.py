"""Cumulative HTML gallery for generated/edited images.

Maintains a JSON manifest (``manifest.json``) inside the configured output
directory and renders it into a single self-contained ``gallery.html`` file
using a stdlib-only template substitution (no Jinja2, no external deps).

Public API:
    append_records(records, settings) -> None
    load_records(settings) -> list[GeneratedRecord]
    render(settings) -> Path
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from importlib import resources
from typing import TYPE_CHECKING, Sequence

import filelock

if TYPE_CHECKING:
    from .core import GeneratedRecord

MANIFEST_NAME = "manifest.json"
GALLERY_NAME = "gallery.html"
LOCK_NAME = "manifest.json.lock"
TEMPLATE_PACKAGE = "gemini_image_mcp.templates"
TEMPLATE_NAME = "gallery.html"

RECORDS_TOKEN = "/*__RECORDS_JSON__*/"
GENERATED_AT_TOKEN = "__GENERATED_AT__"

# Multiple MCP server processes (e.g. two concurrent Claude Code sessions) can each call
# append_records() against the same manifest.json. The final write is atomic (os.replace),
# but the read -> merge -> write sequence is not: two processes can both read the same
# "before" state, merge their own new record in, and then race to os.replace() - the loser's
# record is silently dropped even though its image file was already saved to disk. This is a
# real inter-process race (separate OS processes), not just a threading issue within one
# process, so it needs an inter-process lock, not an `asyncio.Lock`/`threading.Lock`.
#
# `filelock` is used (rather than a hand-rolled O_CREAT|O_EXCL lock file) because:
#   - It has correct, well-tested Windows support (this project's primary target platform),
#     using `msvcrt.locking` under the hood. `fcntl` (the natural POSIX choice) does not
#     exist on Windows at all.
#   - Stale-lock recovery comes for free: both its Windows and POSIX backends acquire an
#     OS-level lock tied to the *file handle/process*, not a sentinel file whose presence has
#     to be interpreted. If the holding process crashes or is killed, the OS releases the
#     lock automatically - there is no "is this lock file stale?" heuristic to get wrong,
#     which a plain O_CREAT|O_EXCL lock-file implementation would otherwise need (e.g. a
#     stat-based mtime check racing against a process that is merely slow, not dead).
#   - It's a small, dependency-light, actively maintained package with no compiled extension
#     to build on Windows.
MANIFEST_LOCK_TIMEOUT = 10  # seconds


class ManifestLockTimeout(Exception):
    """Raised when the manifest file lock could not be acquired within the timeout.

    This means another gemini-image-mcp process is (or was) holding the lock while writing
    to the same manifest.json - most commonly two concurrent MCP server sessions pointed at
    the same output directory. Raised instead of letting an MCP tool call hang forever.
    """


def _manifest_path(settings) -> "object":
    return settings.output_dir / MANIFEST_NAME


def _lock_path(settings) -> "object":
    return settings.output_dir / LOCK_NAME


def _gallery_path(settings) -> "object":
    return settings.output_dir / GALLERY_NAME


def _read_manifest_raw(manifest_path) -> list[dict]:
    """Read the manifest file, tolerating a missing/empty/corrupt file.

    A corrupt file is moved aside to ``manifest.corrupt.json`` (best effort)
    and an empty list is returned so the caller can start fresh.
    """
    if not manifest_path.exists():
        return []

    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return []

    if not text.strip():
        return []

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        corrupt_path = manifest_path.with_name("manifest.corrupt.json")
        try:
            os.replace(manifest_path, corrupt_path)
        except OSError:
            pass
        return []

    if not isinstance(data, list):
        corrupt_path = manifest_path.with_name("manifest.corrupt.json")
        try:
            os.replace(manifest_path, corrupt_path)
        except OSError:
            pass
        return []

    return [d for d in data if isinstance(d, dict)]


def _do_append(
    records: Sequence["GeneratedRecord"],
    output_dir,
    manifest_path,
    *,
    _test_delay: float = 0.0,
) -> None:
    """Read-merge-write the manifest. Callers MUST hold the manifest lock (or accept the race).

    Split out of `append_records` so tests can call it directly, unlocked, to demonstrate the
    data-loss race it would otherwise be exposed to (see `tests/test_concurrency.py`).

    `_test_delay` is a private test-only hook: it sleeps after reading the "before" state and
    before writing, to deterministically widen the read-modify-write race window under test.
    It defaults to 0 (no-op) and is never set by any real call path.
    """
    existing = _read_manifest_raw(manifest_path)

    if _test_delay:
        time.sleep(_test_delay)

    by_id: dict[str, dict] = {}
    order: list[str] = []
    for d in existing:
        rid = d.get("id")
        if rid is None:
            continue
        if rid not in by_id:
            order.append(rid)
        by_id[rid] = d

    for record in records:
        d = record.to_dict()
        rid = d.get("id")
        if rid is None:
            continue
        if rid not in by_id:
            order.append(rid)
        by_id[rid] = d

    merged = [by_id[rid] for rid in order]

    fd, tmp_path = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=str(output_dir)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, manifest_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def append_records(
    records: Sequence["GeneratedRecord"], settings, *, _test_delay: float = 0.0
) -> None:
    """Atomically append ``records`` to the manifest, de-duplicated by id.

    The entire read -> merge -> write sequence runs under an inter-process file lock (see the
    module docstring above `ManifestLockTimeout`), so concurrent processes appending to the
    same manifest can never silently drop each other's records.
    """
    from .core import GeneratedRecord  # noqa: F401  (lazy, avoids circular import)

    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(settings)
    lock_path = _lock_path(settings)

    lock = filelock.FileLock(str(lock_path), timeout=MANIFEST_LOCK_TIMEOUT)
    try:
        with lock:
            _do_append(records, output_dir, manifest_path, _test_delay=_test_delay)
    except filelock.Timeout as exc:
        raise ManifestLockTimeout(
            f"Could not acquire the gallery manifest lock at '{lock_path}' within "
            f"{MANIFEST_LOCK_TIMEOUT}s. Another gemini-image-mcp process is likely writing "
            "to the same output directory at the same time. If no other process is running "
            "and this persists, it may be a stale lock left by a killed process - delete "
            "the lock file and retry."
        ) from exc


def load_records(settings) -> list["GeneratedRecord"]:
    """Load all records from the manifest, newest first.

    Deliberately does NOT take the manifest lock. A torn read is impossible: the manifest is
    only ever mutated via `os.replace()` in `_do_append`, which is atomic on both POSIX and
    Windows, so any reader always sees either the fully-old or fully-new file, never a partial
    write. The only thing skipping the lock risks is a *stale* read (a concurrent writer's
    update not yet visible) - acceptable here because this is used for read-only display
    (`render()`) and a subsequent render will pick up the update; it is not a value anything
    is written back based on, so there is nothing to lose by not locking it.
    """
    from .core import GeneratedRecord

    manifest_path = _manifest_path(settings)
    raw = _read_manifest_raw(manifest_path)

    records: list[GeneratedRecord] = []
    for d in raw:
        try:
            records.append(GeneratedRecord.from_dict(d))
        except (KeyError, TypeError, ValueError):
            continue

    records.sort(key=lambda r: r.created_at, reverse=True)
    return records


def _escape_for_script(json_text: str) -> str:
    """Escape sequences that could break out of an inline <script> block."""
    return (
        json_text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def render(settings) -> "object":
    """Render the gallery HTML from the current manifest. Returns the path."""
    records = load_records(settings)
    records_json = json.dumps([r.to_dict() for r in records], ensure_ascii=False)
    records_json = _escape_for_script(records_json)

    from datetime import datetime, timezone

    generated_at = datetime.now(timezone.utc).isoformat()

    template_text = (
        resources.files(TEMPLATE_PACKAGE).joinpath(TEMPLATE_NAME).read_text(encoding="utf-8")
    )

    rendered = template_text.replace(RECORDS_TOKEN, records_json)
    rendered = rendered.replace(GENERATED_AT_TOKEN, generated_at)

    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = _gallery_path(settings)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".gallery.", suffix=".tmp", dir=str(output_dir)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(rendered)
        os.replace(tmp_path, gallery_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return gallery_path.resolve()
