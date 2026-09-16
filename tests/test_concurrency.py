"""Inter-process concurrency test for gallery.append_records (Task 1).

Spawns real OS subprocesses (not threads) that each append records to the same manifest.json
at (as close to) the same time, to exercise the actual race two concurrent MCP server
processes can hit. Two directions are asserted:

  * test_unlocked_append_loses_records_under_race - calls the pre-lock code path
    (`gallery._do_append` directly, skipping the file lock) and demonstrates that records get
    silently dropped. This is the "before the fix" control: if this test ever starts passing,
    it means the race stopped being reproducible and the positive test below is no longer
    meaningful.
  * test_locked_append_records_never_lost_under_race - calls the real public
    `gallery.append_records` (which takes the file lock) under the identical race conditions
    and asserts zero records are lost.

Both use the internal `_test_delay` hook to deterministically widen the read-modify-write
window (see `gallery._do_append`), so the outcome doesn't depend on timing luck / CI slowness.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKER = str(Path(__file__).parent / "_worker_append.py")

NUM_WORKERS = 5
RECORDS_PER_WORKER = 6
RACE_DELAY = 0.15  # seconds


def _run_workers(mode: str, output_dir: Path) -> list[int]:
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                WORKER,
                mode,
                str(output_dir),
                f"worker{i}",
                str(RECORDS_PER_WORKER),
                str(RACE_DELAY),
            ]
        )
        for i in range(NUM_WORKERS)
    ]
    return [p.wait(timeout=60) for p in procs]


def _manifest_ids(output_dir: Path) -> set[str]:
    manifest_path = output_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {d["id"] for d in data}


def test_unlocked_append_loses_records_under_race(tmp_path):
    """Control: without the lock, concurrent read-modify-write cycles corrupt the manifest.

    On this platform the unsynchronized race manifests as `os.replace()` raising
    `PermissionError` (WinError 5) when one process tries to replace the manifest while
    another has it open for reading - a harder failure than silent data loss, but still
    exactly the kind of crash an MCP tool call must not suffer under concurrent sessions.
    On platforms where the race instead resolves silently, it shows up as missing record ids.
    Either way, "some worker process failed" or "records went missing" proves the unlocked
    path is unsafe; a passing run of this test on unlocked code would mean the race stopped
    being reproducible, which would make the companion locked-path test below meaningless.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    results = _run_workers("unlocked", output_dir)

    expected = {f"worker{i}-{j}" for i in range(NUM_WORKERS) for j in range(RECORDS_PER_WORKER)}
    actual = _manifest_ids(output_dir) if (output_dir / "manifest.json").exists() else set()

    some_worker_failed = any(rc != 0 for rc in results)
    records_lost = actual != expected

    assert some_worker_failed or records_lost, (
        "expected the unlocked code path to either crash or lose records under this race, "
        f"but all {len(results)} workers exited 0 and every record survived - the race is no "
        "longer reproducible with these parameters"
    )


def test_locked_append_records_never_lost_under_race(tmp_path):
    """The real fix: append_records() must never lose data or crash under the identical race."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    results = _run_workers("locked", output_dir)

    assert all(rc == 0 for rc in results), f"a worker process failed: {results}"

    expected = {f"worker{i}-{j}" for i in range(NUM_WORKERS) for j in range(RECORDS_PER_WORKER)}
    actual = _manifest_ids(output_dir)

    assert actual == expected
    assert len(actual) == NUM_WORKERS * RECORDS_PER_WORKER
