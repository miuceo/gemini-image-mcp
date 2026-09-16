"""Standalone worker process for tests/test_concurrency.py.

Not a test module itself (pytest only collects test_*.py). Invoked via `subprocess.Popen` as a
genuinely separate OS process, so it exercises the real inter-process race that
`gallery.append_records`'s file lock has to defend against - not just a threading race within
one Python process.

Usage:
    python _worker_append.py <mode> <output_dir> <worker_id> <count> <delay>

    mode: "locked"   -> calls gallery.append_records (goes through the file lock)
          "unlocked" -> calls gallery._do_append directly (skips the lock - the "before" control)
    output_dir: directory containing manifest.json
    worker_id: unique string prefix used to build unique record ids for this process
    count: number of records this process appends, one append call per record
    delay: seconds to sleep between reading and writing the manifest, widening the race window
           deterministically (passed straight through to the internal `_test_delay` hook)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from gemini_image_mcp import gallery
from gemini_image_mcp.core import GeneratedRecord


def main() -> None:
    mode, output_dir_raw, worker_id, count_raw, delay_raw = sys.argv[1:6]
    output_dir = Path(output_dir_raw)
    count = int(count_raw)
    delay = float(delay_raw)

    settings = SimpleNamespace(output_dir=output_dir)
    manifest_path = output_dir / gallery.MANIFEST_NAME

    for i in range(count):
        record = GeneratedRecord(
            id=f"{worker_id}-{i}",
            prompt=f"prompt from {worker_id} #{i}",
            model="test-model",
            created_at="2026-01-01T00:00:00+00:00",
            image_path=f"images/{worker_id}-{i}.png",
            kind="generate",
            source_images=[],
        )
        if mode == "locked":
            gallery.append_records([record], settings, _test_delay=delay)
        elif mode == "unlocked":
            # Bypasses the lock entirely: this is the pre-fix code path, used only to prove
            # the race is real and that the test would catch a regression.
            gallery._do_append([record], output_dir, manifest_path, _test_delay=delay)
        else:
            raise ValueError(f"unknown mode: {mode!r}")


if __name__ == "__main__":
    main()
