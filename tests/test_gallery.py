"""Tests for gemini_image_mcp.gallery: append_records, load_records, render."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gemini_image_mcp import gallery
from gemini_image_mcp.core import GeneratedRecord


def _settings(tmp_path):
    return SimpleNamespace(output_dir=tmp_path)


def _record(rid: str, **overrides) -> GeneratedRecord:
    defaults = dict(
        id=rid,
        prompt=f"prompt {rid}",
        model="test-model",
        created_at="2026-01-01T00:00:00+00:00",
        image_path=f"images/{rid}.png",
        kind="generate",
        source_images=[],
    )
    defaults.update(overrides)
    return GeneratedRecord(**defaults)


# ---------------------------------------------------------------------------
# append_records
# ---------------------------------------------------------------------------


def test_append_records_creates_manifest_when_missing(tmp_path):
    settings = _settings(tmp_path)
    gallery.append_records([_record("a")], settings)

    manifest_path = tmp_path / gallery.MANIFEST_NAME
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == ["a"]


def test_append_records_is_cumulative(tmp_path):
    settings = _settings(tmp_path)
    gallery.append_records([_record("a")], settings)
    gallery.append_records([_record("b")], settings)

    data = json.loads((tmp_path / gallery.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {d["id"] for d in data} == {"a", "b"}


def test_append_records_dedupes_by_id_keeping_latest(tmp_path):
    settings = _settings(tmp_path)
    gallery.append_records([_record("a", prompt="first")], settings)
    gallery.append_records([_record("a", prompt="second")], settings)

    data = json.loads((tmp_path / gallery.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["prompt"] == "second"


def test_append_records_preserves_first_seen_order_for_unchanged_ids(tmp_path):
    settings = _settings(tmp_path)
    gallery.append_records([_record("a"), _record("b")], settings)
    gallery.append_records([_record("a", prompt="updated")], settings)

    data = json.loads((tmp_path / gallery.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == ["a", "b"]
    assert data[0]["prompt"] == "updated"


def test_append_records_empty_list_is_a_noop_on_existing_manifest(tmp_path):
    settings = _settings(tmp_path)
    gallery.append_records([_record("a")], settings)
    gallery.append_records([], settings)

    data = json.loads((tmp_path / gallery.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == ["a"]


def test_append_records_writes_atomically_no_leftover_tmp_files(tmp_path):
    settings = _settings(tmp_path)
    gallery.append_records([_record("a")], settings)

    leftovers = list(tmp_path.glob(".manifest.*.tmp"))
    assert leftovers == []


def test_append_records_recovers_from_corrupt_manifest(tmp_path):
    settings = _settings(tmp_path)
    manifest_path = tmp_path / gallery.MANIFEST_NAME
    manifest_path.write_text("{not valid json::", encoding="utf-8")

    gallery.append_records([_record("a")], settings)

    corrupt_path = tmp_path / "manifest.corrupt.json"
    assert corrupt_path.exists()
    assert corrupt_path.read_text(encoding="utf-8") == "{not valid json::"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == ["a"]


def test_append_records_recovers_from_manifest_that_is_not_a_json_list(tmp_path):
    settings = _settings(tmp_path)
    manifest_path = tmp_path / gallery.MANIFEST_NAME
    manifest_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    gallery.append_records([_record("a")], settings)

    assert (tmp_path / "manifest.corrupt.json").exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == ["a"]


def test_append_records_handles_empty_manifest_file(tmp_path):
    settings = _settings(tmp_path)
    manifest_path = tmp_path / gallery.MANIFEST_NAME
    manifest_path.write_text("", encoding="utf-8")

    gallery.append_records([_record("a")], settings)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == ["a"]


def test_append_records_skips_entries_missing_id(tmp_path):
    settings = _settings(tmp_path)
    manifest_path = tmp_path / gallery.MANIFEST_NAME
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([{"prompt": "no id here"}]), encoding="utf-8")

    gallery.append_records([_record("a")], settings)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == ["a"]


# ---------------------------------------------------------------------------
# load_records
# ---------------------------------------------------------------------------


def test_load_records_missing_manifest_returns_empty(tmp_path):
    settings = _settings(tmp_path)
    assert gallery.load_records(settings) == []


def test_load_records_sorts_newest_first(tmp_path):
    settings = _settings(tmp_path)
    gallery.append_records(
        [
            _record("old", created_at="2026-01-01T00:00:00+00:00"),
            _record("new", created_at="2026-06-01T00:00:00+00:00"),
        ],
        settings,
    )

    records = gallery.load_records(settings)
    assert [r.id for r in records] == ["new", "old"]


def test_load_records_skips_malformed_entries(tmp_path):
    settings = _settings(tmp_path)
    manifest_path = tmp_path / gallery.MANIFEST_NAME
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([{"id": "bad"}, _record("good").to_dict()]), encoding="utf-8"
    )

    records = gallery.load_records(settings)
    assert [r.id for r in records] == ["good"]


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_with_records_embeds_json_and_returns_path(tmp_path):
    settings = _settings(tmp_path)
    gallery.append_records([_record("a", prompt="a cat")], settings)

    gallery_path = gallery.render(settings)

    assert gallery_path.exists()
    html = gallery_path.read_text(encoding="utf-8")
    assert 'id="records-data"' in html
    assert "a cat" in html
    assert "GENERATED_AT" in html


def test_render_with_empty_manifest_does_not_raise(tmp_path):
    settings = _settings(tmp_path)
    gallery_path = gallery.render(settings)

    assert gallery_path.exists()
    html = gallery_path.read_text(encoding="utf-8")
    assert '<script type="application/json" id="records-data">[]</script>' in html


def test_render_escapes_script_closing_tag_in_prompt(tmp_path):
    """A prompt containing `</script>` must not be able to break out of the JSON <script> block."""
    settings = _settings(tmp_path)
    malicious_prompt = 'nice image</script><script>alert(1)</script>'
    gallery.append_records([_record("a", prompt=malicious_prompt)], settings)

    gallery_path = gallery.render(settings)
    html = gallery_path.read_text(encoding="utf-8")

    # The raw closing tag must never appear verbatim inside the embedded JSON payload.
    assert "</script><script>alert(1)</script>" not in html
    # It must have been escaped (unicode-escaped '<' / '>') rather than dropped.
    assert "\\u003c/script\\u003e" in html
    # The rest of the document must still be intact (one real closing </script> per block).
    assert html.count("<script") >= 2


def test_render_is_idempotent_and_reflects_latest_manifest_state(tmp_path):
    settings = _settings(tmp_path)
    gallery.append_records([_record("a")], settings)
    gallery.render(settings)

    gallery.append_records([_record("b")], settings)
    gallery_path = gallery.render(settings)

    html = gallery_path.read_text(encoding="utf-8")
    assert '"a"' in html and '"b"' in html
