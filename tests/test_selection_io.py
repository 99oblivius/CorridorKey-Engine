"""Tests for tui.selection_io — SelectionMap persistence."""

from __future__ import annotations

import json

from tui.selection_io import SelectionMap


class TestSelectionMap:
    def test_default_is_empty(self):
        s = SelectionMap()
        assert s.is_default()
        assert s.clips == {}

    def test_non_default(self):
        s = SelectionMap(clips={"clip_001": [0, 1, 2]})
        assert not s.is_default()

    def test_save_deletes_file_when_default(self, tmp_path):
        path = tmp_path / ".corridorkey_selection.json"
        # Create a file first
        path.write_text("{}")
        assert path.exists()

        # Save default selection → file should be deleted
        SelectionMap().save(tmp_path)
        assert not path.exists()

    def test_save_noop_when_default_and_no_file(self, tmp_path):
        """Saving default selection when file doesn't exist is a no-op."""
        SelectionMap().save(tmp_path)
        assert not (tmp_path / ".corridorkey_selection.json").exists()

    def test_round_trip(self, tmp_path):
        original = SelectionMap(clips={
            "clip_001": [0, 5, 10],
            "clip_002": [],
            "clip_003": None,
        })
        original.save(tmp_path)
        loaded = SelectionMap.load(tmp_path)

        assert loaded.clips["clip_001"] == [0, 5, 10]
        assert loaded.clips["clip_002"] == []
        assert loaded.clips["clip_003"] is None

    def test_load_missing_file_returns_default(self, tmp_path):
        s = SelectionMap.load(tmp_path)
        assert s.is_default()

    def test_schema_version_present(self, tmp_path):
        SelectionMap(clips={"clip_001": [0]}).save(tmp_path)
        raw = json.loads((tmp_path / ".corridorkey_selection.json").read_text())
        assert raw["_version"] == 1

    def test_load_corrupt_file_returns_default(self, tmp_path):
        path = tmp_path / ".corridorkey_selection.json"
        path.write_text("not valid json{{{")
        s = SelectionMap.load(tmp_path)
        assert s.is_default()
