"""Tests for tui.settings_io — GlobalSettings and ProjectSettings persistence."""

from __future__ import annotations

from ck_engine.settings import GlobalSettings, ProjectSettings


class TestGlobalSettings:
    def test_defaults(self):
        s = GlobalSettings()
        assert s.device == "auto"
        assert s.precision == "fp16"
        assert s.recent_projects == []
        assert s.flash_attention is None

    def test_round_trip(self, tmp_path):
        path = tmp_path / "settings.toml"
        original = GlobalSettings(
            device="cuda",
            backend="torch_optimized",
            devices=["cuda:0", "cuda:1"],
            img_size=1024,
            precision="bf16",
            flash_attention=True,
            tiled_refiner=False,
            cache_clearing=None,
            comp_format="png",
            comp_checkerboard=True,
            dma_buffers=3,
            recent_projects=["/proj/a", "/proj/b"],
        )
        original.save(path)
        loaded = GlobalSettings.load(path)

        assert loaded.device == "cuda"
        assert loaded.backend == "torch_optimized"
        assert loaded.devices == ["cuda:0", "cuda:1"]
        assert loaded.img_size == 1024
        assert loaded.precision == "bf16"
        assert loaded.flash_attention is True
        assert loaded.tiled_refiner is False
        assert loaded.cache_clearing is None
        assert loaded.comp_format == "png"
        assert loaded.comp_checkerboard is True
        assert loaded.dma_buffers == 3
        assert loaded.recent_projects == ["/proj/a", "/proj/b"]

    def test_missing_file_returns_defaults(self, tmp_path):
        s = GlobalSettings.load(tmp_path / "nonexistent.toml")
        assert s.device == "auto"

    def test_missing_keys_use_defaults(self, tmp_path):
        path = tmp_path / "partial.toml"
        path.write_text('_version = 1\ndevice = "mps"\n')
        s = GlobalSettings.load(path)
        assert s.device == "mps"
        assert s.precision == "fp16"  # default

    def test_extra_keys_ignored(self, tmp_path):
        path = tmp_path / "extra.toml"
        path.write_text('_version = 1\ndevice = "cpu"\nfuture_key = "value"\n')
        s = GlobalSettings.load(path)
        assert s.device == "cpu"

    def test_add_recent_project(self):
        s = GlobalSettings()
        s.add_recent_project("/proj/a")
        s.add_recent_project("/proj/b")
        assert s.recent_projects == ["/proj/b", "/proj/a"]

    def test_add_recent_project_dedupes(self):
        s = GlobalSettings(recent_projects=["/proj/a", "/proj/b", "/proj/c"])
        s.add_recent_project("/proj/b")
        assert s.recent_projects == ["/proj/b", "/proj/a", "/proj/c"]

    def test_add_recent_project_max_entries(self):
        s = GlobalSettings(recent_projects=[f"/proj/{i}" for i in range(20)])
        s.add_recent_project("/proj/new")
        assert len(s.recent_projects) == 20
        assert s.recent_projects[0] == "/proj/new"


class TestProjectSettings:
    def test_defaults(self):
        s = ProjectSettings()
        assert s.input_is_linear is False
        assert s.alpha_model == "birefnet"
        assert s.alpha_mode == "replace"

    def test_round_trip(self, tmp_path):
        original = ProjectSettings(
            input_is_linear=True,
            despill_strength=0.8,
            auto_despeckle=False,
            despeckle_size=200,
            refiner_scale=1.5,
            alpha_model="gvm",
            alpha_mode="fill",
        )
        original.save(tmp_path)
        loaded = ProjectSettings.load(tmp_path)

        assert loaded.input_is_linear is True
        assert loaded.despill_strength == 0.8
        assert loaded.auto_despeckle is False
        assert loaded.despeckle_size == 200
        assert loaded.refiner_scale == 1.5
        assert loaded.alpha_model == "gvm"
        assert loaded.alpha_mode == "fill"

    def test_missing_file_returns_defaults(self, tmp_path):
        s = ProjectSettings.load(tmp_path)
        assert s.despill_strength == 0.5

    def test_missing_keys_use_defaults(self, tmp_path):
        import json

        path = tmp_path / ".corridorkey_settings.json"
        path.write_text(json.dumps({"_version": 1, "alpha_model": "videomama"}))
        s = ProjectSettings.load(tmp_path)
        assert s.alpha_model == "videomama"
        assert s.despill_strength == 0.5  # default

    def test_extra_keys_ignored(self, tmp_path):
        import json

        path = tmp_path / ".corridorkey_settings.json"
        path.write_text(json.dumps({"_version": 1, "future_key": "val", "alpha_model": "gvm"}))
        s = ProjectSettings.load(tmp_path)
        assert s.alpha_model == "gvm"

    def test_schema_version_present(self, tmp_path):
        import json

        ProjectSettings().save(tmp_path)
        raw = json.loads((tmp_path / ".corridorkey_settings.json").read_text())
        assert raw["_version"] == 1
