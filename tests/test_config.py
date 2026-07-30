"""Settings resolution: where the library lives, and how paths are stored."""

from __future__ import annotations

from pathlib import Path

import pytest

from biotimbre.config import ENV_CONFIG, Settings, find_config_file, load_settings


def write_config(directory: Path, body: str, name: str = "biotimbre.toml") -> Path:
    path = directory / name
    path.write_text(body, "utf-8")
    return path


class TestDataDirResolution:
    def test_defaults_to_data_beside_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        settings = load_settings()
        assert settings.data_dir == tmp_path / "data"
        assert settings.db_path == tmp_path / "data" / "library.db"
        assert settings.audio_dir == tmp_path / "data" / "audio"

    def test_explicit_data_dir_wins(self, tmp_path):
        settings = load_settings(data_dir=tmp_path / "elsewhere")
        assert settings.data_dir == tmp_path / "elsewhere"

    def test_data_dir_overrides_the_config_file(self, tmp_path):
        config = write_config(tmp_path, '[library]\ndata_dir = "from_config"\n')
        settings = load_settings(config, data_dir=tmp_path / "from_flag")
        # --data-dir is the last word, so a one-off invocation can point anywhere.
        assert settings.data_dir == tmp_path / "from_flag"

    def test_relative_paths_resolve_against_the_config_file(self, tmp_path):
        nested = tmp_path / "somewhere"
        nested.mkdir()
        config = write_config(nested, '[library]\ndata_dir = "mylib"\n')
        settings = load_settings(config)
        # Not the cwd: a config file and its library can be moved together.
        assert settings.data_dir == nested / "mylib"

    def test_absolute_paths_are_used_as_given(self, tmp_path):
        config = write_config(
            tmp_path, f'[library]\ndata_dir = "{tmp_path / "abs"}"\n'
        )
        assert load_settings(config).data_dir == tmp_path / "abs"

    def test_db_and_audio_can_be_placed_independently(self, tmp_path):
        config = write_config(
            tmp_path,
            f'[library]\ndata_dir = "{tmp_path / "lib"}"\n'
            f'db_path = "{tmp_path / "fast" / "library.db"}"\n'
            f'audio_dir = "{tmp_path / "bulk" / "audio"}"\n',
        )
        settings = load_settings(config)
        assert settings.db_path == tmp_path / "fast" / "library.db"
        assert settings.audio_dir == tmp_path / "bulk" / "audio"

    def test_user_home_is_expanded(self, tmp_path):
        config = write_config(tmp_path, '[library]\ndata_dir = "~/biotimbre"\n')
        assert load_settings(config).data_dir == Path.home() / "biotimbre"


class TestConfigDiscovery:
    def test_no_config_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(ENV_CONFIG, raising=False)
        assert find_config_file() is None

    def test_finds_biotimbre_toml_in_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(ENV_CONFIG, raising=False)
        config = write_config(tmp_path, "")
        assert find_config_file() == config.resolve()

    def test_environment_variable_is_honoured(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, "", name="custom.toml")
        monkeypatch.setenv(ENV_CONFIG, str(config))
        assert find_config_file() == config.resolve()

    def test_explicit_path_beats_the_environment(self, tmp_path, monkeypatch):
        env_config = write_config(tmp_path, "", name="env.toml")
        explicit = write_config(tmp_path, "", name="explicit.toml")
        monkeypatch.setenv(ENV_CONFIG, str(env_config))
        assert find_config_file(explicit) == explicit.resolve()

    def test_missing_requested_config_is_an_error(self, tmp_path):
        # A missing implicit config is fine; a missing *requested* one is a typo.
        with pytest.raises(FileNotFoundError):
            find_config_file(tmp_path / "nope.toml")

    def test_missing_env_config_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_CONFIG, str(tmp_path / "nope.toml"))
        with pytest.raises(FileNotFoundError, match=ENV_CONFIG):
            find_config_file()


class TestStoredAudioPaths:
    def _settings(self, root: Path, audio: Path | None = None) -> Settings:
        return Settings(
            data_dir=root,
            db_path=root / "library.db",
            audio_dir=audio if audio is not None else root / "audio",
        )

    def test_paths_inside_the_library_are_stored_relative(self, tmp_path):
        settings = self._settings(tmp_path / "lib")
        stored = settings.relative_audio_path(
            settings.audio_path_for("00A3", 7)
        )
        # Relative, so moving or renaming the library keeps every reference valid.
        assert stored == "audio/00A3/7.wav"

    def test_relative_round_trip(self, tmp_path):
        settings = self._settings(tmp_path / "lib")
        absolute = settings.audio_path_for("00A3", 7)
        stored = settings.relative_audio_path(absolute)
        assert settings.absolute_audio_path(stored) == absolute

    def test_a_moved_library_still_resolves(self, tmp_path):
        original = self._settings(tmp_path / "before")
        stored = original.relative_audio_path(original.audio_path_for("0001", 3))
        moved = self._settings(tmp_path / "after")
        assert moved.absolute_audio_path(stored) == (
            tmp_path / "after" / "audio" / "0001" / "3.wav"
        )

    def test_audio_outside_the_library_is_stored_absolute(self, tmp_path):
        # Database on fast storage, audio on a large drive. This used to raise a
        # bare ValueError at ingest, long after the config had loaded happily.
        settings = self._settings(tmp_path / "lib", audio=tmp_path / "bulk")
        absolute = settings.audio_path_for("0000", 1)
        stored = settings.relative_audio_path(absolute)
        assert Path(stored).is_absolute()
        assert settings.absolute_audio_path(stored) == absolute

    def test_absolute_stored_path_ignores_data_dir(self, tmp_path):
        settings = self._settings(tmp_path / "lib", audio=tmp_path / "bulk")
        stored = str(tmp_path / "bulk" / "0000" / "1.wav")
        assert settings.absolute_audio_path(stored) == Path(stored)


class TestValidation:
    def test_unsupported_language_is_rejected(self, tmp_path):
        config = write_config(tmp_path, '[ui]\nlanguage = "klingon"\n')
        with pytest.raises(ValueError, match="Unsupported language"):
            load_settings(config)

    def test_empty_model_list_is_rejected(self, tmp_path):
        config = write_config(tmp_path, "[models]\nactive = []\n")
        with pytest.raises(ValueError, match="at least one model"):
            load_settings(config)

    def test_single_model_string_is_accepted(self, tmp_path):
        config = write_config(tmp_path, '[models]\nactive = "perch_v2"\n')
        assert load_settings(config).active_models == ("perch_v2",)

    @pytest.mark.parametrize(
        "body,message",
        [
            ("[search]\ntop_k = 0\n", "top_k"),
            ("[search]\nbatch_size = 0\n", "batch_size"),
            ("[search]\nquery_cache_entries = 0\n", "query_cache_entries"),
            ("[search]\nmax_query_duration_s = 0\n", "max_query_duration_s"),
            ("[models]\ninference_batch_size = 0\n", "inference_batch_size"),
            ("[models]\ninference_threads = -1\n", "inference_threads"),
        ],
    )
    def test_out_of_range_numbers_are_rejected(self, tmp_path, body, message):
        config = write_config(tmp_path, body)
        with pytest.raises(ValueError, match=message):
            load_settings(config)

    def test_defaults_survive_a_partial_config(self, tmp_path):
        config = write_config(tmp_path, '[search]\ntop_k = 3\n')
        settings = load_settings(config)
        assert settings.default_top_k == 3
        # Everything unspecified keeps its default.
        assert settings.default_metric == "cosine"
        assert settings.max_query_duration_s == 15.0
        assert settings.active_models == ("perch_v2", "birdnet_v2.4")
