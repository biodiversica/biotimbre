"""Sidecar parsing, against the real example files."""

from __future__ import annotations

import pytest

from biotimbre.domain.errors import ValidationError
from biotimbre.io.sidecar import (
    draft_from_fields,
    draft_from_sidecar,
    find_sidecar,
    parse_sidecar,
    read_sidecar,
)
from conftest import EXAMPLES_DIR


class TestRealExamples:
    def test_parses_guyvir(self):
        fields = read_sidecar(EXAMPLES_DIR / "GUYVIR.txt")
        assert fields == {
            "date": "2020-01-12",
            "time": "09:40:21",
            "sample_window": "34,39",
            "location": "-26,-48",
            "device": "Audiomoth",
            "filtered_band": "2000,-",
        }

    def test_parses_zamtym(self):
        fields = read_sidecar(EXAMPLES_DIR / "ZAMTYM.txt")
        assert fields["taxa"] == "cicadidae"
        assert fields["filtered_band"] == "2000,10000"

    def test_guyvir_draft_has_unknown_taxa_and_open_band(self):
        draft = draft_from_sidecar(EXAMPLES_DIR / "GUYVIR.txt")
        # No taxa line at all in this file, which the spec says means unknown.
        assert draft.taxa == ()
        assert draft.taxa_display == "unknown"
        assert draft.filtered_band.low == 2000.0
        assert draft.filtered_band.high is None
        assert draft.device == "Audiomoth"
        assert draft.sample_window.start == 34.0
        assert draft.speed == 1.0

    def test_zamtym_draft(self):
        draft = draft_from_sidecar(EXAMPLES_DIR / "ZAMTYM.txt")
        assert draft.taxa == ("cicadidae",)
        assert draft.location.latitude == -20.0
        assert draft.filtered_band.high == 10000.0

    def test_finds_sidecar_beside_audio(self):
        found = find_sidecar(EXAMPLES_DIR / "GUYVIR.wav")
        assert found is not None
        assert found.name == "GUYVIR.txt"

    def test_returns_none_when_no_sidecar(self, tmp_path):
        (tmp_path / "lonely.wav").write_bytes(b"")
        assert find_sidecar(tmp_path / "lonely.wav") is None


class TestParsing:
    def test_splits_on_first_whitespace_run_only(self):
        # Both tabs and spaces separate name from value, and values may contain
        # spaces (device names, notes).
        fields = parse_sidecar("device\tAudioMoth 1.2.0\nnotes  two  words here")
        assert fields["device"] == "AudioMoth 1.2.0"
        assert fields["notes"] == "two  words here"

    def test_ignores_blank_lines_and_comments(self):
        fields = parse_sidecar("# a comment\n\ndate 2020-01-01\n; another\n")
        assert fields == {"date": "2020-01-01"}

    def test_field_names_are_case_insensitive(self):
        assert parse_sidecar("DATE 2020-01-01")["date"] == "2020-01-01"

    def test_field_with_no_value_is_empty_string(self):
        assert parse_sidecar("notes")["notes"] == ""

    def test_rejects_unknown_field(self):
        # A typo like 'locaton' would otherwise silently drop a mandatory value
        # and surface later as a confusing "missing field" error.
        with pytest.raises(ValidationError, match="unknown field 'locaton'"):
            parse_sidecar("locaton -26,-48")

    def test_rejects_duplicate_field(self):
        with pytest.raises(ValidationError, match="appears twice"):
            parse_sidecar("date 2020-01-01\ndate 2021-01-01")

    def test_reports_line_number(self):
        with pytest.raises(ValidationError, match="line 3"):
            parse_sidecar("date 2020-01-01\ntime 09:00:00\nbogus x")

    def test_accepts_speed_field(self):
        draft = draft_from_fields(
            {
                "date": "2020-01-01",
                "time": "09:00:00",
                "location": "-26,-48",
                "sample_window": "0,5",
                "speed": "0.25",
            }
        )
        assert draft.speed == 0.25

    def test_accepts_new_optional_fields(self):
        draft = draft_from_fields(
            {
                "date": "2020-01-01",
                "time": "09:00:00",
                "location": "-26,-48",
                "sample_window": "0,5",
                "microphone": "SMX-II",
                "recorded_by": "J. Doe",
                "filter_type": "Butterworth",
                "filter_order": "4",
                "temperature": "21.5",
                "humidity": "55",
            }
        )
        assert draft.microphone == "SMX-II"
        assert draft.recorded_by == "J. Doe"
        assert draft.filter_type == "Butterworth"
        assert draft.filter_order == 4
        assert draft.temperature == 21.5
        assert draft.humidity == 55.0


class TestMandatoryFields:
    def _fields(self, **overrides) -> dict[str, str]:
        base = {
            "date": "2020-01-01",
            "time": "09:00:00",
            "location": "-26,-48",
            "sample_window": "0,5",
        }
        base.update(overrides)
        return base

    def test_accepts_the_minimum(self):
        assert draft_from_fields(self._fields()).date == "2020-01-01"

    @pytest.mark.parametrize(
        "missing", ["date", "time", "location", "sample_window"]
    )
    def test_rejects_each_missing_mandatory_field(self, missing):
        fields = self._fields()
        del fields[missing]
        with pytest.raises(ValidationError, match=missing):
            draft_from_fields(fields)

    def test_reports_all_missing_fields_at_once(self):
        # Fixing a sidecar one error per run is tedious; report everything wrong.
        with pytest.raises(ValidationError) as exc:
            draft_from_fields({"date": "2020-01-01"})
        message = str(exc.value)
        assert "time" in message
        assert "location" in message
        assert "sample_window" in message

    def test_empty_mandatory_value_counts_as_missing(self):
        with pytest.raises(ValidationError, match="location"):
            draft_from_fields(self._fields(location=""))

    def test_unreadable_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="Could not read"):
            read_sidecar(tmp_path / "nope.txt")
