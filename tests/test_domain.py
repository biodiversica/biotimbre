"""Domain validation tests, including the real sidecar edge cases."""

from __future__ import annotations

import pytest

from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import (
    FilteredBand,
    ItemDraft,
    Location,
    SampleWindow,
    generate_biotimbre_id,
    is_species_name,
    normalize_taxa,
    parse_biotimbre_id,
)


class TestBiotimbreId:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (
                "3fae2ca1-1b2a-4b8a-9c3e-1a2b3c4d5e6f",
                "3fae2ca1-1b2a-4b8a-9c3e-1a2b3c4d5e6f",
            ),
            (
                "3FAE2CA1-1B2A-4B8A-9C3E-1A2B3C4D5E6F",
                "3fae2ca1-1b2a-4b8a-9c3e-1a2b3c4d5e6f",
            ),
            (
                "  3fae2ca1-1b2a-4b8a-9c3e-1a2b3c4d5e6f  ",
                "3fae2ca1-1b2a-4b8a-9c3e-1a2b3c4d5e6f",
            ),
        ],
    )
    def test_accepts_and_canonicalises(self, value, expected):
        assert parse_biotimbre_id(value) == expected

    @pytest.mark.parametrize(
        "value", ["", "   ", "not-a-uuid", "00A3", "0000", "12345"]
    )
    def test_rejects_malformed(self, value):
        with pytest.raises(ValidationError):
            parse_biotimbre_id(value)

    def test_rejects_non_string(self):
        with pytest.raises(ValidationError):
            parse_biotimbre_id(163)

    def test_generate_returns_distinct_valid_ids(self):
        first = generate_biotimbre_id()
        second = generate_biotimbre_id()
        assert first != second
        # Round-trips through parse_biotimbre_id unchanged, i.e. is already
        # in canonical form.
        assert parse_biotimbre_id(first) == first


class TestTaxa:
    def test_absent_means_unknown(self):
        assert normalize_taxa(None) == ()

    def test_parses_comma_separated_string(self):
        # This is how the sidecar files and a single text input both supply tags.
        assert normalize_taxa("insecta, orthoptera") == ("insecta", "orthoptera")

    def test_lowercases_dedupes_and_sorts(self):
        assert normalize_taxa(["Orthoptera", "insecta", "ORTHOPTERA"]) == (
            "insecta",
            "orthoptera",
        )

    def test_collapses_internal_whitespace(self):
        assert normalize_taxa(["  Tettigonia   viridissima "]) == (
            "tettigonia viridissima",
        )

    def test_drops_empty_tags(self):
        assert normalize_taxa("cicadidae,,  ,") == ("cicadidae",)

    def test_rejects_non_string_tag(self):
        with pytest.raises(ValidationError, match="strings"):
            normalize_taxa(["insecta", 42])


class TestLocation:
    def test_parses_sidecar_format(self):
        # From input_example/GUYVIR.txt: "location -26,-48"
        loc = Location.parse("-26,-48")
        assert (loc.latitude, loc.longitude) == (-26.0, -48.0)

    def test_parses_tuple(self):
        assert Location.parse((-20.5, -46.25)).latitude == -20.5

    @pytest.mark.parametrize(
        "value,message",
        [
            ("91,0", r"\[-90, 90\]"),
            ("0,181", r"\[-180, 180\]"),
            ("-26", "latitude,longitude"),
            ("1,2,3", "latitude,longitude"),
        ],
    )
    def test_rejects_bad_values(self, value, message):
        with pytest.raises(ValidationError, match=message):
            Location.parse(value)

    def test_roundtrips_through_string(self):
        assert str(Location.parse("-26,-48")) == "-26.0,-48.0"


class TestSpeciesNames:
    """Which tags name a species, and so share one biotimbre."""

    @pytest.mark.parametrize(
        "tag", ["guyalna bonaerensis", "fidicina mannifera", "dorisiana viridis"]
    )
    def test_a_binomial_is_a_species(self, tag):
        assert is_species_name(tag) is True

    @pytest.mark.parametrize("tag", ["cicadidae", "insecta", "orthoptera"])
    def test_a_rank_is_not(self, tag):
        assert is_species_name(tag) is False

    @pytest.mark.parametrize(
        "tag", ["guyalna sp", "guyalna spp", "guyalna cf", "guyalna aff"]
    )
    def test_open_nomenclature_is_not(self, tag):
        # "Guyalna sp." says the species is unknown; two samples carrying it
        # may be different species and must not be pulled into one biotimbre.
        assert is_species_name(tag) is False

    def test_capitalisation_does_not_decide_it(self):
        # Tags are stored lowercase, so a species typed either way is the same
        # tag by the time anything looks at it.
        typed_either_way = normalize_taxa(["Guyalna bonaerensis", "guyalna bonaerensis"])
        assert typed_either_way == ("guyalna bonaerensis",)
        assert is_species_name(typed_either_way[0]) is True


class TestFilteredBand:
    def test_parses_open_high_bound(self):
        # From input_example/GUYVIR.txt: "filtered_band 2000,-"
        band = FilteredBand.parse("2000,-")
        assert band.low == 2000.0
        assert band.high is None
        assert not band.is_unset

    def test_parses_closed_band(self):
        # From input_example/ZAMTYM.txt: "filtered_band 2000,10000"
        band = FilteredBand.parse("2000,10000")
        assert (band.low, band.high) == (2000.0, 10000.0)

    def test_parses_open_low_bound(self):
        band = FilteredBand.parse("-,10000")
        assert band.low is None
        assert band.high == 10000.0

    @pytest.mark.parametrize("value", [None, "", "-"])
    def test_absent_band_is_unset(self, value):
        assert FilteredBand.parse(value).is_unset

    def test_rejects_inverted_band(self):
        with pytest.raises(ValidationError, match="must be below"):
            FilteredBand.parse("10000,2000")

    def test_rejects_negative_frequency(self):
        with pytest.raises(ValidationError, match="negative"):
            FilteredBand.parse("-100,5000")

    def test_roundtrips_open_bound_as_dash(self):
        assert str(FilteredBand.parse("2000,-")) == "2000,-"

    def test_displays_a_closed_band_with_a_dash(self):
        assert FilteredBand.parse("2000,10000").display == "2000 - 10000"

    @pytest.mark.parametrize(
        ("value", "shown"), [("2000,-", "≥ 2000"), ("-,10000", "≤ 10000")]
    )
    def test_displays_an_open_bound_as_a_comparison(self, value, shown):
        # A lone cutoff would not say whether it is the highpass or the
        # lowpass; the comparison names the side without naming the filter.
        assert FilteredBand.parse(value).display == shown

    def test_displays_an_unset_band_as_nothing(self):
        assert FilteredBand.parse("-").display == ""

    def test_display_leaves_the_sidecar_form_alone(self):
        assert str(FilteredBand.parse("2000,10000")) == "2000,10000"


class TestSampleWindow:
    def test_parses_sidecar_format(self):
        # From input_example/GUYVIR.txt: "sample_window 34,39"
        window = SampleWindow.parse("34,39")
        assert (window.start, window.end) == (34.0, 39.0)
        assert window.duration == 5.0

    def test_rejects_end_before_start(self):
        with pytest.raises(ValidationError, match="greater than start"):
            SampleWindow.parse("39,34")

    def test_rejects_zero_length(self):
        with pytest.raises(ValidationError, match="greater than start"):
            SampleWindow.parse("10,10")

    def test_rejects_negative_start(self):
        with pytest.raises(ValidationError, match="negative"):
            SampleWindow.parse("-1,5")

    @pytest.mark.parametrize("value", [None, "", "-", (None, None), ("", "")])
    def test_absent_window_parses_as_none(self, value):
        assert SampleWindow.parse_optional(value) is None

    @pytest.mark.parametrize(
        ("value", "field"),
        [
            ((None, 5), "sample_window.start"),
            ((5, None), "sample_window.end"),
            (("", 5), "sample_window.start"),
        ],
    )
    def test_rejects_half_a_window(self, value, field):
        # Half a window places nothing; it is a typo, not a shorter window.
        with pytest.raises(ValidationError, match="both ends") as exc:
            SampleWindow.parse_optional(value)
        assert exc.value.field == field

    def test_parse_optional_still_validates_a_given_window(self):
        assert SampleWindow.parse_optional("34,39").duration == 5.0
        with pytest.raises(ValidationError, match="greater than start"):
            SampleWindow.parse_optional((39, 34))

    def test_displays_the_bounds_with_a_dash(self):
        assert SampleWindow.parse("5,10").display == "5 - 10"

    def test_display_leaves_the_sidecar_form_alone(self):
        assert str(SampleWindow.parse("5,10")) == "5,10"


class TestItemDraft:
    def _base(self, **overrides) -> ItemDraft:
        fields = {
            "date": "2020-01-12",
            "time": "09:40:21",
            "location": "-26,-48",
            "sample_window": "34,39",
        }
        fields.update(overrides)
        return ItemDraft(**fields)

    def test_window_is_optional(self):
        draft = ItemDraft(
            date="2020-01-12", time="09:40:21", location="-26,-48"
        ).validated()
        assert draft.sample_window is None

    def test_validates_guyvir_example(self):
        """input_example/GUYVIR.txt — no taxa line, open-ended filtered band."""
        draft = self._base(
            device="Audiomoth", filtered_band="2000,-"
        ).validated()
        assert draft.taxa == ()
        assert draft.taxa_display == "unknown"
        assert draft.filtered_band.high is None
        assert draft.speed == 1.0
        assert draft.biotimbre_id is None

    def test_validates_zamtym_example(self):
        """input_example/ZAMTYM.txt — has taxa, fully closed filtered band."""
        draft = self._base(
            date="2020-01-14",
            location="-20,-46",
            sample_window="5,10",
            taxa="cicadidae",
            device="Audiomoth",
            filtered_band="2000,10000",
        ).validated()
        assert draft.taxa == ("cicadidae",)
        assert draft.filtered_band == FilteredBand(2000.0, 10000.0)

    @pytest.mark.parametrize(
        "date", ["14-01-2020", "2020/01/14", "not-a-date", "2020-02-30", "2021-02-29"]
    )
    def test_rejects_bad_dates(self, date):
        with pytest.raises(ValidationError, match="date"):
            self._base(date=date).validated()

    @pytest.mark.parametrize("time", ["9:40", "25:00:00", "09:40", "09:40:61"])
    def test_rejects_bad_times(self, time):
        with pytest.raises(ValidationError, match="time"):
            self._base(time=time).validated()

    @pytest.mark.parametrize(
        "supplied,canonical",
        [("2020-1-14", "2020-01-14"), ("2020-01-4", "2020-01-04")],
    )
    def test_zero_pads_dates(self, supplied, canonical):
        # strptime accepts unpadded input, but date range filters compare these
        # as strings in SQL, so storage must be zero-padded to keep
        # lexicographic order equal to chronological order.
        assert self._base(date=supplied).validated().date == canonical

    def test_zero_pads_times(self):
        assert self._base(time="9:40:21").validated().time == "09:40:21"

    def test_blank_biotimbre_id_requests_allocation(self):
        # A form submits "" for an untouched field; that must mean "allocate",
        # not "the ID is empty string".
        assert self._base(biotimbre_id="   ").validated().biotimbre_id is None

    def test_normalises_supplied_biotimbre_id(self):
        uid = "3FAE2CA1-1B2A-4B8A-9C3E-1A2B3C4D5E6F"
        assert (
            self._base(biotimbre_id=uid).validated().biotimbre_id == uid.lower()
        )

    def test_blank_optional_text_becomes_none(self):
        draft = self._base(device="  ", notes="", link_to_full_recording=" ").validated()
        assert draft.device is None
        assert draft.notes is None
        assert draft.link_to_full_recording is None

    @pytest.mark.parametrize("speed", [0.25, 0.5, 1.0, 2.0, 10.0])
    def test_accepts_reasonable_speeds(self, speed):
        assert self._base(speed=speed).validated().speed == speed

    @pytest.mark.parametrize("speed", [0.0, -1.0, 0.001, 1000.0])
    def test_rejects_out_of_range_speeds(self, speed):
        with pytest.raises(ValidationError, match="speed"):
            self._base(speed=speed).validated()

    def test_validated_is_idempotent(self):
        once = self._base(taxa="Insecta, orthoptera").validated()
        assert once.validated() == once

    def test_coerces_string_numerics(self):
        # A web form delivers everything as strings.
        draft = self._base(
            speed="0.25", filter_order="4", temperature="21.5", humidity="55"
        ).validated()
        assert draft.speed == 0.25
        assert draft.filter_order == 4
        assert draft.temperature == 21.5
        assert draft.humidity == 55.0

    def test_blank_optional_numerics_become_none(self):
        draft = self._base(filter_order="", temperature="", humidity="").validated()
        assert draft.filter_order is None
        assert draft.temperature is None
        assert draft.humidity is None

    def test_rejects_non_integer_filter_order(self):
        with pytest.raises(ValidationError, match="filter_order"):
            self._base(filter_order=4.5).validated()

    def test_rejects_non_numeric_temperature(self):
        with pytest.raises(ValidationError, match="temperature"):
            self._base(temperature="warm").validated()

    def test_blank_optional_text_becomes_none_for_new_fields(self):
        draft = self._base(microphone="  ", recorded_by="", filter_type=" ").validated()
        assert draft.microphone is None
        assert draft.recorded_by is None
        assert draft.filter_type is None
