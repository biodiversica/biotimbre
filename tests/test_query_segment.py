"""The query length cap and segment selection.

Searching is for short vocalization samples. A query longer than the cap must be
narrowed explicitly rather than silently truncated.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from biotimbre.domain.errors import ValidationError
from biotimbre.embeddings.audio import (
    MAX_QUERY_DURATION_S,
    load_mono,
    resolve_query_segment,
)
from conftest import EXAMPLES_DIR

GUYVIR_WAV = EXAMPLES_DIR / "GUYVIR.wav"


@pytest.fixture
def long_wav(tmp_path):
    """A 40 s file, comfortably over the 15 s cap."""
    samples, rate = sf.read(GUYVIR_WAV, dtype="float32")
    path = tmp_path / "long.wav"
    sf.write(path, np.tile(samples, 8), rate)
    return path


class TestResolveQuerySegment:
    def test_short_file_is_used_whole(self):
        assert resolve_query_segment(5.0) == (0.0, 5.0)

    def test_file_exactly_at_the_cap_is_used_whole(self):
        assert resolve_query_segment(15.0) == (0.0, 15.0)

    def test_long_file_without_a_segment_is_refused(self):
        # Silently taking the first 15 s would search something other than what
        # the user pointed at.
        with pytest.raises(ValidationError, match="limited to 15 s"):
            resolve_query_segment(60.0)

    def test_refusal_names_the_duration_and_the_remedy(self):
        with pytest.raises(ValidationError) as exc:
            resolve_query_segment(62.5)
        message = str(exc.value)
        assert "62.5 s" in message
        # Worded for both CLI and API callers, so it names neither's exact flags.
        assert "start and end time" in message
        assert "15 s" in message

    def test_explicit_segment_within_the_cap(self):
        assert resolve_query_segment(60.0, 10.0, 20.0) == (10.0, 20.0)

    def test_start_only_takes_the_cap_forward(self):
        assert resolve_query_segment(60.0, 40.0, None) == (40.0, 55.0)

    def test_start_only_stops_at_the_end_of_the_file(self):
        assert resolve_query_segment(20.0, 12.0, None) == (12.0, 20.0)

    def test_end_beyond_the_file_is_clamped(self):
        # Asking for 10-25 s of a 20 s file is a near-miss, not an error.
        assert resolve_query_segment(20.0, 10.0, 25.0) == (10.0, 20.0)

    def test_segment_over_the_cap_is_refused(self):
        with pytest.raises(ValidationError, match="30.0 s"):
            resolve_query_segment(60.0, 0.0, 30.0)

    def test_negative_start_is_refused(self):
        with pytest.raises(ValidationError, match="negative"):
            resolve_query_segment(60.0, -1.0, 10.0)

    def test_start_past_the_end_is_refused(self):
        with pytest.raises(ValidationError, match="past the end"):
            resolve_query_segment(20.0, 25.0, 30.0)

    def test_inverted_segment_is_refused(self):
        with pytest.raises(ValidationError, match="greater than"):
            resolve_query_segment(60.0, 20.0, 10.0)

    def test_zero_length_segment_is_refused(self):
        with pytest.raises(ValidationError, match="greater than"):
            resolve_query_segment(60.0, 10.0, 10.0)

    def test_cap_is_configurable(self):
        assert resolve_query_segment(25.0, max_duration_s=30.0) == (0.0, 25.0)
        with pytest.raises(ValidationError):
            resolve_query_segment(25.0, max_duration_s=20.0)

    def test_default_cap_value(self):
        assert MAX_QUERY_DURATION_S == 15.0


class TestSegmentDecoding:
    def test_loads_only_the_requested_frames(self, long_wav):
        samples, rate = load_mono(long_wav, start_time=10.0, end_time=15.0)
        # Decoding only the segment is what keeps a long file off the heap.
        assert rate == 32_000
        assert samples.size == pytest.approx(5.0 * rate, abs=2)

    def test_segment_matches_the_same_slice_of_the_whole_file(self, long_wav):
        whole, rate = load_mono(long_wav)
        segment, _ = load_mono(long_wav, start_time=7.0, end_time=11.0)
        expected = whole[int(7.0 * rate) : int(11.0 * rate)]
        np.testing.assert_allclose(segment, expected, atol=1e-6)

    def test_start_only_runs_to_the_end(self, long_wav):
        samples, rate = load_mono(long_wav, start_time=38.0)
        assert samples.size == pytest.approx(2.0 * rate, abs=2)

    def test_empty_segment_is_refused(self, long_wav):
        with pytest.raises(ValidationError, match="no audio"):
            load_mono(long_wav, start_time=10.0, end_time=10.0)


@pytest.mark.models
class TestSearchEnforcesTheCap:
    @pytest.fixture
    def service(self, settings, conn):
        from biotimbre.service import LibraryService

        svc = LibraryService(settings, connection=conn)
        svc.import_directory(EXAMPLES_DIR)
        yield svc
        svc.close()

    def test_short_query_works_unchanged(self, service):
        assert service.search(GUYVIR_WAV, k=1)

    def test_long_query_is_refused_before_any_decoding(self, service, long_wav):
        with pytest.raises(ValidationError, match="limited to 15 s"):
            service.search(long_wav, k=1)

    def test_long_query_works_with_a_segment(self, service, long_wav):
        hits = service.search(long_wav, k=1, start_time=0.0, end_time=5.0)
        assert hits
        # The first 5 s of the tiled file is GUYVIR itself.
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)

    def test_over_long_segment_is_refused(self, service, long_wav):
        with pytest.raises(ValidationError, match="limited to 15 s"):
            service.search(long_wav, k=1, start_time=0.0, end_time=20.0)

    def test_different_segments_are_cached_separately(self, service, long_wav):
        service.search(long_wav, k=1, start_time=0.0, end_time=5.0)
        first = service.query_cache.misses
        service.search(long_wav, k=1, start_time=5.0, end_time=10.0)
        assert service.query_cache.misses > first

    def test_same_segment_reuses_the_cache(self, service, long_wav):
        from biotimbre.search.filters import SearchFilters

        service.search(long_wav, k=1, start_time=0.0, end_time=5.0)
        first = service.query_cache.misses
        service.search(
            long_wav, k=1, start_time=0.0, end_time=5.0,
            filters=SearchFilters(taxa=["cicadidae"]),
        )
        assert service.query_cache.misses == first
