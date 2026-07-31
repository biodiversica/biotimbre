"""Query embedding cache and bounded inference batching."""

from __future__ import annotations

import numpy as np
import pytest

from biotimbre.embeddings.extractor import WindowEmbeddings
from biotimbre.embeddings.onnx_backend import DEFAULT_MAX_BATCH
from biotimbre.embeddings.query_cache import AudioKey, QueryEmbeddingCache
from conftest import EXAMPLES_DIR

GUYVIR_WAV = EXAMPLES_DIR / "GUYVIR.wav"


class FakeExtractor:
    """Counts how often it is asked to embed, standing in for a real model."""

    def __init__(self, spec):
        self.spec = spec
        self.calls: list[float] = []

    def embed_array(self, samples, native_rate, *, speed=1.0):
        self.calls.append(float(speed))
        return WindowEmbeddings(
            model_id=self.spec.model_id,
            vectors=np.zeros((1, 4), dtype=np.float32),
            spans=[(0.0, 5.0)],
            speed=float(speed),
        )


class FakePool:
    """Minimal ExtractorPool stand-in over :class:`FakeExtractor`."""

    def __init__(self, specs):
        self._extractors = {spec.model_id: FakeExtractor(spec) for spec in specs}
        self.model_ids = tuple(self._extractors)

    def get(self, model_id):
        return self._extractors[model_id]

    def calls(self, model_id):
        return self._extractors[model_id].calls


@pytest.fixture
def pool(fake_spec):
    import dataclasses

    second = dataclasses.replace(fake_spec, model_id="other_model")
    return FakePool([fake_spec, second])


@pytest.fixture
def cache(pool):
    return QueryEmbeddingCache(pool)


def audio():
    return np.zeros(1000, dtype=np.float32), 32_000


class TestAudioKey:
    def test_path_key_is_stable(self):
        assert AudioKey.for_path(GUYVIR_WAV) == AudioKey.for_path(GUYVIR_WAV)

    def test_path_key_changes_when_the_file_changes(self, tmp_path):
        import shutil
        import os

        copy = tmp_path / "q.wav"
        shutil.copy2(GUYVIR_WAV, copy)
        before = AudioKey.for_path(copy)
        with open(copy, "ab") as handle:
            handle.write(b"\0" * 64)
        os.utime(copy, ns=(0, 12345))
        assert AudioKey.for_path(copy) != before

    def test_different_files_differ(self, tmp_path):
        import shutil

        a, b = tmp_path / "a.wav", tmp_path / "b.wav"
        shutil.copy2(GUYVIR_WAV, a)
        shutil.copy2(GUYVIR_WAV, b)
        # Same content, different paths: distinct keys, which is correct — the
        # key identifies a query, not a checksum.
        assert AudioKey.for_path(a) != AudioKey.for_path(b)

    def test_array_key_depends_on_content(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(5000).astype(np.float32)
        b = rng.standard_normal(5000).astype(np.float32)
        assert AudioKey.for_array(a, 32_000) == AudioKey.for_array(a, 32_000)
        assert AudioKey.for_array(a, 32_000) != AudioKey.for_array(b, 32_000)

    def test_array_key_depends_on_rate(self):
        a = np.zeros(5000, dtype=np.float32)
        assert AudioKey.for_array(a, 32_000) != AudioKey.for_array(a, 48_000)

    def test_array_key_depends_on_length(self):
        assert AudioKey.for_array(
            np.zeros(5000, np.float32), 32_000
        ) != AudioKey.for_array(np.zeros(6000, np.float32), 32_000)

    def test_segment_is_part_of_the_path_key(self):
        # Two 15 s windows of one recording are different queries; serving one
        # for the other would be silently wrong.
        whole = AudioKey.for_path(GUYVIR_WAV)
        first = AudioKey.for_path(GUYVIR_WAV, segment=(0.0, 2.0))
        second = AudioKey.for_path(GUYVIR_WAV, segment=(2.0, 4.0))
        assert len({whole, first, second}) == 3

    def test_same_segment_gives_the_same_key(self):
        assert AudioKey.for_path(
            GUYVIR_WAV, segment=(1.0, 3.0)
        ) == AudioKey.for_path(GUYVIR_WAV, segment=(1.0, 3.0))

    def test_filter_key_is_part_of_the_path_key(self):
        # A filtered and an unfiltered query are different audio by the time
        # they reach the model, so they must not share a cache entry.
        unfiltered = AudioKey.for_path(GUYVIR_WAV)
        filtered = AudioKey.for_path(GUYVIR_WAV, filter_key="1000,None,4")
        other_filter = AudioKey.for_path(GUYVIR_WAV, filter_key="2000,None,4")
        assert len({unfiltered, filtered, other_filter}) == 3

    def test_filter_key_is_part_of_the_array_key(self):
        a = np.zeros(5000, dtype=np.float32)
        assert AudioKey.for_array(a, 32_000) != AudioKey.for_array(
            a, 32_000, filter_key="1000,None,4"
        )


class TestCaching:
    def test_first_call_computes_every_model_and_speed(self, cache, pool):
        samples, rate = audio()
        result = cache.embed(
            AudioKey(kind="test", token="a"), [1.0, 0.5],
            samples=samples, native_rate=rate,
        )
        assert set(result) == {
            ("fake_model", 1.0), ("fake_model", 0.5),
            ("other_model", 1.0), ("other_model", 0.5),
        }
        assert cache.misses == 4
        assert cache.hits == 0

    def test_repeat_call_computes_nothing(self, cache, pool):
        samples, rate = audio()
        key = AudioKey(kind="test", token="a")
        cache.embed(key, [1.0], samples=samples, native_rate=rate)
        before = cache.misses
        cache.embed(key, [1.0], samples=samples, native_rate=rate)
        # This is the case the cache exists for: same query, different filter.
        assert cache.misses == before
        assert cache.hits == 2

    def test_repeat_call_does_not_need_the_audio(self, cache):
        samples, rate = audio()
        key = AudioKey(kind="test", token="a")
        cache.embed(key, [1.0], samples=samples, native_rate=rate)

        def explode():
            raise AssertionError("loader must not run on a full cache hit")

        # A cached query never touches the file, so decoding is skipped too.
        cache.embed(key, [1.0], loader=explode)

    def test_new_speed_computes_only_that_entry(self, cache, pool):
        samples, rate = audio()
        key = AudioKey(kind="test", token="a")
        cache.embed(key, [1.0], samples=samples, native_rate=rate)
        assert pool.calls("fake_model") == [1.0]

        # A filter newly admitting speed-0.5 items must not discard the 1.0 work.
        cache.embed(key, [1.0, 0.5], samples=samples, native_rate=rate)
        assert pool.calls("fake_model") == [1.0, 0.5]

    def test_different_audio_is_a_separate_entry(self, cache, pool):
        samples, rate = audio()
        cache.embed(AudioKey(kind="test", token="a"), [1.0],
                    samples=samples, native_rate=rate)
        cache.embed(AudioKey(kind="test", token="b"), [1.0],
                    samples=samples, native_rate=rate)
        assert pool.calls("fake_model") == [1.0, 1.0]

    def test_changed_preprocessing_invalidates(self, pool, fake_spec):
        import dataclasses

        cache = QueryEmbeddingCache(pool)
        samples, rate = audio()
        key = AudioKey(kind="test", token="a")
        cache.embed(key, [1.0], samples=samples, native_rate=rate)

        # Re-registering a model with a different overlap changes what its
        # vectors mean, so cached ones must not be served.
        pool._extractors["fake_model"].spec = dataclasses.replace(
            fake_spec, window_overlap=0.0
        )
        before = cache.misses
        cache.embed(key, [1.0], samples=samples, native_rate=rate)
        assert cache.misses == before + 1

    def test_lru_evicts_oldest(self, pool):
        cache = QueryEmbeddingCache(pool, max_entries=2)
        samples, rate = audio()
        for token in ("a", "b", "c"):
            cache.embed(AudioKey(kind="test", token=token), [1.0],
                        samples=samples, native_rate=rate)
        # "a" was evicted, so it must be recomputed.
        before = cache.misses
        cache.embed(AudioKey(kind="test", token="a"), [1.0],
                    samples=samples, native_rate=rate)
        assert cache.misses > before

    def test_miss_without_audio_or_loader_raises(self, cache):
        with pytest.raises(ValueError, match="not cached"):
            cache.embed(AudioKey(kind="test", token="a"), [1.0])

    def test_clear_resets(self, cache):
        samples, rate = audio()
        cache.embed(AudioKey(kind="test", token="a"), [1.0],
                    samples=samples, native_rate=rate)
        cache.clear()
        assert cache.hits == 0
        assert cache.misses == 0


@pytest.mark.models
class TestBoundedInferenceBatch:
    """Chunked inference must be a pure memory bound, not a behaviour change.

    The query length cap keeps searches small, but a speed shift still multiplies
    window count (15 s at speed 0.1 is 150 s of shifted audio), so the cap stays
    as a safety net. Verified on a short input with a small batch rather than a
    long one, which is equivalent and fast.
    """

    def test_chunking_matches_a_single_batch(self):
        from biotimbre.embeddings.extractor import Extractor
        from biotimbre.models_registry import get_spec

        spec = get_spec("birdnet_v2.4")
        rng = np.random.default_rng(0)
        samples = rng.standard_normal(5 * 32_000).astype(np.float32) * 0.1

        one_at_a_time = Extractor(spec, max_batch=1)
        all_at_once = Extractor(spec, max_batch=64)
        chunked = one_at_a_time.embed_array(samples, 32_000)
        whole = all_at_once.embed_array(samples, 32_000)

        assert chunked.n_windows == 3
        np.testing.assert_allclose(chunked.vectors, whole.vectors, atol=1e-4)
        one_at_a_time.close()
        all_at_once.close()

    def test_default_is_per_window(self):
        # Measured: batching makes Perch 2.1x slower and helps BirdNET only
        # within noise, so per-window is both the fastest option and the tightest
        # memory bound (~19 MB per BirdNET window).
        assert DEFAULT_MAX_BATCH == 1


@pytest.mark.models
class TestServiceUsesTheCache:
    def test_refiltering_does_not_recompute(self, settings, conn, shared_pool):
        from biotimbre.search.filters import SearchFilters
        from biotimbre.service import LibraryService

        service = LibraryService(settings, connection=conn, pool=shared_pool)
        try:
            service.import_directory(EXAMPLES_DIR)
            service.search(GUYVIR_WAV, k=2)
            after_first = service.query_cache.misses
            assert after_first > 0

            for filters in (
                SearchFilters(taxa=["cicadidae"]),
                SearchFilters(date_from="2020-01-01"),
                SearchFilters(latitude=-26.0, longitude=-48.0, radius_km=500.0),
            ):
                service.search(GUYVIR_WAV, filters=filters, k=2)

            # Only filters changed, so the query embeddings are reused.
            assert service.query_cache.misses == after_first
            assert service.query_cache.hits > 0
        finally:
            service.close()

    def test_results_are_identical_cached_or_not(self, settings, conn, shared_pool):
        from biotimbre.service import LibraryService

        service = LibraryService(settings, connection=conn, pool=shared_pool)
        try:
            service.import_directory(EXAMPLES_DIR)
            first = service.search(GUYVIR_WAV, k=2)
            service.query_cache.clear()
            second = service.search(GUYVIR_WAV, k=2)

            assert [h.item.item_id for h in first] == [h.item.item_id for h in second]
            assert [round(h.score, 6) for h in first] == [
                round(h.score, 6) for h in second
            ]
        finally:
            service.close()
