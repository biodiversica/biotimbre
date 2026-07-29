"""Model registry: spec arithmetic and cache-first file resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from biotimbre.models_registry import (
    DEFAULT_WINDOW_OVERLAP,
    MODEL_SPECS,
    get_spec,
    resolve_model_file,
)


class TestSpecs:
    def test_both_backbones_are_registered(self):
        assert set(MODEL_SPECS) == {"perch_v2", "birdnet_v2.4"}

    def test_unknown_model_lists_the_valid_ids(self):
        with pytest.raises(KeyError, match="perch_v2"):
            get_spec("nope")

    @pytest.mark.parametrize(
        "model_id,rate,window",
        [("perch_v2", 32_000, 5.0), ("birdnet_v2.4", 48_000, 3.0)],
    )
    def test_preprocessing_matches_the_graphs(self, model_id, rate, window):
        spec = get_spec(model_id)
        assert spec.sample_rate == rate
        assert spec.window_duration_s == window
        assert spec.window_samples == int(rate * window)

    def test_default_overlap_is_applied(self):
        assert get_spec("perch_v2").window_overlap == DEFAULT_WINDOW_OVERLAP

    def test_stride_follows_overlap(self):
        spec = get_spec("perch_v2")
        assert spec.stride_samples == spec.window_samples // 2

    @pytest.mark.parametrize(
        "model_id,nyquist", [("perch_v2", 16_000.0), ("birdnet_v2.4", 24_000.0)]
    )
    def test_nyquist(self, model_id, nyquist):
        # What the speed shift exists to work around: content above this is gone
        # after resampling.
        assert get_spec(model_id).nyquist_hz == nyquist

    def test_source_is_recorded_for_provenance(self):
        assert get_spec("perch_v2").source == "hf://biodiversica/Perch-onnx-backbone"


class TestResolutionIsCacheFirst:
    """Resolution must not touch the network when the file is already cached.

    Every search resolves its models, and revalidating against the Hub cost
    0.18-0.49 s per model per call. It also risked picking up a new build of the
    same ``model_id``, whose embeddings would not be comparable with the stored
    ones — and printed huggingface_hub's unauthenticated-request warning each time.
    """

    def _patch(self, monkeypatch, *, cached: bool):
        """Replace hf_hub_download, recording how it was called."""
        import huggingface_hub

        calls: list[dict] = []

        def fake(**kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only") and not cached:
                raise OSError("not in cache")
            return "/fake/model.onnx"

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
        return calls

    def test_cached_file_never_reaches_the_network(self, monkeypatch):
        calls = self._patch(monkeypatch, cached=True)
        path = resolve_model_file(get_spec("perch_v2"))
        assert path == Path("/fake/model.onnx")
        # Exactly one call, and it was the offline one.
        assert len(calls) == 1
        assert calls[0]["local_files_only"] is True

    def test_cache_miss_falls_through_to_a_download(self, monkeypatch):
        calls = self._patch(monkeypatch, cached=False)
        assert resolve_model_file(get_spec("perch_v2")) == Path("/fake/model.onnx")
        # Offline attempt first, then the real download.
        assert len(calls) == 2
        assert calls[0]["local_files_only"] is True
        assert "local_files_only" not in calls[1]

    def test_download_failure_reports_the_model(self, monkeypatch):
        import huggingface_hub

        def always_fail(**kwargs):
            raise OSError("no route to host")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", always_fail)
        with pytest.raises(RuntimeError, match="perch_v2"):
            resolve_model_file(get_spec("perch_v2"))

    def test_failure_message_carries_the_cause(self, monkeypatch):
        import huggingface_hub

        def always_fail(**kwargs):
            raise OSError("no route to host")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", always_fail)
        with pytest.raises(RuntimeError, match="no route to host"):
            resolve_model_file(get_spec("perch_v2"))

    def test_cache_dir_is_forwarded(self, monkeypatch, tmp_path):
        calls = self._patch(monkeypatch, cached=True)
        resolve_model_file(get_spec("perch_v2"), tmp_path)
        assert calls[0]["cache_dir"] == str(tmp_path)

    def test_revision_is_forwarded(self, monkeypatch):
        import dataclasses

        calls = self._patch(monkeypatch, cached=True)
        pinned = dataclasses.replace(get_spec("perch_v2"), revision="abc123")
        resolve_model_file(pinned)
        assert calls[0]["revision"] == "abc123"


@pytest.mark.models
class TestRealResolution:
    def test_resolves_from_cache_quickly(self):
        import time

        spec = get_spec("perch_v2")
        resolve_model_file(spec)  # ensure cached
        t = time.perf_counter()
        path = resolve_model_file(spec)
        elapsed = time.perf_counter() - t

        assert path.is_file()
        # A network round-trip would be tens of milliseconds at best; this should
        # be a local cache lookup.
        assert elapsed < 0.05, f"resolution took {elapsed * 1000:.0f} ms"
