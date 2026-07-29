"""Extraction against the real ONNX backbones.

Marked ``models``: these need the model files, downloaded on first run and cached
afterwards. Run ``pytest -m "not models"`` to skip them offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from biotimbre.domain.errors import EmbeddingError
from biotimbre.embeddings.extractor import Extractor, ExtractorPool
from biotimbre.embeddings.onnx_backend import OnnxEmbedder
from biotimbre.models_registry import get_spec
from conftest import EXAMPLES_DIR

GUYVIR = EXAMPLES_DIR / "GUYVIR.wav"
ZAMTYM = EXAMPLES_DIR / "ZAMTYM.wav"

#: Verified from the ONNX graphs, not assumed. Perch declares
#: ('batch', 1536) and BirdNET (None, 1024).
EXPECTED_DIMS = {"perch_v2": 1536, "birdnet_v2.4": 1024}


class TestOutputFlattening:
    """``_flatten_output`` shape handling — pure logic, needs no model."""

    def _embedder(self) -> OnnxEmbedder:
        return OnnxEmbedder("/nonexistent.onnx", model_id="test")

    def test_1d_becomes_single_row(self):
        out = self._embedder()._flatten_output(np.arange(4, dtype=np.float32))
        assert out.shape == (1, 4)

    def test_2d_passes_through(self):
        out = self._embedder()._flatten_output(np.zeros((3, 8), np.float32))
        assert out.shape == (3, 8)

    def test_squeezes_unit_axes(self):
        out = self._embedder()._flatten_output(np.zeros((2, 1, 1, 16), np.float32))
        assert out.shape == (2, 16)

    def test_mean_pools_a_real_time_axis(self):
        raw = np.zeros((2, 4, 8), dtype=np.float32)
        raw[:, 0, :] = 4.0
        out = self._embedder()._flatten_output(raw)
        assert out.shape == (2, 8)
        np.testing.assert_allclose(out, 1.0)

    def test_rejects_scalar_output(self):
        with pytest.raises(EmbeddingError, match="scalar"):
            self._embedder()._flatten_output(np.float32(1.0))


@pytest.mark.models
class TestGraphContracts:
    @pytest.mark.parametrize("model_id", sorted(EXPECTED_DIMS))
    def test_embedding_dim_read_from_graph(self, model_id):
        extractor = Extractor(get_spec(model_id))
        assert extractor.embedding_dim() == EXPECTED_DIMS[model_id]
        extractor.close()

    @pytest.mark.parametrize("model_id", sorted(EXPECTED_DIMS))
    def test_batch_dimension_is_dynamic(self, model_id):
        # Both graphs accept many windows per call, so inference is batched
        # rather than looped. Phase 3 was written not to assume this.
        extractor = Extractor(get_spec(model_id))
        assert extractor._embedder.has_dynamic_batch is True
        extractor.close()

    @pytest.mark.parametrize("model_id", sorted(EXPECTED_DIMS))
    def test_takes_raw_audio_not_spectrograms(self, model_id):
        # The input width equals window_duration x sample_rate, which is only
        # true for raw audio input.
        spec = get_spec(model_id)
        extractor = Extractor(spec)
        embedder = extractor._embedder
        embedder._ensure_session()
        declared = embedder._session.get_inputs()[0].shape
        assert declared[-1] == spec.window_samples
        extractor.close()


@pytest.mark.models
class TestExtraction:
    @pytest.mark.parametrize(
        "model_id,expected_windows",
        [("perch_v2", 1), ("birdnet_v2.4", 3)],
    )
    def test_window_counts_on_the_examples(self, model_id, expected_windows):
        # A 5 s sample at 50% overlap: Perch's 5 s window leaves no room for a
        # second hop, BirdNET's 3 s window fits three.
        extractor = Extractor(get_spec(model_id))
        result = extractor.embed_file(GUYVIR)
        assert result.n_windows == expected_windows
        assert result.embedding_dim == EXPECTED_DIMS[model_id]
        extractor.close()

    def test_vectors_are_finite_and_nonzero(self):
        extractor = Extractor(get_spec("perch_v2"))
        result = extractor.embed_file(GUYVIR)
        assert np.isfinite(result.vectors).all()
        assert np.linalg.norm(result.vectors, axis=1).min() > 0
        extractor.close()

    def test_extraction_is_deterministic(self):
        extractor = Extractor(get_spec("perch_v2"))
        first = extractor.embed_file(GUYVIR)
        second = extractor.embed_file(GUYVIR)
        np.testing.assert_array_equal(first.vectors, second.vectors)
        extractor.close()

    def test_different_audio_gives_different_vectors(self):
        extractor = Extractor(get_spec("perch_v2"))
        guyvir = extractor.embed_file(GUYVIR).vectors[0]
        zamtym = extractor.embed_file(ZAMTYM).vectors[0]
        cosine = float(
            guyvir @ zamtym
            / (np.linalg.norm(guyvir) * np.linalg.norm(zamtym))
        )
        assert cosine < 0.999
        extractor.close()

    def test_spans_are_in_the_sample_time_base(self):
        extractor = Extractor(get_spec("birdnet_v2.4"))
        result = extractor.embed_file(GUYVIR)
        assert result.spans[0] == (0.0, 3.0)
        assert result.spans[1] == (1.5, 4.5)
        extractor.close()

    def test_as_rows_matches_repository_shape(self):
        extractor = Extractor(get_spec("birdnet_v2.4"))
        rows = extractor.embed_file(GUYVIR).as_rows()
        assert len(rows) == 3
        start, end, vector = rows[0]
        assert (start, end) == (0.0, 3.0)
        assert vector.shape == (EXPECTED_DIMS["birdnet_v2.4"],)
        extractor.close()


@pytest.mark.models
class TestSpeedShift:
    def test_slower_speed_yields_more_windows(self):
        extractor = Extractor(get_spec("perch_v2"))
        assert extractor.embed_file(GUYVIR, speed=1.0).n_windows == 1
        assert extractor.embed_file(GUYVIR, speed=0.5).n_windows == 3
        assert extractor.embed_file(GUYVIR, speed=0.25).n_windows == 7
        extractor.close()

    def test_spans_stay_inside_the_real_sample(self):
        extractor = Extractor(get_spec("perch_v2"))
        result = extractor.embed_file(GUYVIR, speed=0.25)
        # The shifted timeline is 20 s long, but the stored spans describe the
        # real 5 s file so the player can mark the right region.
        assert result.spans[0][0] == 0.0
        assert result.spans[-1][1] == pytest.approx(5.0)
        extractor.close()

    def test_speed_changes_the_embedding(self):
        # This is why the query-side sweep exists (PLAN.md D2c): the same audio
        # at a different speed lands somewhere else in embedding space, so a
        # native-speed query would not match a slowed-down library item.
        extractor = Extractor(get_spec("perch_v2"))
        native = extractor.embed_file(GUYVIR, speed=1.0).vectors[0]
        slowed = extractor.embed_file(GUYVIR, speed=0.25).vectors[0]
        cosine = float(
            native @ slowed / (np.linalg.norm(native) * np.linalg.norm(slowed))
        )
        assert cosine < 0.95
        extractor.close()

    def test_speed_shift_is_reproducible(self):
        extractor = Extractor(get_spec("perch_v2"))
        first = extractor.embed_file(GUYVIR, speed=0.25)
        second = extractor.embed_file(GUYVIR, speed=0.25)
        np.testing.assert_array_equal(first.vectors, second.vectors)
        extractor.close()


@pytest.mark.models
class TestExtractorPool:
    def test_embeds_with_every_model(self):
        pool = ExtractorPool(("perch_v2", "birdnet_v2.4"))
        results = pool.embed_file(GUYVIR)
        assert set(results) == {"perch_v2", "birdnet_v2.4"}
        assert results["perch_v2"].n_windows == 1
        assert results["birdnet_v2.4"].n_windows == 3
        pool.close()

    def test_reuses_sessions_across_calls(self):
        pool = ExtractorPool(("perch_v2",))
        first = pool.get("perch_v2")
        second = pool.get("perch_v2")
        # Reloading a 45 MB graph per item would dominate ingest time.
        assert first is second
        pool.close()

    def test_multi_speed_sweep_covers_the_cross_product(self):
        pool = ExtractorPool(("perch_v2", "birdnet_v2.4"))
        samples, rate = __import__(
            "biotimbre.embeddings.audio", fromlist=["load_mono"]
        ).load_mono(GUYVIR)
        results = pool.embed_array_multi_speed(samples, rate, [1.0, 0.5])
        assert set(results) == {
            ("perch_v2", 1.0),
            ("perch_v2", 0.5),
            ("birdnet_v2.4", 1.0),
            ("birdnet_v2.4", 0.5),
        }
        pool.close()

    def test_rejects_unknown_model(self):
        pool = ExtractorPool(("perch_v2",))
        with pytest.raises(KeyError, match="not in this pool"):
            pool.get("birdnet_v2.4")
