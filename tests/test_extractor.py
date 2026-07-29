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
    def test_embedding_dim_read_from_graph(self, shared_pool, model_id):
        assert shared_pool.get(model_id).embedding_dim() == EXPECTED_DIMS[model_id]

    @pytest.mark.parametrize("model_id", sorted(EXPECTED_DIMS))
    def test_batch_dimension_is_dynamic(self, shared_pool, model_id):
        # Both graphs accept many windows per call, so inference is batched
        # rather than looped. Phase 3 was written not to assume this.
        assert shared_pool.get(model_id)._embedder.has_dynamic_batch is True

    @pytest.mark.parametrize("model_id", sorted(EXPECTED_DIMS))
    def test_takes_raw_audio_not_spectrograms(self, shared_pool, model_id):
        # The input width equals window_duration x sample_rate, which is only
        # true for raw audio input.
        extractor = shared_pool.get(model_id)
        embedder = extractor._embedder
        embedder._ensure_session()
        declared = embedder._session.get_inputs()[0].shape
        assert declared[-1] == extractor.spec.window_samples


@pytest.mark.models
class TestExtraction:
    @pytest.mark.parametrize(
        "model_id,expected_windows",
        [("perch_v2", 1), ("birdnet_v2.4", 3)],
    )
    def test_window_counts_on_the_examples(
        self, shared_pool, model_id, expected_windows
    ):
        # A 5 s sample at 50% overlap: Perch's 5 s window leaves no room for a
        # second hop, BirdNET's 3 s window fits three.
        result = shared_pool.get(model_id).embed_file(GUYVIR)
        assert result.n_windows == expected_windows
        assert result.embedding_dim == EXPECTED_DIMS[model_id]

    def test_vectors_are_finite_and_nonzero(self, perch):
        result = perch.embed_file(GUYVIR)
        assert np.isfinite(result.vectors).all()
        assert np.linalg.norm(result.vectors, axis=1).min() > 0

    def test_extraction_is_deterministic(self, perch):
        first = perch.embed_file(GUYVIR)
        second = perch.embed_file(GUYVIR)
        np.testing.assert_array_equal(first.vectors, second.vectors)

    def test_different_audio_gives_different_vectors(self, perch):
        guyvir = perch.embed_file(GUYVIR).vectors[0]
        zamtym = perch.embed_file(ZAMTYM).vectors[0]
        cosine = float(
            guyvir @ zamtym
            / (np.linalg.norm(guyvir) * np.linalg.norm(zamtym))
        )
        assert cosine < 0.999

    def test_spans_are_in_the_sample_time_base(self, birdnet):
        result = birdnet.embed_file(GUYVIR)
        assert result.spans[0] == (0.0, 3.0)
        assert result.spans[1] == (1.5, 4.5)

    def test_as_rows_matches_repository_shape(self, birdnet):
        rows = birdnet.embed_file(GUYVIR).as_rows()
        assert len(rows) == 3
        start, end, vector = rows[0]
        assert (start, end) == (0.0, 3.0)
        assert vector.shape == (EXPECTED_DIMS["birdnet_v2.4"],)


@pytest.fixture(scope="module")
def shifted(shared_pool):
    """Perch embeddings of the example at 1.0, 0.5 and 0.25, computed once.

    Speed 0.25 stretches a 5 s sample to 20 s of audio, making this the most
    expensive inference in the suite (~4 s of Perch). Computing it once and
    asserting against it several times keeps the coverage without the repetition.
    """
    extractor = shared_pool.get("perch_v2")
    return {
        speed: extractor.embed_file(GUYVIR, speed=speed)
        for speed in (1.0, 0.5, 0.25)
    }


@pytest.mark.models
class TestSpeedShift:
    """Window counts and span arithmetic are covered cheaply and exhaustively in
    test_audio.py; these confirm the real pipeline agrees."""

    def test_slower_speed_yields_more_windows(self, shifted):
        assert shifted[1.0].n_windows == 1
        assert shifted[0.5].n_windows == 3
        assert shifted[0.25].n_windows == 7

    def test_spans_stay_inside_the_real_sample(self, shifted):
        # The shifted timeline is 20 s long, but the stored spans describe the
        # real 5 s file so the player can mark the right region.
        result = shifted[0.25]
        assert result.spans[0][0] == 0.0
        assert result.spans[-1][1] == pytest.approx(5.0)

    def test_speed_changes_the_embedding(self, shifted):
        # This is why the query-side sweep exists (PLAN.md D2c): the same audio
        # at a different speed lands somewhere else in embedding space, so a
        # native-speed query would not match a slowed-down library item.
        native = shifted[1.0].vectors[0]
        slowed = shifted[0.25].vectors[0]
        cosine = float(
            native @ slowed / (np.linalg.norm(native) * np.linalg.norm(slowed))
        )
        assert cosine < 0.95

    def test_speed_shift_is_reproducible(self, perch, shifted):
        again = perch.embed_file(GUYVIR, speed=0.5)
        np.testing.assert_array_equal(shifted[0.5].vectors, again.vectors)


@pytest.mark.models
class TestExtractorPool:
    def test_embeds_with_every_model(self, shared_pool):
        results = shared_pool.embed_file(GUYVIR)
        assert set(results) == {"perch_v2", "birdnet_v2.4"}
        assert results["perch_v2"].n_windows == 1
        assert results["birdnet_v2.4"].n_windows == 3

    def test_reuses_sessions_across_calls(self, shared_pool):
        # Reloading a 45 MB graph per item would dominate ingest time.
        assert shared_pool.get("perch_v2") is shared_pool.get("perch_v2")

    def test_multi_speed_sweep_covers_the_cross_product(self, shared_pool):
        from biotimbre.embeddings.audio import load_mono

        samples, rate = load_mono(GUYVIR)
        results = shared_pool.embed_array_multi_speed(samples, rate, [1.0])
        assert set(results) == {("perch_v2", 1.0), ("birdnet_v2.4", 1.0)}

    def test_rejects_unknown_model(self):
        # No session needed: the check happens before any model is built.
        pool = ExtractorPool(("perch_v2",))
        with pytest.raises(KeyError, match="not in this pool"):
            pool.get("birdnet_v2.4")
