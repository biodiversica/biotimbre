"""Service-layer tests, including the real end-to-end ingest and search."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from biotimbre.config import Settings
from biotimbre.domain.errors import NotFoundError, ValidationError
from biotimbre.domain.item import ItemDraft, Location, SampleWindow
from biotimbre.search.filters import SearchFilters
from biotimbre.service import LibraryService
from conftest import EXAMPLES_DIR

GUYVIR_WAV = EXAMPLES_DIR / "GUYVIR.wav"
GUYVIR_TXT = EXAMPLES_DIR / "GUYVIR.txt"
ZAMTYM_WAV = EXAMPLES_DIR / "ZAMTYM.wav"


@pytest.fixture
def service(settings: Settings, conn, shared_pool):
    """A service over an empty temporary library.

    Shares the session-scoped ONNX pool, so each test pays for inference but not
    for ~2 s of session setup.
    """
    svc = LibraryService(settings, connection=conn, pool=shared_pool)
    yield svc
    svc.close()


@pytest.fixture
def draft_for_guyvir() -> ItemDraft:
    from biotimbre.io.sidecar import draft_from_sidecar

    return draft_from_sidecar(GUYVIR_TXT)


class TestIngestWithoutModels:
    """Metadata-only ingest, so these need no model files."""

    def test_stores_audio_under_biotimbre_and_item_id(self, service, draft_for_guyvir):
        result = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        )
        item = result.item
        assert item.audio_path == f"audio/{item.biotimbre_id}/{item.item_id}.wav"
        stored = service.settings.absolute_audio_path(item.audio_path)
        assert stored.is_file()
        assert stored.read_bytes() == GUYVIR_WAV.read_bytes()

    def test_source_file_is_copied_not_moved(self, service, draft_for_guyvir):
        service.add_item(GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False)
        assert GUYVIR_WAV.is_file()

    def test_reads_audio_properties(self, service, draft_for_guyvir):
        item = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        ).item
        assert item.sample_rate == 32_000
        assert item.channels == 1
        assert item.duration_s == pytest.approx(5.0)

    def test_reports_duplicate_audio_without_refusing(self, service, draft_for_guyvir):
        first = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        )
        second = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        )
        # The same sound can legitimately be filed twice, so this reports rather
        # than blocks.
        assert second.duplicate_of == (first.item.item_id,)
        assert second.item.item_id != first.item.item_id

    def test_missing_audio_file_raises(self, service, draft_for_guyvir):
        with pytest.raises(ValidationError, match="No such audio file"):
            service.add_item("/nonexistent.wav", draft_for_guyvir)

    def test_failed_ingest_leaves_no_row_and_no_file(self, service, draft_for_guyvir):
        # Force the embedding step to fail after the row and file are written.
        broken = dataclasses.replace(draft_for_guyvir, speed=-1.0)
        with pytest.raises(ValidationError):
            service.add_item(GUYVIR_WAV, broken, compute_embeddings=False)
        assert service.repo.count_items() == 0
        assert not any(service.settings.audio_dir.rglob("*.wav"))

    def test_add_from_sidecar_finds_the_txt(self, service):
        result = service.add_from_sidecar(GUYVIR_WAV)
        assert result.item.device == "Audiomoth"
        assert result.item.taxa == ()

    def test_add_from_sidecar_without_one_raises(self, service, tmp_path):
        import shutil

        lonely = tmp_path / "lonely.wav"
        shutil.copy2(GUYVIR_WAV, lonely)
        with pytest.raises(ValidationError, match="No sidecar"):
            service.add_from_sidecar(lonely)

    def test_import_directory_adds_both_examples(self, service):
        results, failures = service.import_directory(
            EXAMPLES_DIR, compute_embeddings=False
        )
        assert len(results) == 2
        assert failures == []
        assert {r.item.biotimbre_id for r in results} == {"0000", "0001"}

    def test_import_directory_collects_failures(self, service, tmp_path):
        import shutil

        shutil.copy2(GUYVIR_WAV, tmp_path / "good.wav")
        shutil.copy2(GUYVIR_TXT, tmp_path / "good.txt")
        shutil.copy2(GUYVIR_WAV, tmp_path / "orphan.wav")
        results, failures = service.import_directory(
            tmp_path, compute_embeddings=False
        )
        # One bad file must not abort the run.
        assert len(results) == 1
        assert len(failures) == 1
        assert "no sidecar" in failures[0][1]


class TestNyquistAdvisory:
    def test_warns_when_band_exceeds_model_nyquist(self, service):
        draft = ItemDraft(
            date="2020-01-01",
            time="09:00:00",
            location=Location(-26.0, -48.0),
            sample_window=SampleWindow(0.0, 5.0),
            filtered_band="4000,40000",
        ).validated()
        result = service.add_item(GUYVIR_WAV, draft, compute_embeddings=False)
        # 40 kHz exceeds Perch's 16 kHz and BirdNET's 24 kHz Nyquist, so the
        # upper band would be filtered away silently without this prompt.
        assert len(result.warnings) == 2
        assert any("perch_v2" in w for w in result.warnings)
        assert all("speed" in w for w in result.warnings)

    def test_no_warning_when_speed_brings_it_into_range(self, service):
        draft = ItemDraft(
            date="2020-01-01",
            time="09:00:00",
            location=Location(-26.0, -48.0),
            sample_window=SampleWindow(0.0, 5.0),
            filtered_band="4000,40000",
            speed=0.25,
        ).validated()
        result = service.add_item(GUYVIR_WAV, draft, compute_embeddings=False)
        # 40 kHz x 0.25 = 10 kHz, inside both models' range.
        assert result.warnings == ()

    def test_no_warning_for_an_open_band(self, service, draft_for_guyvir):
        # GUYVIR declares "2000,-": nothing is claimed about the upper bound, so
        # there is nothing to check.
        result = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        )
        assert result.warnings == ()


class TestEditAndDelete:
    def test_delete_removes_row_and_file(self, service, draft_for_guyvir):
        item = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        ).item
        stored = service.settings.absolute_audio_path(item.audio_path)
        service.delete_item(item.item_id)
        assert not stored.exists()
        with pytest.raises(NotFoundError):
            service.get_item(item.item_id)

    def test_delete_can_keep_the_audio(self, service, draft_for_guyvir):
        item = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        ).item
        stored = service.settings.absolute_audio_path(item.audio_path)
        service.delete_item(item.item_id, remove_audio=False)
        assert stored.exists()

    def test_update_metadata_keeps_the_file_in_place(self, service, draft_for_guyvir):
        item = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        ).item
        edited = dataclasses.replace(
            draft_for_guyvir, notes="a note", taxa="cicadidae"
        ).validated()
        updated = service.update_item(item.item_id, edited).item
        assert updated.notes == "a note"
        assert updated.taxa == ("cicadidae",)
        assert updated.audio_path == item.audio_path

    def test_changing_biotimbre_relocates_the_audio(self, service, draft_for_guyvir):
        item = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        ).item
        old_path = service.settings.absolute_audio_path(item.audio_path)

        moved = dataclasses.replace(draft_for_guyvir, biotimbre_id="00FF").validated()
        updated = service.update_item(item.item_id, moved).item

        assert updated.biotimbre_id == "00FF"
        assert updated.audio_path == f"audio/00FF/{item.item_id}.wav"
        new_path = service.settings.absolute_audio_path(updated.audio_path)
        assert new_path.is_file()
        assert not old_path.exists()

    def test_get_audio_returns_path_and_mime(self, service, draft_for_guyvir):
        item = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        ).item
        path, mime = service.get_audio(item.item_id)
        assert path.is_file()
        assert mime == "audio/wav"

    def test_get_audio_raises_when_file_is_gone(self, service, draft_for_guyvir):
        item = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        ).item
        service.settings.absolute_audio_path(item.audio_path).unlink()
        with pytest.raises(NotFoundError, match="missing"):
            service.get_audio(item.item_id)


@pytest.mark.models
class TestEndToEnd:
    """Real ingest and search through both ONNX backbones."""

    def test_ingest_computes_both_models(self, service, draft_for_guyvir):
        result = service.add_item(GUYVIR_WAV, draft_for_guyvir)
        assert result.window_counts == {"perch_v2": 1, "birdnet_v2.4": 3}
        counts = service.repo.embedding_counts(result.item.item_id)
        assert counts == {"perch_v2": 1, "birdnet_v2.4": 3}

    def test_registers_models_with_graph_dimensions(self, service):
        dims = service.ensure_models_registered()
        assert dims == {"perch_v2": 1536, "birdnet_v2.4": 1024}

    def test_searching_with_a_stored_sample_finds_it_first(self, service):
        results, failures = service.import_directory(EXAMPLES_DIR)
        assert failures == []
        by_name = {
            r.item.item_id: r.item.audio_sha256 for r in results
        }

        hits = service.search(GUYVIR_WAV, k=5)
        assert hits, "query should match something"

        import hashlib

        guyvir_digest = hashlib.sha256(GUYVIR_WAV.read_bytes()).hexdigest()
        assert by_name[hits[0].item.item_id] == guyvir_digest
        # Identical audio through the same pipeline: cosine should be ~1.
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)

    def test_search_reports_per_model_scores_and_matched_span(self, service):
        service.import_directory(EXAMPLES_DIR)
        hit = service.search(GUYVIR_WAV, k=1)[0]
        assert set(hit.per_model) == {"perch_v2", "birdnet_v2.4"}
        span = hit.matched_span
        assert span is not None
        # Clamped to the sample, so a UI can seek there. BirdNET's third window
        # genuinely ends at 6.0 s on a 5 s file.
        assert 0.0 <= span[0] < span[1] <= hit.item.duration_s

    def test_search_distinguishes_the_two_examples(self, service):
        service.import_directory(EXAMPLES_DIR)
        guyvir_hit = service.search(GUYVIR_WAV, k=2)[0]
        zamtym_hit = service.search(ZAMTYM_WAV, k=2)[0]
        assert guyvir_hit.item.item_id != zamtym_hit.item.item_id

    def test_filter_narrows_the_search(self, service):
        service.import_directory(EXAMPLES_DIR)
        # Only ZAMTYM declares taxa; GUYVIR has none.
        hits = service.search(
            GUYVIR_WAV, filters=SearchFilters(taxa=["cicadidae"]), k=5
        )
        assert len(hits) == 1
        assert hits[0].item.taxa == ("cicadidae",)

    def test_filter_matching_nothing_returns_no_hits(self, service):
        service.import_directory(EXAMPLES_DIR)
        hits = service.search(
            GUYVIR_WAV, filters=SearchFilters(taxa=["mammalia"]), k=5
        )
        assert hits == []

    def test_single_model_fusion(self, service):
        service.import_directory(EXAMPLES_DIR)
        hit = service.search(GUYVIR_WAV, k=1, fusion="perch_v2")[0]
        assert set(hit.per_model) == {"perch_v2"}

    def test_export_npy_round_trips(self, service, draft_for_guyvir, tmp_path):
        item = service.add_item(GUYVIR_WAV, draft_for_guyvir).item
        out = service.export_npy(item.item_id, "birdnet_v2.4", tmp_path)
        loaded = np.load(out)
        assert loaded.shape == (3, 1024)
        stored, _ = service.repo.get_embedding_matrix(item.item_id, "birdnet_v2.4")
        np.testing.assert_array_equal(loaded, stored)

    def test_export_npy_without_embeddings_raises(self, service, draft_for_guyvir, tmp_path):
        item = service.add_item(
            GUYVIR_WAV, draft_for_guyvir, compute_embeddings=False
        ).item
        with pytest.raises(NotFoundError, match="no perch_v2 embeddings"):
            service.export_npy(item.item_id, "perch_v2", tmp_path)

    def test_backfill_embeds_deferred_items(self, service):
        results, _ = service.import_directory(EXAMPLES_DIR, compute_embeddings=False)
        assert all(r.window_counts == {} for r in results)

        written = service.backfill("perch_v2")
        assert len(written) == 2
        assert all(count == 1 for count in written.values())
        # Only the backfilled model is populated.
        counts = service.repo.embedding_counts(results[0].item.item_id)
        assert counts == {"perch_v2": 1}

    def test_backfill_is_idempotent(self, service):
        service.import_directory(EXAMPLES_DIR)
        assert service.backfill("perch_v2") == {}


@pytest.mark.models
class TestSpeedShiftEndToEnd:
    """The D2c regression: the failure mode the query-side sweep exists for."""

    def test_speed_shifted_item_is_found_by_a_native_speed_query(self, service):
        from biotimbre.io.sidecar import draft_from_sidecar

        # 0.5 rather than 0.25: the sweep either works at a shifted speed or it
        # does not, and 0.5 halves the inference. The 0.25 window arithmetic is
        # asserted in test_changing_speed_recomputes_embeddings.
        draft = dataclasses.replace(
            draft_from_sidecar(GUYVIR_TXT), speed=0.5
        ).validated()
        item = service.add_item(GUYVIR_WAV, draft).item
        assert item.speed == 0.5

        # The query is submitted at native speed, with no hint that the library
        # item was indexed at 0.5. Without the sweep this returns nothing useful.
        hits = service.search(GUYVIR_WAV, k=5)

        assert hits, "a speed-shifted item must still be findable"
        assert hits[0].item.item_id == item.item_id
        assert hits[0].query_speed == 0.5
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)

    def test_pinning_the_wrong_query_speed_scores_worse_but_still_returns(self, service):
        # Demonstrates that the sweep is load-bearing rather than incidental:
        # forcing the query to 1.0 against an item indexed at 0.25 scores clearly
        # worse than the sweep's automatic match.
        from biotimbre.io.sidecar import draft_from_sidecar

        draft = dataclasses.replace(
            draft_from_sidecar(GUYVIR_TXT), speed=0.25
        ).validated()
        service.add_item(GUYVIR_WAV, draft)

        swept = service.search(GUYVIR_WAV, k=1)
        pinned = service.search(GUYVIR_WAV, k=1, query_speed=1.0)

        # Pinning must degrade the score, not empty the results: returning nothing
        # would read as "the library has no match" rather than "your speed did
        # not line up with anything indexed".
        assert pinned, "a pinned speed must still return hits"
        assert swept[0].score > pinned[0].score
        assert pinned[0].query_speed == 1.0

    def test_mixed_speed_library_scores_each_item_correctly(self, service):
        from biotimbre.io.sidecar import draft_from_sidecar

        native = draft_from_sidecar(GUYVIR_TXT)
        shifted = dataclasses.replace(native, speed=0.5).validated()

        native_item = service.add_item(GUYVIR_WAV, native).item.item_id
        shifted_item = service.add_item(GUYVIR_WAV, shifted).item.item_id
        assert service.repo.distinct_speeds() == [0.5, 1.0]

        hits = service.search(GUYVIR_WAV, k=5)
        by_id = {hit.item.item_id: hit for hit in hits}
        # Both are the same audio, so both should match near-perfectly, each
        # against the query embedded at its own speed.
        assert by_id[native_item].query_speed == 1.0
        assert by_id[shifted_item].query_speed == 0.5
        assert by_id[native_item].score == pytest.approx(1.0, abs=1e-3)
        assert by_id[shifted_item].score == pytest.approx(1.0, abs=1e-3)

    def test_changing_speed_recomputes_embeddings(self, service):
        from biotimbre.io.sidecar import draft_from_sidecar

        draft = draft_from_sidecar(GUYVIR_TXT)
        item = service.add_item(GUYVIR_WAV, draft).item
        before, _ = service.repo.get_embedding_matrix(item.item_id, "perch_v2")
        assert before.shape[0] == 1

        edited = dataclasses.replace(draft, speed=0.25).validated()
        result = service.update_item(item.item_id, edited)

        # Speed 0.25 stretches 5 s to 20 s of shifted audio: 7 Perch windows.
        assert result.window_counts["perch_v2"] == 7
        after, _ = service.repo.get_embedding_matrix(item.item_id, "perch_v2")
        assert after.shape[0] == 7
