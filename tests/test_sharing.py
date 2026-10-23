"""Sharing items between libraries: export, plan, import.

Every test here runs two real libraries — one sends, one receives — because
that is the only arrangement in which the questions mean anything. None of
them need model files: a fake model is registered directly, which is enough to
exercise fingerprint comparison and vector reuse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biotimbre.config import Settings
from biotimbre.domain.errors import NotFoundError, ValidationError
from biotimbre.domain.item import ItemDraft, Location, generate_biotimbre_id
from biotimbre.io.bundle import BUNDLE_SUFFIX, read_bundle
from biotimbre.models_registry import ModelSpec
from biotimbre.service import LibraryService, bundle_filename
from biotimbre.sharing import ImportPolicy, fingerprint_from_row
from conftest import EXAMPLES_DIR, unit_vectors

GUYVIR_WAV = EXAMPLES_DIR / "GUYVIR.wav"
ZAMTYM_WAV = EXAMPLES_DIR / "ZAMTYM.wav"

FAKE_MODEL = "fake_model"
FAKE_DIM = 16

#: The model both libraries agree on, so vectors may cross between them.
FAKE_SPEC = ModelSpec(
    model_id=FAKE_MODEL,
    repo_id="test/fake",
    filename="fake.onnx",
    sample_rate=32_000,
    window_duration_s=5.0,
    window_overlap=0.5,
    description="test double",
)

#: Nothing here computes embeddings: there is no graph behind the fake model.
#: Vectors are placed directly, which is what makes these tests fast and
#: offline while still exercising every path that moves them.
OFFLINE = ImportPolicy(compute_embeddings=False)


@pytest.fixture
def library(tmp_path):
    """Build independent libraries, as two people on two machines would have."""
    services: list[LibraryService] = []

    def make(name: str, *, spec: ModelSpec | None = FAKE_SPEC, dim: int = FAKE_DIM):
        data_dir = tmp_path / name
        settings = Settings(
            data_dir=data_dir,
            db_path=data_dir / "library.db",
            audio_dir=data_dir / "audio",
            active_models=(FAKE_MODEL,),
        )
        service = LibraryService(settings)
        service.rename(name)
        if spec is not None:
            with service._transaction():
                service.repo.register_model(spec, dim)
        services.append(service)
        return service

    yield make
    for service in services:
        service.close()


def add(service: LibraryService, audio: Path = GUYVIR_WAV, **fields):
    """Add one item with no embeddings, returning it."""
    draft = ItemDraft(
        date=fields.pop("date", "2020-01-12"),
        time=fields.pop("time", "09:40:21"),
        location=fields.pop("location", Location(-26.0, -48.0)),
        **fields,
    )
    return service.add_item(audio, draft, compute_embeddings=False).item


def embed(service: LibraryService, item_id: int, n: int = 3, seed: int = 0):
    """Place vectors for the fake model, as an inference pass would have."""
    vectors = unit_vectors(n, FAKE_DIM, seed=seed)
    rows = [(index * 2.5, index * 2.5 + 5.0, vectors[index]) for index in range(n)]
    with service._transaction():
        service.repo.set_embeddings(item_id, FAKE_MODEL, rows)
    return rows


class TestExport:
    def test_writes_a_bundle_that_reads_back(self, library, tmp_path):
        sender = library("sender")
        item = add(sender, taxa="cicadidae")
        path = sender.export_bundle([item.item_id], tmp_path / "share")

        assert path.suffix == BUNDLE_SUFFIX
        bundle = read_bundle(path, into=tmp_path / "unpacked")
        assert len(bundle) == 1
        assert bundle.entries[0].draft.biotimbre_id == item.biotimbre_id

    def test_records_the_sending_library(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        bundle = read_bundle(path, into=tmp_path / "unpacked")
        assert bundle.source_library.name == "sender"
        assert bundle.source_library.uuid == sender.identity().uuid
        assert bundle.entries[0].provenance.item_id == item.item_id

    def test_carries_the_biotimbre_label(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        sender.set_biotimbre_label(item.biotimbre_id, "Big green one")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        bundle = read_bundle(path, into=tmp_path / "unpacked")
        assert bundle.entries[0].biotimbre_label == "Big green one"

    def test_carries_embeddings_and_their_fingerprint(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        rows = embed(sender, item.item_id, n=3)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        bundle = read_bundle(path, into=tmp_path / "unpacked")
        carried = bundle.entries[0].embeddings[FAKE_MODEL]
        assert len(carried) == 3
        assert np.array_equal(carried[0][2], rows[0][2])
        assert bundle.models[FAKE_MODEL].embedding_dim == FAKE_DIM

    def test_can_leave_embeddings_out(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        embed(sender, item.item_id)
        path = sender.export_bundle(
            [item.item_id], tmp_path / "share.btz", include_embeddings=False
        )

        bundle = read_bundle(path, into=tmp_path / "unpacked")
        assert bundle.entries[0].embeddings == {}
        assert bundle.models == {}

    def test_exports_several_items_in_order(self, library, tmp_path):
        sender = library("sender")
        first = add(sender, taxa="cicadidae")
        second = add(sender, ZAMTYM_WAV, taxa="orthoptera")
        path = sender.export_bundle(
            [second.item_id, first.item_id], tmp_path / "share.btz"
        )

        bundle = read_bundle(path, into=tmp_path / "unpacked")
        assert [e.provenance.item_id for e in bundle.entries] == [
            second.item_id,
            first.item_id,
        ]

    def test_bytes_match_the_file(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        data, name = sender.export_bundle_bytes([item.item_id])

        assert name == bundle_filename([item.item_id])
        (tmp_path / name).write_bytes(data)
        bundle = read_bundle(tmp_path / name, into=tmp_path / "unpacked")
        assert bundle.entries[0].draft.biotimbre_id == item.biotimbre_id

    def test_names_one_item_and_many_differently(self):
        assert bundle_filename([12]) == f"biotimbre_item_12{BUNDLE_SUFFIX}"
        assert bundle_filename([12, 13, 14]) == f"biotimbre_3_items{BUNDLE_SUFFIX}"

    def test_refuses_an_unknown_item(self, library, tmp_path):
        sender = library("sender")
        with pytest.raises(NotFoundError):
            sender.export_bundle([999], tmp_path / "share.btz")

    def test_refuses_an_empty_selection(self, library, tmp_path):
        sender = library("sender")
        with pytest.raises(ValidationError, match="Nothing to export"):
            sender.export_bundle([], tmp_path / "share.btz")

    def test_refuses_when_the_audio_is_gone(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        sender.settings.absolute_audio_path(item.audio_path).unlink()
        with pytest.raises(NotFoundError, match="missing from disk"):
            sender.export_bundle([item.item_id], tmp_path / "share.btz")


class TestImport:
    def test_the_item_arrives_whole(self, library, tmp_path):
        sender = library("sender")
        item = add(
            sender,
            taxa="cicadidae",
            device="Audiomoth",
            notes="close by",
            recorded_by="Kaue",
        )
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        report = receiver.import_bundle(path, policy=OFFLINE)

        assert len(report.added) == 1
        arrived = report.added[0].item
        assert arrived.taxa == item.taxa
        assert arrived.device == "Audiomoth"
        assert arrived.notes == "close by"
        assert arrived.recorded_by == "Kaue"
        assert arrived.date == item.date
        assert arrived.location == item.location

    def test_the_biotimbre_identity_is_kept(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        arrived = receiver.import_bundle(path, policy=OFFLINE).added[0].item
        # The point of the whole exercise: one sound-being, one id, both sides.
        assert arrived.biotimbre_id == item.biotimbre_id

    def test_the_item_id_is_local(self, library, tmp_path):
        sender = library("sender")
        for _ in range(4):
            add(sender)
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        arrived = receiver.import_bundle(path, policy=OFFLINE).added[0].item
        assert item.item_id == 5
        assert arrived.item_id == 1

    def test_the_audio_lands_in_the_library_tree(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        arrived = receiver.import_bundle(path, policy=OFFLINE).added[0].item
        stored = receiver.settings.absolute_audio_path(arrived.audio_path)
        assert stored.is_file()
        assert stored.read_bytes() == GUYVIR_WAV.read_bytes()
        assert arrived.audio_sha256 == item.audio_sha256

    def test_the_label_comes_along(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        sender.set_biotimbre_label(item.biotimbre_id, "Big green one")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        receiver.import_bundle(path, policy=OFFLINE)
        assert receiver.get_biotimbre_label(item.biotimbre_id) == "Big green one"

    def test_a_local_label_is_not_overwritten(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        sender.set_biotimbre_label(item.biotimbre_id, "Sender's name")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        other = add(receiver, ZAMTYM_WAV, biotimbre_id=item.biotimbre_id)
        receiver.set_biotimbre_label(other.biotimbre_id, "My own name")
        receiver.import_bundle(path, policy=OFFLINE)

        assert receiver.get_biotimbre_label(item.biotimbre_id) == "My own name"

    def test_provenance_survives(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        outcome = receiver.import_bundle(path, policy=OFFLINE).added[0]
        assert outcome.provenance.library_name == "sender"
        assert outcome.provenance.library_uuid == sender.identity().uuid
        assert outcome.provenance.item_id == item.item_id

    def test_several_items_arrive(self, library, tmp_path):
        sender = library("sender")
        first = add(sender, taxa="cicadidae")
        second = add(sender, ZAMTYM_WAV, taxa="orthoptera")
        path = sender.export_bundle(
            [first.item_id, second.item_id], tmp_path / "share.btz"
        )

        receiver = library("receiver")
        report = receiver.import_bundle(path, policy=OFFLINE)
        assert len(report.added) == 2
        assert receiver.stats()["items"] == 2

    def test_a_round_trip_changes_nothing(self, library, tmp_path):
        sender = library("sender")
        item = add(sender, taxa="cicadidae", temperature=21.5, humidity=80.0)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        arrived = receiver.import_bundle(path, policy=OFFLINE).added[0].item
        # Everything but the two facts that are local by definition.
        assert arrived.to_draft() == item.to_draft()


class TestEmbeddingReuse:
    def test_matching_vectors_are_taken_as_they_are(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        rows = embed(sender, item.item_id, n=3)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        outcome = receiver.import_bundle(path, policy=OFFLINE).added[0]

        assert outcome.plan.reusable == (FAKE_MODEL,)
        assert outcome.window_counts == {FAKE_MODEL: 3}
        stored = receiver.repo.get_embeddings(outcome.item.item_id, FAKE_MODEL)
        assert len(stored) == 3
        for (_, start, end, vector), (o_start, o_end, o_vector) in zip(stored, rows):
            assert (start, end) == (o_start, o_end)
            assert np.array_equal(vector, o_vector)

    def test_an_unregistered_model_is_refused(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        embed(sender, item.item_id)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        # A library that has never registered this model has no dimension to
        # check the vectors against.
        receiver = library("receiver", spec=None)
        outcome = receiver.import_bundle(path, policy=OFFLINE).added[0]

        assert outcome.plan.reusable == ()
        assert "not registered" in outcome.plan.rejected[FAKE_MODEL]
        assert receiver.repo.embedding_counts(outcome.item.item_id) == {}

    def test_a_differently_registered_model_is_refused(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        embed(sender, item.item_id)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        # Same id, different recipe: the vectors are not comparable, and the
        # reason names the field.
        import dataclasses

        receiver = library(
            "receiver", spec=dataclasses.replace(FAKE_SPEC, window_overlap=0.0)
        )
        outcome = receiver.import_bundle(path, policy=OFFLINE).added[0]

        assert outcome.plan.reusable == ()
        assert "window_overlap" in outcome.plan.rejected[FAKE_MODEL]

    def test_distrust_refuses_everything(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        embed(sender, item.item_id)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        outcome = receiver.import_bundle(
            path,
            policy=ImportPolicy(trust_embeddings=False, compute_embeddings=False),
        ).added[0]

        assert outcome.plan.reusable == ()
        assert "not trusted" in outcome.plan.rejected[FAKE_MODEL]
        assert receiver.repo.embedding_counts(outcome.item.item_id) == {}

    def test_a_fingerprint_row_round_trips(self, library):
        sender = library("sender")
        row = sender.repo.get_model(FAKE_MODEL)
        fingerprint = fingerprint_from_row(row)
        assert fingerprint.model_id == FAKE_MODEL
        assert fingerprint.embedding_dim == FAKE_DIM
        assert fingerprint.window_overlap == 0.5


class TestDuplicates:
    def test_the_same_item_twice_is_skipped(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        receiver.import_bundle(path, policy=OFFLINE)
        report = receiver.import_bundle(path, policy=OFFLINE)

        assert len(report.added) == 0
        assert len(report.skipped) == 1
        assert "already in this library" in report.skipped[0].reason
        assert receiver.stats()["items"] == 1

    def test_the_same_recording_under_another_biotimbre_is_skipped(
        self, library, tmp_path
    ):
        sender = library("sender")
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        # The receiver catalogued the same recording independently, so it has
        # its own id for it. Two libraries disagree; nothing is guessed.
        receiver = library("receiver")
        add(receiver)
        report = receiver.import_bundle(path, policy=OFFLINE)

        assert len(report.skipped) == 1
        assert "different biotimbre" in report.skipped[0].reason
        assert receiver.stats()["items"] == 1

    def test_it_can_be_imported_anyway(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        add(receiver)
        report = receiver.import_bundle(
            path, policy=ImportPolicy(compute_embeddings=False, duplicate_audio="add")
        )

        assert len(report.added) == 1
        assert any("already holds" in w for w in report.added[0].warnings)
        assert receiver.stats()["items"] == 2


class TestSpeciesConflict:
    def test_it_joins_the_local_biotimbre(self, library, tmp_path):
        sender = library("sender")
        item = add(sender, taxa="guyalna bonaerensis")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        # The receiver already knows this species, under its own id.
        receiver = library("receiver")
        mine = add(receiver, ZAMTYM_WAV, taxa="guyalna bonaerensis")
        assert mine.biotimbre_id != item.biotimbre_id

        outcome = receiver.import_bundle(path, policy=OFFLINE).added[0]

        assert outcome.item.biotimbre_id == mine.biotimbre_id
        assert outcome.plan.rebased
        assert any("joins it" in w for w in outcome.warnings)
        assert item.biotimbre_id in outcome.warnings[0]

    def test_it_can_keep_the_senders_biotimbre(self, library, tmp_path):
        sender = library("sender")
        item = add(sender, taxa="guyalna bonaerensis")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        mine = add(receiver, ZAMTYM_WAV, taxa="guyalna bonaerensis")
        outcome = receiver.import_bundle(
            path,
            policy=ImportPolicy(
                compute_embeddings=False, species_conflict="keep_incoming"
            ),
        ).added[0]

        assert outcome.item.biotimbre_id == item.biotimbre_id
        assert not outcome.plan.rebased
        assert any("two biotimbres" in w for w in outcome.warnings)

    def test_a_rank_does_not_trigger_it(self, library, tmp_path):
        sender = library("sender")
        item = add(sender, taxa="cicadidae")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        # "cicadidae" says nothing about which being was recorded.
        receiver = library("receiver")
        add(receiver, ZAMTYM_WAV, taxa="cicadidae")
        outcome = receiver.import_bundle(path, policy=OFFLINE).added[0]

        assert outcome.item.biotimbre_id == item.biotimbre_id
        assert not outcome.plan.rebased

    def test_open_nomenclature_does_not_trigger_it(self, library, tmp_path):
        sender = library("sender")
        item = add(sender, taxa="guyalna sp.")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        add(receiver, ZAMTYM_WAV, taxa="guyalna sp.")
        outcome = receiver.import_bundle(path, policy=OFFLINE).added[0]

        assert outcome.item.biotimbre_id == item.biotimbre_id

    def test_a_bundle_that_splits_one_species_converges_here(
        self, library, tmp_path
    ):
        # The sender filed one species under two biotimbres. Importing that as
        # it stands would reproduce the mistake; instead the second entry joins
        # what the first settled on.
        sender = library("sender")
        first = add(sender, taxa="guyalna bonaerensis")
        second = add(
            sender,
            ZAMTYM_WAV,
            taxa="guyalna bonaerensis",
            biotimbre_id=generate_biotimbre_id(),
        )
        assert first.biotimbre_id != second.biotimbre_id
        path = sender.export_bundle(
            [first.item_id, second.item_id], tmp_path / "share.btz"
        )

        receiver = library("receiver")
        report = receiver.import_bundle(path, policy=OFFLINE)

        assert len(report.added) == 2
        landed = {outcome.item.biotimbre_id for outcome in report.added}
        assert len(landed) == 1
        assert any("earlier in this bundle" in w for w in report.added[1].warnings)


class TestPreview:
    def test_it_says_what_would_happen_without_doing_it(self, library, tmp_path):
        sender = library("sender")
        item = add(sender, taxa="cicadidae")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        preview = receiver.inspect_bundle(path, policy=OFFLINE)

        assert len(preview) == 1
        assert len(preview.to_add) == 1
        assert preview.plans[0].biotimbre_id == item.biotimbre_id
        assert receiver.stats()["items"] == 0

    def test_it_predicts_a_skip(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        receiver.import_bundle(path, policy=OFFLINE)
        preview = receiver.inspect_bundle(path, policy=OFFLINE)

        assert len(preview.to_skip) == 1
        assert not preview.to_add

    def test_it_predicts_the_species_join(self, library, tmp_path):
        sender = library("sender")
        item = add(sender, taxa="guyalna bonaerensis")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        mine = add(receiver, ZAMTYM_WAV, taxa="guyalna bonaerensis")
        preview = receiver.inspect_bundle(path, policy=OFFLINE)

        assert preview.has_warnings
        assert preview.plans[0].biotimbre_id == mine.biotimbre_id

    def test_the_preview_matches_the_import(self, library, tmp_path):
        sender = library("sender")
        first = add(sender, taxa="guyalna bonaerensis")
        second = add(sender, ZAMTYM_WAV, taxa="cicadidae")
        path = sender.export_bundle(
            [first.item_id, second.item_id], tmp_path / "share.btz"
        )

        receiver = library("receiver")
        add(receiver, taxa="guyalna bonaerensis")
        preview = receiver.inspect_bundle(path, policy=OFFLINE)
        report = receiver.import_bundle(path, policy=OFFLINE)

        assert [p.action for p in preview.plans] == [
            o.action for o in report.outcomes
        ]
        assert [p.biotimbre_id for p in preview.plans] == [
            o.biotimbre_id for o in report.outcomes
        ]

    def test_it_can_keep_the_audio_for_a_caller(self, library, tmp_path):
        sender = library("sender")
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = library("receiver")
        preview = receiver.inspect_bundle(
            path, policy=OFFLINE, into=tmp_path / "kept"
        )
        audio = preview.plans[0].entry.audio
        assert audio.is_file()
        assert audio.read_bytes() == GUYVIR_WAV.read_bytes()

    def test_a_bad_bundle_is_reported_not_imported(self, library, tmp_path):
        receiver = library("receiver")
        broken = tmp_path / "broken.btz"
        broken.write_bytes(b"not a zip")
        with pytest.raises(ValidationError, match="Not a readable bundle"):
            receiver.inspect_bundle(broken)


class TestImportLibrary:
    """Importing every item of another library on the same machine."""

    def test_every_item_arrives(self, library):
        sender = library("sender")
        add(sender, taxa="cicadidae")
        add(sender, ZAMTYM_WAV, taxa="orthoptera")

        receiver = library("receiver")
        report = receiver.import_library(sender.settings.data_dir, policy=OFFLINE)

        assert len(report.added) == 2
        assert {item.taxa for item in receiver.list_items()} == {
            ("cicadidae",),
            ("orthoptera",),
        }

    def test_the_other_library_is_left_alone(self, library):
        sender = library("sender")
        item = add(sender)
        audio = sender.settings.absolute_audio_path(item.audio_path)

        receiver = library("receiver")
        added = receiver.import_library(
            sender.settings.data_dir, policy=OFFLINE
        ).added[0]

        assert audio.is_file()
        assert len(sender.list_items()) == 1
        stored = receiver.settings.absolute_audio_path(added.item.audio_path)
        assert stored.is_relative_to(receiver.settings.audio_dir)

    def test_identity_label_and_provenance_come_along(self, library):
        sender = library("sender")
        item = add(sender, taxa="cicadidae")
        sender.set_biotimbre_label(item.biotimbre_id, "Big green one")

        receiver = library("receiver")
        outcome = receiver.import_library(
            sender.settings.data_dir, policy=OFFLINE
        ).added[0]

        assert outcome.item.biotimbre_id == item.biotimbre_id
        assert receiver.get_biotimbre_label(item.biotimbre_id) == "Big green one"
        assert outcome.provenance.library_name == "sender"
        assert outcome.provenance.item_id == item.item_id

    def test_matching_vectors_are_reused(self, library):
        sender = library("sender")
        item = add(sender)
        rows = embed(sender, item.item_id)

        receiver = library("receiver")
        added = receiver.import_library(
            sender.settings.data_dir, policy=OFFLINE
        ).added[0]

        stored = receiver.repo.get_embeddings(added.item.item_id, FAKE_MODEL)
        assert len(stored) == len(rows)
        assert np.array_equal(stored[0][3], rows[0][2])

    def test_a_second_import_skips_everything(self, library):
        sender = library("sender")
        add(sender)
        add(sender, ZAMTYM_WAV)

        receiver = library("receiver")
        receiver.import_library(sender.settings.data_dir, policy=OFFLINE)
        report = receiver.import_library(sender.settings.data_dir, policy=OFFLINE)

        assert len(report.added) == 0
        assert len(report.skipped) == 2
        assert len(receiver.list_items()) == 2

    def test_species_join_the_local_biotimbre(self, library):
        sender = library("sender")
        add(sender, taxa="guyalna bonaerensis")

        receiver = library("receiver")
        mine = add(receiver, ZAMTYM_WAV, taxa="guyalna bonaerensis")
        outcome = receiver.import_library(
            sender.settings.data_dir, policy=OFFLINE
        ).added[0]

        assert outcome.item.biotimbre_id == mine.biotimbre_id

    def test_the_preview_changes_nothing_and_matches(self, library):
        sender = library("sender")
        add(sender)
        add(sender, ZAMTYM_WAV)

        receiver = library("receiver")
        external = receiver.open_external_library(sender.settings.data_dir)
        preview = receiver.inspect_library(external, policy=OFFLINE)

        assert len(preview.to_add) == 2
        assert receiver.list_items() == []
        report = receiver.import_library(external, policy=OFFLINE)
        assert [o.action for o in report.outcomes] == [
            p.action for p in preview.plans
        ]

    def test_an_item_without_audio_is_left_out(self, library):
        sender = library("sender")
        gone = add(sender)
        kept = add(sender, ZAMTYM_WAV)
        sender.settings.absolute_audio_path(gone.audio_path).unlink()

        receiver = library("receiver")
        external = receiver.open_external_library(sender.settings.data_dir)

        assert external.missing_audio == (gone.item_id,)
        assert [e.provenance.item_id for e in external.bundle.entries] == [
            kept.item_id
        ]

    def test_refuses_the_open_library(self, library):
        receiver = library("receiver")
        add(receiver)
        with pytest.raises(ValidationError, match="already open"):
            receiver.open_external_library(receiver.settings.data_dir)

    def test_refuses_a_directory_without_a_library(self, library, tmp_path):
        receiver = library("receiver")
        (tmp_path / "nothing").mkdir()
        with pytest.raises(ValidationError, match="No biotimbre library"):
            receiver.open_external_library(tmp_path / "nothing")

    def test_refuses_an_empty_library(self, library):
        sender = library("sender")
        receiver = library("receiver")
        with pytest.raises(ValidationError, match="no items"):
            receiver.open_external_library(sender.settings.data_dir)


class TestPolicyValidation:
    def test_rejects_an_unknown_duplicate_policy(self):
        with pytest.raises(ValidationError, match="duplicate_audio"):
            ImportPolicy(duplicate_audio="explode")

    def test_rejects_an_unknown_species_policy(self):
        with pytest.raises(ValidationError, match="species_conflict"):
            ImportPolicy(species_conflict="whatever")


@pytest.mark.models
class TestMixedEmbeddings:
    """The default policy: take what fits, compute the rest.

    These need the real model files, because the point is what happens when an
    import must run inference for one model and not for another.
    """

    @pytest.fixture
    def real_library(self, tmp_path, shared_pool):
        """Libraries over the real models, sharing the session's ONNX sessions."""
        services: list[LibraryService] = []

        def make(name: str, models: tuple[str, ...]):
            data_dir = tmp_path / name
            settings = Settings(
                data_dir=data_dir,
                db_path=data_dir / "library.db",
                audio_dir=data_dir / "audio",
                active_models=models,
            )
            service = LibraryService(settings, pool=shared_pool)
            service.rename(name)
            services.append(service)
            return service

        yield make
        for service in services:
            service.close()

    def test_one_model_is_reused_and_the_other_computed(
        self, real_library, tmp_path
    ):
        # The sender runs only Perch, the receiver runs both.
        sender = real_library("sender", ("perch_v2",))
        item = sender.add_item(
            GUYVIR_WAV,
            ItemDraft(
                date="2020-01-12", time="09:40:21", location=Location(-26.0, -48.0)
            ),
        ).item
        assert set(sender.repo.embedding_counts(item.item_id)) == {"perch_v2"}
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = real_library("receiver", ("perch_v2", "birdnet_v2.4"))
        outcome = receiver.import_bundle(path).added[0]

        assert outcome.plan.reusable == ("perch_v2",)
        assert set(outcome.window_counts) == {"perch_v2", "birdnet_v2.4"}
        assert set(receiver.repo.embedding_counts(outcome.item.item_id)) == {
            "perch_v2",
            "birdnet_v2.4",
        }

    def test_the_reused_vectors_are_not_recomputed(self, real_library, tmp_path):
        sender = real_library("sender", ("perch_v2",))
        item = sender.add_item(
            GUYVIR_WAV,
            ItemDraft(
                date="2020-01-12", time="09:40:21", location=Location(-26.0, -48.0)
            ),
        ).item
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")
        theirs = sender.repo.get_embeddings(item.item_id, "perch_v2")

        receiver = real_library("receiver", ("perch_v2", "birdnet_v2.4"))
        arrived = receiver.import_bundle(path).added[0].item
        mine = receiver.repo.get_embeddings(arrived.item_id, "perch_v2")

        # Bit-identical: these are the sender's numbers, not a second run's.
        assert len(mine) == len(theirs)
        for (_, start, end, vector), (_, o_start, o_end, o_vector) in zip(
            mine, theirs
        ):
            assert (start, end) == (o_start, o_end)
            assert np.array_equal(vector, o_vector)

    def test_deferring_leaves_gaps_for_backfill(self, real_library, tmp_path):
        sender = real_library("sender", ("perch_v2",))
        item = sender.add_item(
            GUYVIR_WAV,
            ItemDraft(
                date="2020-01-12", time="09:40:21", location=Location(-26.0, -48.0)
            ),
        ).item
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = real_library("receiver", ("perch_v2", "birdnet_v2.4"))
        arrived = receiver.import_bundle(path, policy=OFFLINE).added[0].item

        assert set(receiver.repo.embedding_counts(arrived.item_id)) == {"perch_v2"}
        assert receiver.repo.items_missing_embeddings("birdnet_v2.4") == [
            arrived.item_id
        ]

    def test_distrust_recomputes_from_the_audio(self, real_library, tmp_path):
        sender = real_library("sender", ("perch_v2",))
        item = sender.add_item(
            GUYVIR_WAV,
            ItemDraft(
                date="2020-01-12", time="09:40:21", location=Location(-26.0, -48.0)
            ),
        ).item
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        receiver = real_library("receiver", ("perch_v2",))
        outcome = receiver.import_bundle(
            path, policy=ImportPolicy(trust_embeddings=False)
        ).added[0]

        assert outcome.plan.reusable == ()
        # Recomputed, and the same audio through the same model agrees.
        mine = receiver.repo.get_embeddings(outcome.item.item_id, "perch_v2")
        theirs = sender.repo.get_embeddings(item.item_id, "perch_v2")
        assert len(mine) == len(theirs)
        assert np.allclose(mine[0][3], theirs[0][3], atol=1e-5)
