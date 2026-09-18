"""The shareable bundle format: what it carries, and what it refuses."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import (
    FilteredBand,
    ItemDraft,
    Location,
    SampleWindow,
    generate_biotimbre_id,
)
from biotimbre.io.bundle import (
    AUDIO_NAME,
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    MANIFEST_NAME,
    Bundle,
    ExportEntry,
    ModelFingerprint,
    SourceLibrary,
    draft_from_json,
    draft_to_json,
    read_bundle,
    write_bundle,
)

FINGERPRINT = ModelFingerprint(
    model_id="perch_v2",
    source="hf://google/perch",
    filename="perch.onnx",
    revision=None,
    sample_rate=32_000,
    window_duration_s=5.0,
    window_overlap=0.5,
    embedding_dim=8,
)


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    """A short real WAV, so audio handling is exercised on actual audio."""
    import soundfile as sf

    path = tmp_path / "source.wav"
    samples = np.sin(
        2 * np.pi * 440 * np.arange(32_000, dtype=np.float32) / 32_000
    )
    sf.write(path, samples, 32_000)
    return path


@pytest.fixture
def full_draft() -> ItemDraft:
    """A draft with every field set, so nothing can be silently dropped."""
    return ItemDraft(
        date="2020-01-14",
        time="09:40:21",
        location=Location(-20.5, -46.25),
        sample_window=SampleWindow(5.0, 10.0),
        biotimbre_id=generate_biotimbre_id(),
        taxa="cicadidae, guyalna bonaerensis",
        device="Audiomoth",
        microphone="built-in",
        recorded_by="Kaue",
        filtered_band=FilteredBand(2000.0, 10_000.0),
        filter_type="Butterworth",
        filter_order=4,
        temperature=21.5,
        humidity=78.0,
        speed=0.25,
        link_to_full_recording="https://example.org/rec/1",
        notes="two cicadas, one close",
    ).validated()


def rows(n: int, dim: int = 8, seed: int = 0):
    """``n`` window rows of width ``dim``, spans overlapping by half."""
    rng = np.random.default_rng(seed)
    return [
        (
            float(index) * 2.5,
            float(index) * 2.5 + 5.0,
            rng.standard_normal(dim).astype(np.float32),
        )
        for index in range(n)
    ]


def entry_for(draft: ItemDraft, wav: Path, **overrides) -> ExportEntry:
    from biotimbre.embeddings.audio import file_sha256

    defaults = dict(
        draft=draft,
        audio=wav,
        audio_sha256=file_sha256(wav),
        duration_s=1.0,
        sample_rate=32_000,
        channels=1,
        biotimbre_label="Guyalna bonaerensis",
        embeddings={},
        source_item_id=12,
    )
    return ExportEntry(**{**defaults, **overrides})


def round_trip(entries, tmp_path: Path, **kwargs) -> Bundle:
    """Write a bundle and read it straight back."""
    path = tmp_path / "out" / "share.btz"
    write_bundle(path, entries, **kwargs)
    return read_bundle(path, into=tmp_path / "unpacked")


class TestDraftJson:
    def test_round_trips_every_field(self, full_draft: ItemDraft):
        assert draft_from_json(draft_to_json(full_draft)) == full_draft

    def test_round_trips_a_minimal_draft(self):
        draft = ItemDraft(
            date="2020-01-14",
            time="09:40:21",
            location=Location(0.0, 0.0),
            biotimbre_id=generate_biotimbre_id(),
        ).validated()
        restored = draft_from_json(draft_to_json(draft))
        assert restored == draft
        assert restored.sample_window is None
        assert restored.filtered_band.is_unset
        assert restored.taxa == ()

    def test_json_is_plain_types(self, full_draft: ItemDraft):
        # It has to survive json.dumps without a custom encoder, or it is not
        # a wire format.
        assert json.loads(json.dumps(draft_to_json(full_draft)))

    def test_field_names_match_the_sidecar(self, full_draft: ItemDraft):
        from biotimbre.io.sidecar import SIDECAR_FIELDS

        assert set(draft_to_json(full_draft)) == set(SIDECAR_FIELDS)

    def test_accepts_the_sidecars_string_shapes(self):
        # A sender's writer need not be this one.
        draft = draft_from_json(
            {
                "date": "2020-01-14",
                "time": "09:40:21",
                "location": "-20,-46",
                "sample_window": "5,10",
                "filtered_band": "2000,-",
            }
        )
        assert draft.location.latitude == -20.0
        assert draft.sample_window.end == 10.0
        assert draft.filtered_band.high is None

    def test_rejects_a_missing_mandatory_field(self):
        with pytest.raises(ValidationError, match="mandatory"):
            draft_from_json({"date": "2020-01-14", "time": "09:40:21"})

    def test_rejects_a_bad_value(self):
        with pytest.raises(ValidationError):
            draft_from_json(
                {
                    "date": "2020-01-14",
                    "time": "09:40:21",
                    "location": {"latitude": 200.0, "longitude": 0.0},
                }
            )

    def test_rejects_a_half_location(self):
        with pytest.raises(ValidationError, match="latitude"):
            draft_from_json(
                {
                    "date": "2020-01-14",
                    "time": "09:40:21",
                    "location": {"latitude": 1.0},
                }
            )


class TestRoundTrip:
    def test_metadata_survives_intact(self, full_draft, wav, tmp_path):
        bundle = round_trip([entry_for(full_draft, wav)], tmp_path)
        assert len(bundle) == 1
        assert bundle.entries[0].draft == full_draft

    def test_biotimbre_identity_travels(self, full_draft, wav, tmp_path):
        # The whole point: the recipient files this under the same identity.
        bundle = round_trip([entry_for(full_draft, wav)], tmp_path)
        assert bundle.entries[0].draft.biotimbre_id == full_draft.biotimbre_id
        assert bundle.biotimbre_ids == (full_draft.biotimbre_id,)

    def test_audio_is_byte_identical(self, full_draft, wav, tmp_path):
        bundle = round_trip([entry_for(full_draft, wav)], tmp_path)
        assert bundle.entries[0].audio.read_bytes() == wav.read_bytes()

    def test_audio_properties_and_label_travel(self, full_draft, wav, tmp_path):
        bundle = round_trip([entry_for(full_draft, wav)], tmp_path)
        entry = bundle.entries[0]
        assert entry.audio_info.sample_rate == 32_000
        assert entry.audio_info.channels == 1
        assert entry.biotimbre_label == "Guyalna bonaerensis"

    def test_provenance_travels(self, full_draft, wav, tmp_path):
        bundle = round_trip(
            [entry_for(full_draft, wav)],
            tmp_path,
            source_library=SourceLibrary("lib-uuid", "Kaue's cicadas"),
        )
        provenance = bundle.entries[0].provenance
        assert provenance.library_uuid == "lib-uuid"
        assert provenance.library_name == "Kaue's cicadas"
        assert provenance.item_id == 12
        assert provenance.exported_at == bundle.created_at

    def test_embeddings_survive_exactly(self, full_draft, wav, tmp_path):
        original = rows(3)
        bundle = round_trip(
            [entry_for(full_draft, wav, embeddings={"perch_v2": original})],
            tmp_path,
            models={"perch_v2": FINGERPRINT},
        )
        restored = bundle.entries[0].embeddings["perch_v2"]
        assert len(restored) == 3
        for (start, end, vector), (o_start, o_end, o_vector) in zip(
            restored, original
        ):
            assert (start, end) == (o_start, o_end)
            # Bit-identical, not merely close: a lossy vector is a wrong vector.
            assert np.array_equal(vector, o_vector)
            assert vector.dtype == np.float32

    def test_model_fingerprint_travels(self, full_draft, wav, tmp_path):
        bundle = round_trip(
            [entry_for(full_draft, wav, embeddings={"perch_v2": rows(2)})],
            tmp_path,
            models={"perch_v2": FINGERPRINT},
        )
        assert bundle.models["perch_v2"] == FINGERPRINT
        assert bundle.models["perch_v2"].matches(FINGERPRINT)

    def test_window_counts_are_reported(self, full_draft, wav, tmp_path):
        bundle = round_trip(
            [entry_for(full_draft, wav, embeddings={"perch_v2": rows(4)})],
            tmp_path,
            models={"perch_v2": FINGERPRINT},
        )
        assert bundle.entries[0].window_counts == {"perch_v2": 4}

    def test_works_without_embeddings(self, full_draft, wav, tmp_path):
        bundle = round_trip([entry_for(full_draft, wav)], tmp_path)
        assert bundle.entries[0].embeddings == {}
        assert bundle.models == {}

    def test_carries_several_items_in_order(self, full_draft, wav, tmp_path):
        second = ItemDraft(
            date="2021-06-02",
            time="18:00:00",
            location=Location(1.0, 2.0),
            biotimbre_id=generate_biotimbre_id(),
            taxa="orthoptera",
        ).validated()
        bundle = round_trip(
            [entry_for(full_draft, wav), entry_for(second, wav, source_item_id=13)],
            tmp_path,
        )
        assert [e.draft.date for e in bundle.entries] == ["2020-01-14", "2021-06-02"]
        assert len(bundle.biotimbre_ids) == 2

    def test_entries_get_separate_audio_files(self, full_draft, wav, tmp_path):
        bundle = round_trip(
            [entry_for(full_draft, wav), entry_for(full_draft, wav)], tmp_path
        )
        first, second = bundle.entries
        assert first.audio != second.audio
        assert first.audio.read_bytes() == second.audio.read_bytes()

    def test_writes_and_reads_a_stream(self, full_draft, wav, tmp_path):
        # The interface builds a download in memory; no temporary file.
        buffer = io.BytesIO()
        write_bundle(buffer, [entry_for(full_draft, wav)])
        buffer.seek(0)
        bundle = read_bundle(buffer, into=tmp_path / "unpacked")
        assert bundle.entries[0].draft == full_draft

    def test_declares_its_format_and_version(self, full_draft, wav, tmp_path):
        path = tmp_path / "share.btz"
        write_bundle(path, [entry_for(full_draft, wav)])
        manifest = json.loads(zipfile.ZipFile(path).read(MANIFEST_NAME))
        assert manifest["format"] == BUNDLE_FORMAT
        assert manifest["format_version"] == BUNDLE_VERSION


class TestArchiveLayout:
    def test_members_are_where_the_layout_says(self, full_draft, wav, tmp_path):
        path = tmp_path / "share.btz"
        write_bundle(
            path,
            [entry_for(full_draft, wav, embeddings={"perch_v2": rows(2)})],
            models={"perch_v2": FINGERPRINT},
        )
        assert set(zipfile.ZipFile(path).namelist()) == {
            "manifest.json",
            "README.txt",
            "items/0001/item.json",
            "items/0001/item.txt",
            "items/0001/audio.wav",
            "items/0001/embeddings/perch_v2.npz",
        }

    def test_carries_a_readable_sidecar(self, full_draft, wav, tmp_path):
        from biotimbre.io.sidecar import draft_from_fields, parse_sidecar

        path = tmp_path / "share.btz"
        write_bundle(path, [entry_for(full_draft, wav)])
        text = zipfile.ZipFile(path).read("items/0001/item.txt").decode("utf-8")
        assert draft_from_fields(parse_sidecar(text)) == full_draft

    def test_readme_explains_the_file(self, full_draft, wav, tmp_path):
        path = tmp_path / "share.btz"
        write_bundle(path, [entry_for(full_draft, wav)])
        readme = zipfile.ZipFile(path).read("README.txt").decode("utf-8")
        assert "biotimbre import-bundle" in readme


class TestWriteRefusals:
    def test_refuses_an_item_with_no_biotimbre(self, wav, tmp_path):
        draft = ItemDraft(
            date="2020-01-14", time="09:40:21", location=Location(0.0, 0.0)
        ).validated()
        with pytest.raises(ValidationError, match="biotimbre_id"):
            write_bundle(tmp_path / "x.btz", [entry_for(draft, wav)])

    def test_refuses_missing_audio(self, full_draft, wav, tmp_path):
        entry = entry_for(full_draft, wav)
        with pytest.raises(ValidationError, match="No such audio file"):
            write_bundle(
                tmp_path / "x.btz",
                [replace(entry, audio=tmp_path / "gone.wav")],
            )

    def test_refuses_vectors_without_a_fingerprint(self, full_draft, wav, tmp_path):
        # Shipping numbers nobody can prove comparable is the one thing this
        # format exists to prevent.
        with pytest.raises(ValidationError, match="fingerprint"):
            write_bundle(
                tmp_path / "x.btz",
                [entry_for(full_draft, wav, embeddings={"perch_v2": rows(2)})],
            )


class TestFingerprintComparison:
    def test_differences_names_the_fields(self):
        other = ModelFingerprint(
            **{**FINGERPRINT.__dict__, "window_overlap": 0.0, "embedding_dim": 16}
        )
        assert not FINGERPRINT.matches(other)
        assert set(FINGERPRINT.differences(other)) == {
            "window_overlap",
            "embedding_dim",
        }

    def test_identical_fingerprints_have_no_differences(self):
        assert FINGERPRINT.differences(FINGERPRINT) == ()

    def test_same_id_is_not_enough(self):
        # A library that re-registered perch_v2 at another overlap produces
        # vectors that are not interchangeable with these.
        other = ModelFingerprint(**{**FINGERPRINT.__dict__, "window_overlap": 0.0})
        assert other.model_id == FINGERPRINT.model_id
        assert not FINGERPRINT.matches(other)


def build_archive(path: Path, members: dict[str, bytes | str]) -> Path:
    """Write an archive member by member, for bundles no writer would produce."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return path


def manifest_for(items: list[str], **overrides) -> str:
    base = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "created_at": "2026-01-01T00:00:00+00:00",
        "source_library": {"uuid": None, "name": None},
        "models": {},
        "items": [{"path": path} for path in items],
    }
    return json.dumps({**base, **overrides})


def item_body(**overrides) -> str:
    base = {
        "biotimbre_id": generate_biotimbre_id(),
        "taxa": ["cicadidae"],
        "date": "2020-01-14",
        "time": "09:40:21",
        "location": {"latitude": -20.0, "longitude": -46.0},
        "sample_window": None,
        "device": None,
        "microphone": None,
        "recorded_by": None,
        "filtered_band": None,
        "filter_type": None,
        "filter_order": None,
        "temperature": None,
        "humidity": None,
        "speed": 1.0,
        "link_to_full_recording": None,
        "notes": None,
        "biotimbre_label": None,
        "source_item_id": 1,
        "audio": {
            "filename": AUDIO_NAME,
            "sha256": "0" * 64,
            "duration_s": 1.0,
            "sample_rate": 32_000,
            "channels": 1,
        },
        "embeddings": {},
    }
    base.update(overrides)
    return json.dumps(base)


class TestHostileInput:
    """A bundle arrives from another person, so none of it is trusted."""

    def test_rejects_a_non_zip(self, tmp_path):
        path = tmp_path / "x.btz"
        path.write_bytes(b"not a zip at all")
        with pytest.raises(ValidationError, match="Not a readable bundle"):
            read_bundle(path, into=tmp_path / "out")

    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(ValidationError, match="No such bundle"):
            read_bundle(tmp_path / "absent.btz", into=tmp_path / "out")

    def test_rejects_a_zip_that_is_not_a_bundle(self, tmp_path):
        build_archive(tmp_path / "x.btz", {"notes.txt": "hello"})
        with pytest.raises(ValidationError, match="not a biotimbre bundle"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_another_format(self, tmp_path):
        build_archive(
            tmp_path / "x.btz", {MANIFEST_NAME: manifest_for([], format="something")}
        )
        with pytest.raises(ValidationError, match="declares format"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_a_newer_format_version(self, tmp_path):
        build_archive(
            tmp_path / "x.btz",
            {MANIFEST_NAME: manifest_for([], format_version=BUNDLE_VERSION + 1)},
        )
        with pytest.raises(ValidationError, match="Update to open it"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_an_entry_path_outside_items(self, tmp_path):
        build_archive(
            tmp_path / "x.btz", {MANIFEST_NAME: manifest_for(["../../etc"])}
        )
        with pytest.raises(ValidationError, match="unusable path"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_an_audio_path_escaping_its_entry(self, tmp_path):
        build_archive(
            tmp_path / "x.btz",
            {
                MANIFEST_NAME: manifest_for(["items/0001"]),
                "items/0001/item.json": item_body(
                    audio={
                        "filename": "../../../../.ssh/authorized_keys",
                        "sha256": "0" * 64,
                        "duration_s": 1.0,
                        "sample_rate": 32_000,
                        "channels": 1,
                    }
                ),
            },
        )
        with pytest.raises(ValidationError, match="outside its own folder"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_an_embeddings_path_escaping_its_entry(self, tmp_path):
        build_archive(
            tmp_path / "x.btz",
            {
                MANIFEST_NAME: manifest_for(
                    ["items/0001"], models={"perch_v2": FINGERPRINT.to_json()}
                ),
                "items/0001/item.json": item_body(
                    embeddings={"perch_v2": "../../../evil.npz"}
                ),
            },
        )
        with pytest.raises(ValidationError, match="outside its own folder"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_a_model_id_that_is_a_path(self, tmp_path):
        build_archive(
            tmp_path / "x.btz",
            {
                MANIFEST_NAME: manifest_for(["items/0001"]),
                "items/0001/item.json": item_body(embeddings={"../../evil": "x.npz"}),
            },
        )
        with pytest.raises(ValidationError, match="not a usable model id"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_tampered_audio(self, full_draft, wav, tmp_path):
        path = tmp_path / "share.btz"
        write_bundle(path, [entry_for(full_draft, wav)])

        # Repack with the audio swapped, leaving the declared digest in place.
        original = zipfile.ZipFile(path)
        members = {
            name: (
                b"tampered"
                if name.endswith(AUDIO_NAME)
                else original.read(name)
            )
            for name in original.namelist()
        }
        original.close()
        build_archive(path, members)

        with pytest.raises(ValidationError, match="damaged or was altered"):
            read_bundle(path, into=tmp_path / "out")

    def test_does_not_leave_tampered_audio_on_disk(self, full_draft, wav, tmp_path):
        path = tmp_path / "share.btz"
        build_archive(
            path,
            {
                MANIFEST_NAME: manifest_for(["items/0001"]),
                "items/0001/item.json": item_body(),
                "items/0001/audio.wav": b"tampered",
            },
        )
        with pytest.raises(ValidationError):
            read_bundle(path, into=tmp_path / "out")
        assert not list((tmp_path / "out").rglob("*.wav"))

    def test_rejects_an_item_without_a_biotimbre(self, tmp_path):
        build_archive(
            tmp_path / "x.btz",
            {
                MANIFEST_NAME: manifest_for(["items/0001"]),
                "items/0001/item.json": item_body(biotimbre_id=None),
            },
        )
        with pytest.raises(ValidationError, match="no biotimbre_id"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_a_missing_item_json(self, tmp_path):
        build_archive(tmp_path / "x.btz", {MANIFEST_NAME: manifest_for(["items/0001"])})
        with pytest.raises(ValidationError, match="no item.json"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_vectors_with_no_fingerprint(self, tmp_path):
        build_archive(
            tmp_path / "x.btz",
            {
                MANIFEST_NAME: manifest_for(["items/0001"]),
                "items/0001/item.json": item_body(
                    embeddings={"perch_v2": "embeddings/perch_v2.npz"}
                ),
            },
        )
        with pytest.raises(ValidationError, match="no fingerprint"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_vectors_of_the_wrong_width(self, full_draft, wav, tmp_path):
        wide = ModelFingerprint(**{**FINGERPRINT.__dict__, "embedding_dim": 1536})
        path = tmp_path / "share.btz"
        write_bundle(
            path,
            [entry_for(full_draft, wav, embeddings={"perch_v2": rows(2, dim=8)})],
            models={"perch_v2": wide},
        )
        with pytest.raises(ValidationError, match="fingerprinted at 1536"):
            read_bundle(path, into=tmp_path / "out")

    def test_rejects_a_compression_bomb(self, tmp_path):
        path = tmp_path / "x.btz"
        build_archive(
            path,
            {
                MANIFEST_NAME: manifest_for([]),
                "items/0001/audio.wav": b"\0" * (1024 * 1024),
            },
        )
        with pytest.raises(ValidationError, match="not a bundle of recordings"):
            read_bundle(path, into=tmp_path / "out")

    def test_rejects_unreadable_json(self, tmp_path):
        build_archive(tmp_path / "x.btz", {MANIFEST_NAME: "{ not json"})
        with pytest.raises(ValidationError, match="not readable JSON"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_rejects_a_duplicate_entry_path(self, tmp_path):
        build_archive(
            tmp_path / "x.btz",
            {MANIFEST_NAME: manifest_for(["items/0001", "items/0001"])},
        )
        with pytest.raises(ValidationError, match="same item path twice"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")

    def test_validates_items_as_strictly_as_a_form(self, tmp_path):
        build_archive(
            tmp_path / "x.btz",
            {
                MANIFEST_NAME: manifest_for(["items/0001"]),
                "items/0001/item.json": item_body(speed=1e6),
            },
        )
        with pytest.raises(ValidationError, match="[Ss]peed"):
            read_bundle(tmp_path / "x.btz", into=tmp_path / "out")
