"""Repository tests: CRUD, ID allocation, cascade, embedding storage."""

from __future__ import annotations

import dataclasses
import uuid

import numpy as np
import pytest

from biotimbre.db.connection import SCHEMA_VERSION, initialize
from biotimbre.domain.errors import (
    BiotimbreError,
    NotFoundError,
    ValidationError,
)
from biotimbre.domain.item import ItemDraft, Location, SampleWindow
from conftest import add_item, unit_vectors

DIM = 32


@pytest.fixture
def registered(repo, fake_spec):
    """A repository with ``fake_model`` registered at width :data:`DIM`."""
    repo.register_model(fake_spec, DIM)
    return repo


class TestSchema:
    def test_initialize_is_idempotent(self, settings):
        first = initialize(settings.db_path)
        first.close()
        second = initialize(settings.db_path)
        row = second.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        assert int(row["value"]) == SCHEMA_VERSION
        second.close()

    def test_migrates_a_version_3_database(self, settings):
        # A version-3 items table is today's, with the sample window still
        # mandatory; building it from schema.sql keeps the two in step.
        import re
        import sqlite3

        from biotimbre.db import connection as connection_module

        schema = re.sub(
            r"sample_window_(start|end)(\s+)REAL,",
            r"sample_window_\1\2REAL NOT NULL,",
            connection_module._schema_sql(),
        )
        assert len(re.findall(r"sample_window_\w+\s+REAL NOT NULL,", schema)) == 2

        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        old_db = sqlite3.connect(settings.db_path)
        old_db.executescript(schema)
        old_db.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('schema_version', '3')"
        )
        old_db.execute(
            "INSERT INTO biotimbres (biotimbre_id, created_at) VALUES (?, ?)",
            ("11111111-2222-3333-4444-555555555555", "2020-01-01T00:00:00Z"),
        )
        old_db.execute(
            """
            INSERT INTO items (
                item_id, biotimbre_id, date, time, latitude, longitude,
                sample_window_start, sample_window_end, speed, audio_path,
                audio_sha256, duration_s, sample_rate, channels,
                created_at, updated_at
            ) VALUES (7, ?, '2020-01-01', '09:00:00', -26.0, -48.0, 5, 10, 1.0,
                      'audio/x/7.wav', 'a', 5.0, 32000, 1,
                      '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z')
            """,
            ("11111111-2222-3333-4444-555555555555",),
        )
        old_db.execute("INSERT INTO item_taxa (item_id, tag) VALUES (7, 'cicadidae')")
        old_db.commit()
        old_db.close()

        conn = initialize(settings.db_path)
        try:
            version = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            assert int(version["value"]) == SCHEMA_VERSION

            # Nothing lost: the row, its taxa, and the item's own ID.
            row = conn.execute("SELECT * FROM items WHERE item_id = 7").fetchone()
            assert (row["sample_window_start"], row["sample_window_end"]) == (5.0, 10.0)
            assert conn.execute("SELECT count(*) FROM item_taxa").fetchone()[0] == 1
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

            # The point of the migration: a window may now be absent.
            conn.execute(
                "UPDATE items SET sample_window_start = NULL, "
                "sample_window_end = NULL WHERE item_id = 7"
            )
            # The rebuilt table keeps its indexes and its autoincrement.
            indexes = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' "
                    "AND tbl_name = 'items' AND name LIKE 'idx%'"
                )
            }
            assert "idx_items_sha256" in indexes
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE '%migrating%'"
            ).fetchall() == []
        finally:
            conn.close()

    def test_refuses_a_version_it_cannot_migrate(self, settings):
        import sqlite3

        conn = initialize(settings.db_path)
        conn.execute("UPDATE schema_meta SET value = '2' WHERE key = 'schema_version'")
        conn.commit()
        conn.close()

        with pytest.raises(BiotimbreError, match="No automatic migration"):
            initialize(settings.db_path)

    def test_foreign_keys_are_enforced(self, conn):
        # Off by default in SQLite, and per-connection; without it the
        # ON DELETE CASCADE below is silently a no-op.
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


class TestBiotimbreAllocation:
    def test_allocates_distinct_valid_uuids(self, repo):
        first = repo.allocate_biotimbre_id()
        second = repo.allocate_biotimbre_id()
        assert first != second
        uuid.UUID(first)
        uuid.UUID(second)

    def test_ensure_biotimbre_is_idempotent(self, repo):
        uid = "3FAE2CA1-1B2A-4B8A-9C3E-1A2B3C4D5E6F"
        assert repo.ensure_biotimbre(uid) == uid.lower()
        assert repo.ensure_biotimbre(uid.lower()) == uid.lower()
        assert len(repo.list_biotimbres()) == 1

    def test_label_survives_ensure(self, repo):
        uid = str(uuid.uuid4())
        repo.ensure_biotimbre(uid, label="cicada sp.")
        repo.ensure_biotimbre(uid, label="ignored")
        assert repo.list_biotimbres()[0]["label"] == "cicada sp."

    def test_prune_keeps_labelled_but_empty(self, repo):
        empty = str(uuid.uuid4())
        kept = str(uuid.uuid4())
        repo.ensure_biotimbre(empty)
        repo.ensure_biotimbre(kept, label="worth keeping")
        assert repo.prune_empty_biotimbres() == 1
        assert [b["biotimbre_id"] for b in repo.list_biotimbres()] == [kept]


class TestItemCrud:
    def test_create_allocates_biotimbre(self, repo, draft):
        item = repo.create_item(
            draft,
            audio_path="audio/0000/1.wav",
            audio_sha256="a" * 64,
            duration_s=5.0,
            sample_rate=32_000,
            channels=1,
        )
        assert item.item_id == 1
        uuid.UUID(item.biotimbre_id)
        assert item.duration_s == 5.0

    def test_two_items_can_share_a_biotimbre(self, repo, draft):
        first = add_item(repo, draft, sha256="a" * 64)
        shared = repo.get_item(first).biotimbre_id
        joined = dataclasses.replace(draft, biotimbre_id=shared).validated()
        second = add_item(repo, joined, sha256="b" * 64)
        assert repo.get_item(second).biotimbre_id == shared
        assert first != second

    def test_item_ids_are_not_reused_after_delete(self, repo, draft):
        first = add_item(repo, draft, sha256="a" * 64)
        repo.delete_item(first)
        second = add_item(repo, draft, sha256="b" * 64)
        assert second > first

    def test_get_missing_item_raises(self, repo):
        with pytest.raises(NotFoundError, match="999"):
            repo.get_item(999)

    def test_delete_returns_audio_path_for_caller_cleanup(self, repo, draft):
        item_id = add_item(repo, draft)
        path = repo.delete_item(item_id)
        # The repository never touches the filesystem; it hands back the path.
        assert path.endswith(".wav")

    def test_item_without_a_sample_window_round_trips(self, repo, draft):
        windowless = dataclasses.replace(draft, sample_window=None).validated()
        item_id = add_item(repo, windowless)
        assert repo.get_item(item_id).sample_window is None

    def test_update_can_clear_a_sample_window(self, repo, draft):
        item_id = add_item(repo, draft)
        cleared = dataclasses.replace(draft, sample_window=None).validated()
        repo.update_item(item_id, cleared)
        assert repo.get_item(item_id).sample_window is None

    def test_taxa_round_trip(self, repo):
        tagged = ItemDraft(
            date="2020-01-14",
            time="09:40:21",
            location=Location(-20.0, -46.0),
            sample_window=SampleWindow(5.0, 10.0),
            taxa="Insecta, orthoptera, INSECTA",
        ).validated()
        item_id = add_item(repo, tagged)
        assert repo.get_item(item_id).taxa == ("insecta", "orthoptera")

    def test_update_replaces_taxa_wholesale(self, repo, draft):
        item_id = add_item(repo, draft)
        with_tags = dataclasses.replace(draft, taxa="cicadidae").validated()
        repo.update_item(item_id, with_tags)
        assert repo.get_item(item_id).taxa == ("cicadidae",)
        cleared = dataclasses.replace(draft, taxa=()).validated()
        repo.update_item(item_id, cleared)
        assert repo.get_item(item_id).taxa == ()

    def test_update_preserves_biotimbre_when_draft_omits_it(self, repo, draft):
        item_id = add_item(repo, draft)
        original = repo.get_item(item_id).biotimbre_id
        # A draft with biotimbre_id=None means "unchanged" on update, never
        # "allocate a new one".
        edited = dataclasses.replace(draft, biotimbre_id=None, notes="edited").validated()
        updated, _ = repo.update_item(item_id, edited)
        assert updated.biotimbre_id == original
        assert updated.notes == "edited"

    def test_find_by_sha256_reports_duplicates(self, repo, draft):
        add_item(repo, draft, sha256="c" * 64)
        add_item(repo, draft, sha256="c" * 64)
        # Byte-identical audio is allowed (same sound, two biotimbres) but must
        # be discoverable so the UI can warn.
        assert len(repo.find_by_sha256("c" * 64)) == 2

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"duration_s": 0.0},
            {"duration_s": -1.0},
            {"sample_rate": 0},
        ],
    )
    def test_rejects_impossible_audio_properties(self, repo, draft, kwargs):
        args = {
            "audio_path": "audio/0000/1.wav",
            "audio_sha256": "d" * 64,
            "duration_s": 5.0,
            "sample_rate": 32_000,
            "channels": 1,
        }
        args.update(kwargs)
        with pytest.raises(ValidationError):
            repo.create_item(draft, **args)


class TestSpeedInvalidation:
    def test_changing_speed_deletes_embeddings(self, registered, draft):
        item_id = add_item(registered, draft)
        registered.set_embeddings(
            item_id, "fake_model", [(0.0, 5.0, unit_vectors(1, DIM)[0])]
        )
        assert registered.embedding_counts(item_id) == {"fake_model": 1}

        shifted = dataclasses.replace(draft, speed=0.25).validated()
        _, invalidated = registered.update_item(item_id, shifted)

        # Vectors computed at speed 1.0 describe different audio than the item
        # now claims, so they must not survive the change.
        assert invalidated is True
        assert registered.embedding_counts(item_id) == {}

    def test_editing_other_fields_keeps_embeddings(self, registered, draft):
        item_id = add_item(registered, draft)
        registered.set_embeddings(
            item_id, "fake_model", [(0.0, 5.0, unit_vectors(1, DIM)[0])]
        )
        edited = dataclasses.replace(draft, notes="a note", device="Audiomoth").validated()
        _, invalidated = registered.update_item(item_id, edited)
        assert invalidated is False
        assert registered.embedding_counts(item_id) == {"fake_model": 1}

    def test_distinct_speeds_drives_query_sweep(self, registered, draft):
        add_item(registered, draft, sha256="a" * 64)
        add_item(
            registered,
            dataclasses.replace(draft, speed=0.25).validated(),
            sha256="b" * 64,
        )
        add_item(
            registered,
            dataclasses.replace(draft, speed=0.25).validated(),
            sha256="c" * 64,
        )
        assert registered.distinct_speeds() == [0.25, 1.0]


class TestModelRegistration:
    def test_register_and_read_back(self, repo, fake_spec):
        repo.register_model(fake_spec, DIM)
        model = repo.get_model("fake_model")
        assert model["embedding_dim"] == DIM
        assert model["sample_rate"] == 32_000
        assert model["source"] == "hf://test/fake"

    def test_reregistering_same_dim_updates_parameters(self, repo, fake_spec):
        repo.register_model(fake_spec, DIM)
        changed = dataclasses.replace(fake_spec, window_overlap=0.0)
        repo.register_model(changed, DIM)
        assert repo.get_model("fake_model")["window_overlap"] == 0.0

    def test_reregistering_different_dim_is_refused(self, repo, fake_spec):
        repo.register_model(fake_spec, DIM)
        # Existing vectors are DIM wide; silently accepting a new width would
        # make them unreadable.
        with pytest.raises(ValidationError, match="already registered"):
            repo.register_model(fake_spec, DIM * 2)

    def test_rejects_non_positive_dim(self, repo, fake_spec):
        with pytest.raises(ValidationError, match="positive"):
            repo.register_model(fake_spec, 0)


class TestEmbeddingStorage:
    def test_round_trip_preserves_values_exactly(self, registered, draft):
        item_id = add_item(registered, draft)
        vectors = unit_vectors(3, DIM, seed=7)
        registered.set_embeddings(
            item_id,
            "fake_model",
            [(float(i), float(i) + 5.0, vectors[i]) for i in range(3)],
        )
        matrix, spans = registered.get_embedding_matrix(item_id, "fake_model")
        # float32 through a BLOB is a byte-exact round trip.
        np.testing.assert_array_equal(matrix, vectors)
        assert spans == [(0.0, 5.0), (1.0, 6.0), (2.0, 7.0)]

    def test_replaces_rather_than_appends(self, registered, draft):
        item_id = add_item(registered, draft)
        registered.set_embeddings(
            item_id, "fake_model", [(0.0, 5.0, v) for v in unit_vectors(4, DIM)]
        )
        # Re-running at a coarser overlap must not leave the old windows behind.
        registered.set_embeddings(
            item_id, "fake_model", [(0.0, 5.0, v) for v in unit_vectors(2, DIM)]
        )
        assert registered.embedding_counts(item_id) == {"fake_model": 2}

    def test_rejects_wrong_width(self, registered, draft):
        item_id = add_item(registered, draft)
        with pytest.raises(ValidationError, match="dimension"):
            registered.set_embeddings(
                item_id, "fake_model", [(0.0, 5.0, np.zeros(DIM + 1, np.float32))]
            )

    def test_rejects_unregistered_model(self, repo, draft):
        item_id = add_item(repo, draft)
        with pytest.raises(NotFoundError, match="not registered"):
            repo.set_embeddings(
                item_id, "ghost", [(0.0, 5.0, np.zeros(DIM, np.float32))]
            )

    def test_deleting_item_cascades_to_embeddings(self, registered, draft):
        item_id = add_item(registered, draft)
        registered.set_embeddings(
            item_id, "fake_model", [(0.0, 5.0, v) for v in unit_vectors(3, DIM)]
        )
        registered.delete_item(item_id)
        remaining = registered.connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE item_id = ?", (item_id,)
        ).fetchone()[0]
        assert remaining == 0

    def test_items_missing_embeddings_drives_backfill(self, registered, draft):
        has = add_item(registered, draft, sha256="a" * 64)
        missing = add_item(registered, draft, sha256="b" * 64)
        registered.set_embeddings(
            has, "fake_model", [(0.0, 5.0, unit_vectors(1, DIM)[0])]
        )
        assert registered.items_missing_embeddings("fake_model") == [missing]


class TestEmbeddingStreaming:
    def _populate(self, repo, draft, n_items: int, windows_each: int):
        ids = []
        for i in range(n_items):
            item_id = add_item(repo, draft, sha256=f"{i:064d}")
            repo.set_embeddings(
                item_id,
                "fake_model",
                [
                    (float(w), float(w) + 5.0, v)
                    for w, v in enumerate(unit_vectors(windows_each, DIM, seed=i))
                ],
            )
            ids.append(item_id)
        return ids

    def test_streams_all_rows_in_order(self, registered, draft):
        self._populate(registered, draft, n_items=4, windows_each=3)
        keys, matrices = [], []
        for batch_keys, matrix in registered.iter_embedding_batches(
            "fake_model", batch_size=5
        ):
            keys.extend(batch_keys)
            matrices.append(matrix)
        assert len(keys) == 12
        assert [(k[0], k[1]) for k in keys] == sorted((k[0], k[1]) for k in keys)
        assert np.vstack(matrices).shape == (12, DIM)

    def test_restricts_to_candidates(self, registered, draft):
        ids = self._populate(registered, draft, n_items=4, windows_each=2)
        wanted = [ids[0], ids[3]]
        got = [
            key[0]
            for batch_keys, _ in registered.iter_embedding_batches(
                "fake_model", item_ids=wanted
            )
            for key in batch_keys
        ]
        assert set(got) == set(wanted)
        assert len(got) == 4

    def test_empty_candidate_list_yields_nothing(self, registered, draft):
        self._populate(registered, draft, n_items=2, windows_each=2)
        # An empty list means "no candidate survived filtering", which is not the
        # same as None ("no filter"). Conflating them would score the whole
        # library for a filter that matched nothing.
        batches = list(
            registered.iter_embedding_batches("fake_model", item_ids=[])
        )
        assert batches == []

    def test_candidate_list_beyond_sqlite_parameter_limit(self, registered, draft):
        # A temp-table JOIN is used precisely so this does not hit the 999-host-
        # parameter cap that an IN (...) list would.
        ids = self._populate(registered, draft, n_items=3, windows_each=1)
        oversized = ids + list(range(10_000, 12_000))
        got = [
            key[0]
            for batch_keys, _ in registered.iter_embedding_batches(
                "fake_model", item_ids=oversized
            )
            for key in batch_keys
        ]
        assert set(got) == set(ids)
