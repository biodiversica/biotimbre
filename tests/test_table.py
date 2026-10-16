"""The library as CSV: one row per item, one column per stored value."""

from __future__ import annotations

import csv
import io

from biotimbre.domain.item import (
    FilteredBand,
    LibraryItem,
    Location,
    SampleWindow,
)
from biotimbre.io.table import items_csv, results_csv


def _item(item_id: int, **overrides) -> LibraryItem:
    values = dict(
        item_id=item_id,
        biotimbre_id=f"{item_id:032x}",
        date="2020-01-14",
        time="09:40:21",
        location=Location(-26.3, -48.85),
        sample_window=None,
        audio_path=f"audio/{item_id}.wav",
        audio_sha256=f"{item_id:064x}",
        duration_s=5.0,
        sample_rate=32_000,
        channels=1,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    values.update(overrides)
    return LibraryItem(**values)


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


class TestItemsCsv:
    def test_every_stored_value_has_a_column(self):
        text = items_csv([_item(1)], labels={}, embedding_counts={}, model_ids=())
        header = next(csv.reader(io.StringIO(text)))
        assert header[:3] == ["item_id", "biotimbre_id", "biotimbre_label"]
        # Compound values are split, so each column holds one plain value.
        for name in (
            "latitude", "longitude", "sample_window_start", "sample_window_end",
            "filtered_band_low", "filtered_band_high", "taxa", "device",
            "microphone", "recorded_by", "filter_type", "filter_order",
            "temperature", "humidity", "speed", "link_to_full_recording",
            "notes", "duration_s", "sample_rate", "channels", "audio_path",
            "audio_sha256", "created_at", "updated_at",
        ):
            assert name in header

    def test_one_row_per_item_with_its_values(self):
        item = _item(
            7,
            taxa=("cicadidae", "guyalna"),
            sample_window=SampleWindow(5.0, 10.0),
            filtered_band=FilteredBand(2000.0, None),
            device="Audiomoth",
            notes='Said "hello", then left.\nSecond line.',
        )
        rows = _rows(
            items_csv(
                [item],
                labels={item.biotimbre_id: "Guyalna song"},
                embedding_counts={},
                model_ids=(),
            )
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["item_id"] == "7"
        assert row["biotimbre_label"] == "Guyalna song"
        # Commas inside a cell are quoted, not taken for column breaks.
        assert row["taxa"] == "cicadidae,guyalna"
        assert (row["latitude"], row["longitude"]) == ("-26.3", "-48.85")
        assert (row["sample_window_start"], row["sample_window_end"]) == (
            "5.0",
            "10.0",
        )
        # An open side of the band is an empty cell, as is anything unset.
        assert (row["filtered_band_low"], row["filtered_band_high"]) == ("2000.0", "")
        assert row["microphone"] == ""
        assert row["temperature"] == ""
        # Quotes and line breaks survive the round trip.
        assert row["notes"] == 'Said "hello", then left.\nSecond line.'

    def test_each_model_gets_a_window_count_column(self):
        rows = _rows(
            items_csv(
                [_item(1), _item(2)],
                labels={},
                embedding_counts={1: {"perch_v2": 1, "birdnet_v2.4": 3}},
                model_ids=("birdnet_v2.4", "perch_v2"),
            )
        )
        assert rows[0]["embeddings_birdnet_v2.4"] == "3"
        assert rows[0]["embeddings_perch_v2"] == "1"
        # Never embedded with a model is a count of zero, not a blank.
        assert rows[1]["embeddings_perch_v2"] == "0"

    def test_an_empty_library_is_a_header_alone(self):
        text = items_csv([], labels={}, embedding_counts={}, model_ids=("perch_v2",))
        assert len(text.splitlines()) == 1
        assert text.rstrip().endswith("embeddings_perch_v2")


class TestResultsCsv:
    @staticmethod
    def _hit(item_id: int, **fields):
        from biotimbre.service import SearchHit

        return SearchHit(item=_item(item_id), **fields)

    def test_the_match_comes_first_in_rank_order(self):
        from biotimbre.search.vector_search import WindowMatch

        hits = [
            self._hit(
                9,
                score=0.91,
                per_model={"perch_v2": 0.91, "birdnet_v2.4": 0.80},
                matches={
                    "perch_v2": WindowMatch(0.91, 0, 0, 0.0, 5.0),
                    "birdnet_v2.4": WindowMatch(0.80, 0, 1, 1.5, 4.5),
                },
                distance_km=12.5,
                query_speed=0.5,
            ),
            self._hit(
                3,
                score=0.40,
                per_model={"perch_v2": 0.40},
                matches={"perch_v2": WindowMatch(0.40, 0, 0, 0.0, 6.0)},
            ),
        ]
        text = results_csv(
            hits, labels={}, embedding_counts={}, model_ids=("perch_v2",)
        )
        header = next(csv.reader(io.StringIO(text)))
        assert header[:9] == [
            "rank", "score", "best_model", "matched_start", "matched_end",
            "distance_km", "query_speed", "score_birdnet_v2.4", "score_perch_v2",
        ]
        # The item's own columns follow, as in the library export.
        assert header[9] == "item_id"

        rows = _rows(text)
        assert [row["item_id"] for row in rows] == ["9", "3"]
        assert [row["rank"] for row in rows] == ["1", "2"]
        first, second = rows
        assert first["score"] == "0.91"
        assert first["best_model"] == "perch_v2"
        assert first["distance_km"] == "12.5"
        assert first["query_speed"] == "0.5"
        # A model that did not score a hit is a blank, not a zero.
        assert second["score_birdnet_v2.4"] == ""
        # The matched window is clamped to the sample, as on screen.
        assert (second["matched_start"], second["matched_end"]) == ("0.0", "5.0")

    def test_a_metadata_search_leaves_the_scores_empty(self):
        rows = _rows(
            results_csv(
                [self._hit(4, score=0.0)],
                labels={},
                embedding_counts={},
                model_ids=(),
            )
        )
        row = rows[0]
        assert row["rank"] == "1"
        for name in ("score", "best_model", "matched_start", "query_speed"):
            assert row[name] == ""
