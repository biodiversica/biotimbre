"""Scoring, reduction, fusion and filters — against known-correct answers."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import ItemDraft, Location, SampleWindow
from biotimbre.search.filters import (
    SearchFilters,
    haversine_km,
    select_candidates,
)
from biotimbre.search.vector_search import (
    FusedScore,
    WindowMatch,
    fuse,
    normalize_rows,
    reduce_to_items,
    score_matrix,
    top_k,
)
from conftest import add_item, unit_vectors


class TestScoreMatrix:
    def test_identical_vectors_have_cosine_one(self):
        v = unit_vectors(1, 8, seed=1)
        assert score_matrix(v, v, "cosine")[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_have_cosine_zero(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        assert score_matrix(a, b, "cosine")[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_have_cosine_minus_one(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        assert score_matrix(a, b, "cosine")[0, 0] == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_ignores_magnitude(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[7.0, 0.0]], dtype=np.float32)
        assert score_matrix(a, b, "cosine")[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_dot_respects_magnitude(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[7.0, 0.0]], dtype=np.float32)
        assert score_matrix(a, b, "dot")[0, 0] == pytest.approx(7.0)

    def test_euclidean_is_negated_so_higher_is_better(self):
        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        # Distance 5 becomes score -5, keeping ranking logic metric-agnostic.
        assert score_matrix(a, b, "euclidean")[0, 0] == pytest.approx(-5.0)

    def test_shape_is_query_by_library(self):
        assert score_matrix(unit_vectors(3, 16), unit_vectors(5, 16)).shape == (3, 5)

    def test_zero_vector_does_not_produce_nan(self):
        zero = np.zeros((1, 4), dtype=np.float32)
        out = score_matrix(zero, unit_vectors(2, 4), "cosine")
        assert np.isfinite(out).all()

    def test_rejects_mismatched_widths(self):
        # Almost always a sign that vectors from two models got mixed.
        with pytest.raises(ValidationError, match="widths disagree"):
            score_matrix(unit_vectors(1, 8), unit_vectors(1, 16))

    def test_rejects_unknown_metric(self):
        with pytest.raises(ValidationError, match="Unknown metric"):
            score_matrix(unit_vectors(1, 4), unit_vectors(1, 4), "manhattan")


class TestNormalizeRows:
    def test_produces_unit_norms(self):
        out = normalize_rows(np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_zero_row_stays_zero(self):
        out = normalize_rows(np.zeros((1, 3), dtype=np.float32))
        np.testing.assert_array_equal(out, 0.0)


class TestReduceToItems:
    def test_picks_the_best_window_pair(self):
        # Item 1 window 1 is the winner at 0.9; the reduction must report which
        # pair won, not just the score.
        scores = np.array([[0.1, 0.9], [0.3, 0.2]], dtype=np.float32)
        keys = [(1, 0, 0.0, 3.0), (1, 1, 1.5, 4.5)]
        best = reduce_to_items(scores, keys)
        assert set(best) == {1}
        assert best[1].score == pytest.approx(0.9)
        assert best[1].query_window == 0
        assert best[1].item_window == 1
        assert (best[1].start_time, best[1].end_time) == (1.5, 4.5)

    def test_separates_items(self):
        scores = np.array([[0.2, 0.8]], dtype=np.float32)
        keys = [(1, 0, 0.0, 3.0), (2, 0, 0.0, 3.0)]
        best = reduce_to_items(scores, keys)
        assert best[1].score == pytest.approx(0.2)
        assert best[2].score == pytest.approx(0.8)

    def test_max_lets_a_long_query_match_a_short_item(self):
        # One query window out of four matches strongly. Max keeps it; a mean
        # would dilute it to ~0.3 and lose the item.
        scores = np.array([[0.1], [0.1], [0.95], [0.1]], dtype=np.float32)
        keys = [(1, 0, 0.0, 3.0)]
        best = reduce_to_items(scores, keys)
        assert best[1].score == pytest.approx(0.95)
        assert best[1].query_window == 2

    def test_accumulates_across_batches(self):
        # An item's windows can straddle a batch boundary during streaming.
        keys_a = [(1, 0, 0.0, 3.0)]
        keys_b = [(1, 1, 1.5, 4.5)]
        best = reduce_to_items(np.array([[0.4]], np.float32), keys_a)
        best = reduce_to_items(np.array([[0.7]], np.float32), keys_b, into=best)
        assert best[1].score == pytest.approx(0.7)
        assert best[1].item_window == 1

    def test_batch_order_does_not_change_the_result(self):
        keys_a = [(1, 0, 0.0, 3.0)]
        keys_b = [(1, 1, 1.5, 4.5)]
        forward = reduce_to_items(np.array([[0.4]], np.float32), keys_a)
        forward = reduce_to_items(np.array([[0.7]], np.float32), keys_b, into=forward)
        reverse = reduce_to_items(np.array([[0.7]], np.float32), keys_b)
        reverse = reduce_to_items(np.array([[0.4]], np.float32), keys_a, into=reverse)
        assert forward[1].score == reverse[1].score
        assert forward[1].item_window == reverse[1].item_window

    def test_handles_negative_scores(self):
        # With euclidean every score is negative, so a zero-initialised best
        # would wrongly beat every real match.
        scores = np.array([[-5.0, -2.0]], dtype=np.float32)
        keys = [(1, 0, 0.0, 3.0), (1, 1, 1.5, 4.5)]
        best = reduce_to_items(scores, keys)
        assert best[1].score == pytest.approx(-2.0)
        assert best[1].item_window == 1

    def test_rejects_key_count_mismatch(self):
        with pytest.raises(ValidationError, match="keys"):
            reduce_to_items(np.zeros((1, 3), np.float32), [(1, 0, 0.0, 3.0)])


def _match(score: float, window: int = 0) -> WindowMatch:
    return WindowMatch(
        score=score, query_window=0, item_window=window,
        start_time=0.0, end_time=3.0,
    )


class TestFuse:
    def test_max_takes_the_better_model(self):
        result = fuse(
            {
                "perch_v2": {1: _match(0.5), 2: _match(0.9)},
                "birdnet_v2.4": {1: _match(0.8), 2: _match(0.2)},
            },
            "max",
        )
        by_id = {entry.item_id: entry for entry in result}
        assert by_id[1].score == pytest.approx(0.8)
        assert by_id[2].score == pytest.approx(0.9)

    def test_mean_rewards_agreement(self):
        # Item 1 has both models at 0.7; item 2 has one at 0.9 and one at 0.1.
        # Max would rank item 2 first; mean ranks the agreed match first.
        result = fuse(
            {
                "perch_v2": {1: _match(0.7), 2: _match(0.9)},
                "birdnet_v2.4": {1: _match(0.7), 2: _match(0.1)},
            },
            "mean",
        )
        assert result[0].item_id == 1
        assert result[0].score == pytest.approx(0.7)
        assert result[1].score == pytest.approx(0.5)

    def test_ranks_descending(self):
        result = fuse({"m": {1: _match(0.2), 2: _match(0.9), 3: _match(0.5)}}, "max")
        assert [e.item_id for e in result] == [2, 3, 1]

    def test_ties_break_on_item_id_for_determinism(self):
        result = fuse({"m": {7: _match(0.5), 3: _match(0.5), 5: _match(0.5)}}, "max")
        assert [e.item_id for e in result] == [3, 5, 7]

    def test_missing_model_is_not_scored_as_zero(self):
        # Item 2 has no BirdNET embeddings yet (awaiting backfill). Treating that
        # as 0.0 would halve its mean and unfairly bury it.
        result = fuse(
            {
                "perch_v2": {1: _match(0.6), 2: _match(0.8)},
                "birdnet_v2.4": {1: _match(0.6)},
            },
            "mean",
        )
        by_id = {entry.item_id: entry for entry in result}
        assert by_id[2].score == pytest.approx(0.8)
        assert by_id[2].per_model == {"perch_v2": 0.8}
        assert result[0].item_id == 2

    def test_per_model_breakdown_is_retained(self):
        result = fuse(
            {"perch_v2": {1: _match(0.5)}, "birdnet_v2.4": {1: _match(0.8)}}, "max"
        )
        assert result[0].per_model == {"perch_v2": 0.5, "birdnet_v2.4": 0.8}
        assert result[0].best_model == "birdnet_v2.4"

    def test_winning_window_is_retained_per_model(self):
        result = fuse(
            {
                "perch_v2": {1: _match(0.5, window=2)},
                "birdnet_v2.4": {1: _match(0.8, window=7)},
            },
            "max",
        )
        assert result[0].matches["birdnet_v2.4"].item_window == 7

    def test_single_model_selection(self):
        result = fuse(
            {"perch_v2": {1: _match(0.5), 2: _match(0.9)},
             "birdnet_v2.4": {1: _match(0.99)}},
            "perch_v2",
        )
        # Ranked by Perch alone, so item 2 wins and BirdNET's 0.99 is ignored.
        assert [e.item_id for e in result] == [2, 1]
        assert all(set(e.per_model) == {"perch_v2"} for e in result)

    def test_empty_input_gives_empty_output(self):
        assert fuse({}, "max") == []

    def test_rejects_unknown_fusion(self):
        with pytest.raises(ValidationError, match="Unknown fusion"):
            fuse({"m": {1: _match(0.5)}}, "median")


class TestTopK:
    def _fused(self, scores: list[float]) -> list[FusedScore]:
        return [
            FusedScore(item_id=i, score=s) for i, s in enumerate(sorted(scores, reverse=True))
        ]

    def test_truncates(self):
        assert len(top_k(self._fused([0.9, 0.8, 0.7, 0.6]), 2)) == 2

    def test_returns_all_when_k_exceeds_available(self):
        assert len(top_k(self._fused([0.9, 0.8]), 10)) == 2

    def test_applies_score_floor(self):
        result = top_k(self._fused([0.9, 0.5, 0.2]), 10, min_score=0.4)
        assert [round(e.score, 1) for e in result] == [0.9, 0.5]

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValidationError, match="at least 1"):
            top_k(self._fused([0.9]), 0)


class TestHaversine:
    def test_zero_distance_for_same_point(self):
        assert haversine_km(-26.9, -48.7, -26.9, -48.7) == pytest.approx(0.0)

    def test_one_degree_of_latitude_is_about_111_km(self):
        assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.1)

    def test_longitude_degrees_shrink_with_latitude(self):
        at_equator = haversine_km(0.0, 0.0, 0.0, 1.0)
        at_sixty = haversine_km(60.0, 0.0, 60.0, 1.0)
        assert at_sixty == pytest.approx(at_equator * 0.5, rel=0.01)

    def test_matches_a_known_distance(self):
        # Joinville to Florianopolis, roughly 145 km.
        km = haversine_km(-26.30, -48.85, -27.60, -48.55)
        assert km == pytest.approx(146.0, abs=3.0)


class TestSearchFilters:
    def test_empty_filters_match_everything(self):
        assert SearchFilters().is_empty is True

    def test_normalises_taxa_and_biotimbre_ids(self):
        uid = "3FAE2CA1-1B2A-4B8A-9C3E-1A2B3C4D5E6F"
        f = SearchFilters(taxa=["Insecta", "insecta"], biotimbre_ids=[uid])
        assert f.taxa == ("insecta",)
        assert f.biotimbre_ids == (uid.lower(),)

    def test_canonicalises_dates(self):
        assert SearchFilters(date_from="2020-1-5").date_from == "2020-01-05"

    def test_rejects_inverted_date_range(self):
        with pytest.raises(ValidationError, match="after"):
            SearchFilters(date_from="2021-01-01", date_to="2020-01-01")

    def test_rejects_partial_location(self):
        with pytest.raises(ValidationError, match="together"):
            SearchFilters(latitude=-26.0, longitude=-48.0)

    def test_rejects_non_positive_radius(self):
        with pytest.raises(ValidationError, match="positive"):
            SearchFilters(latitude=-26.0, longitude=-48.0, radius_km=0.0)


class TestSelectCandidates:
    def _item(self, repo, sha, **overrides):
        fields = {
            "date": "2020-01-14",
            "time": "09:40:21",
            "location": Location(-26.0, -48.0),
            "sample_window": SampleWindow(0.0, 5.0),
        }
        fields.update(overrides)
        return add_item(repo, ItemDraft(**fields).validated(), sha256=sha)

    def test_no_filter_returns_none_not_a_list(self):
        # None means "stream everything"; [] means "a filter matched nothing".
        # Conflating them would score the whole library for a filter that
        # excluded every item.
        candidates, _ = select_candidates(None, SearchFilters())
        assert candidates is None

    def test_filter_matching_nothing_returns_empty_list(self, repo):
        self._item(repo, "a" * 64, taxa="cicadidae")
        candidates, _ = select_candidates(
            repo.connection, SearchFilters(taxa=["aves"])
        )
        assert candidates == []

    def test_taxa_any_match(self, repo):
        cicada = self._item(repo, "a" * 64, taxa="cicadidae, insecta")
        self._item(repo, "b" * 64, taxa="aves")
        candidates, _ = select_candidates(
            repo.connection, SearchFilters(taxa=["cicadidae", "orthoptera"])
        )
        assert candidates == [cicada]

    def test_taxa_all_match(self, repo):
        both = self._item(repo, "a" * 64, taxa="cicadidae, insecta")
        self._item(repo, "b" * 64, taxa="cicadidae")
        candidates, _ = select_candidates(
            repo.connection,
            SearchFilters(taxa=["cicadidae", "insecta"], taxa_match_all=True),
        )
        assert candidates == [both]

    def test_date_range_is_inclusive(self, repo):
        early = self._item(repo, "a" * 64, date="2020-01-01")
        mid = self._item(repo, "b" * 64, date="2020-06-15")
        late = self._item(repo, "c" * 64, date="2021-01-01")
        candidates, _ = select_candidates(
            repo.connection,
            SearchFilters(date_from="2020-01-01", date_to="2020-06-15"),
        )
        assert candidates == [early, mid]
        assert late not in candidates

    def test_location_radius_with_distances(self, repo):
        near = self._item(repo, "a" * 64, location=Location(-26.05, -48.05))
        far = self._item(repo, "b" * 64, location=Location(-30.0, -50.0))
        candidates, distances = select_candidates(
            repo.connection,
            SearchFilters(latitude=-26.0, longitude=-48.0, radius_km=50.0),
        )
        assert candidates == [near]
        assert far not in distances
        assert 0.0 < distances[near] < 50.0

    def test_bounding_box_corners_are_removed_by_exact_distance(self, repo):
        # A point just outside the circle but inside its bounding box must be
        # dropped by the haversine refinement, not returned.
        corner = self._item(repo, "a" * 64, location=Location(-26.6, -48.6))
        candidates, _ = select_candidates(
            repo.connection,
            SearchFilters(latitude=-26.0, longitude=-48.0, radius_km=70.0),
        )
        assert corner not in candidates

    def test_device_substring_is_case_insensitive(self, repo):
        moth = self._item(repo, "a" * 64, device="AudioMoth 1.2.0")
        self._item(repo, "b" * 64, device="Zoom H5")
        candidates, _ = select_candidates(
            repo.connection, SearchFilters(device="audiomoth")
        )
        assert candidates == [moth]

    def test_speed_filter(self, repo):
        native = self._item(repo, "a" * 64, speed=1.0)
        slowed = self._item(repo, "b" * 64, speed=0.25)
        candidates, _ = select_candidates(
            repo.connection, SearchFilters(speeds=(0.25,))
        )
        assert candidates == [slowed]
        assert native not in candidates

    def test_exclude_item_ids(self, repo):
        first = self._item(repo, "a" * 64)
        second = self._item(repo, "b" * 64)
        candidates, _ = select_candidates(
            repo.connection, SearchFilters(exclude_item_ids=(first,))
        )
        assert candidates == [second]

    def test_filters_combine_conjunctively(self, repo):
        wanted = self._item(
            repo, "a" * 64, taxa="cicadidae", date="2020-06-01", device="AudioMoth"
        )
        self._item(repo, "b" * 64, taxa="cicadidae", date="2019-06-01", device="AudioMoth")
        self._item(repo, "c" * 64, taxa="aves", date="2020-06-01", device="AudioMoth")
        candidates, _ = select_candidates(
            repo.connection,
            SearchFilters(
                taxa=["cicadidae"], date_from="2020-01-01", device="audiomoth"
            ),
        )
        assert candidates == [wanted]
