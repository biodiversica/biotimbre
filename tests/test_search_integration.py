"""Search over stored vectors: streaming, filtering and reduction together.

Vectors are constructed so the correct answer is known by hand, rather than
inferred from a model's behaviour. Needs no model files.
"""

from __future__ import annotations

import numpy as np
import pytest

from biotimbre.domain.item import ItemDraft, Location, SampleWindow
from biotimbre.search.filters import SearchFilters, select_candidates
from biotimbre.search.vector_search import fuse, reduce_to_items, score_matrix, top_k
from conftest import add_item

DIM = 8


def basis(index: int, dim: int = DIM) -> np.ndarray:
    """A unit vector along one axis, so cosine similarity is exactly known."""
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


def blend(a: int, b: int, weight: float, dim: int = DIM) -> np.ndarray:
    """A unit vector between two axes; cosine against ``basis(a)`` is known."""
    v = np.zeros(dim, dtype=np.float32)
    v[a] = np.cos(weight * np.pi / 2.0)
    v[b] = np.sin(weight * np.pi / 2.0)
    return v


def run_search(
    repo,
    query_per_model: dict[str, np.ndarray],
    filters: SearchFilters | None = None,
    *,
    fusion: str = "max",
    k: int = 10,
    batch_size: int = 4,
):
    """Full search path: filter, stream, score, reduce, fuse, truncate."""
    filters = filters or SearchFilters()
    candidates, _ = select_candidates(repo.connection, filters)

    per_model = {}
    for model_id, query in query_per_model.items():
        best: dict = {}
        for keys, matrix in repo.iter_embedding_batches(
            model_id, item_ids=candidates, batch_size=batch_size
        ):
            best = reduce_to_items(score_matrix(query, matrix, "cosine"), keys, into=best)
        if best:
            per_model[model_id] = best
    return top_k(fuse(per_model, fusion), k)


@pytest.fixture
def library(repo, fake_spec):
    """Three items with hand-chosen vectors, and a second model registered.

    Item A is exactly axis 0. Item B is 30% of the way toward axis 1.
    Item C is orthogonal to axis 0. So a query of ``basis(0)`` must rank A, B, C.
    """
    repo.register_model(fake_spec, DIM)
    other = type(fake_spec)(
        **{**fake_spec.__dict__, "model_id": "other_model"}
    )
    repo.register_model(other, DIM)

    def make(sha, **overrides):
        fields = {
            "date": "2020-06-15",
            "time": "09:40:21",
            "location": Location(-26.0, -48.0),
            "sample_window": SampleWindow(0.0, 5.0),
        }
        fields.update(overrides)
        return add_item(repo, ItemDraft(**fields).validated(), sha256=sha)

    ids = {
        "A": make("a" * 64, taxa="cicadidae"),
        "B": make("b" * 64, taxa="orthoptera", location=Location(-30.0, -52.0)),
        "C": make("c" * 64, taxa="aves", date="2019-01-01"),
    }
    vectors = {"A": basis(0), "B": blend(0, 1, 0.3), "C": basis(1)}
    for name, item_id in ids.items():
        repo.set_embeddings(item_id, "fake_model", [(0.0, 5.0, vectors[name])])
    return ids


class TestEndToEndScoring:
    def test_ranks_by_similarity(self, repo, library):
        results = run_search(repo, {"fake_model": basis(0).reshape(1, -1)})
        assert [r.item_id for r in results] == [
            library["A"], library["B"], library["C"]
        ]
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        assert results[2].score == pytest.approx(0.0, abs=1e-6)

    def test_exact_match_scores_one(self, repo, library):
        results = run_search(repo, {"fake_model": basis(0).reshape(1, -1)}, k=1)
        assert results[0].item_id == library["A"]
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_streaming_batch_size_does_not_change_results(self, repo, library):
        query = {"fake_model": basis(0).reshape(1, -1)}
        one_batch = run_search(repo, query, batch_size=100)
        many_batches = run_search(repo, query, batch_size=1)
        assert [r.item_id for r in one_batch] == [r.item_id for r in many_batches]
        assert [round(r.score, 5) for r in one_batch] == [
            round(r.score, 5) for r in many_batches
        ]

    def test_multi_window_query_takes_the_best_pair(self, repo, library):
        # Query window 1 matches item C exactly; window 0 matches nothing well.
        query = np.vstack([basis(5), basis(1)])
        results = run_search(repo, {"fake_model": query})
        best = {r.item_id: r for r in results}
        assert best[library["C"]].score == pytest.approx(1.0, abs=1e-5)
        assert best[library["C"]].matches["fake_model"].query_window == 1

    def test_reports_the_matched_window_span(self, repo, library):
        results = run_search(repo, {"fake_model": basis(0).reshape(1, -1)}, k=1)
        match = results[0].matches["fake_model"]
        # The span is what the player needs to mark the matched region.
        assert (match.start_time, match.end_time) == (0.0, 5.0)


class TestFilterChangesTopK:
    def test_taxa_filter_changes_the_winner(self, repo, library):
        unfiltered = run_search(repo, {"fake_model": basis(0).reshape(1, -1)}, k=1)
        assert unfiltered[0].item_id == library["A"]

        # Excluding A's taxon promotes B, which is the whole point of filtering
        # before scoring rather than after: the top-k is of the right population.
        filtered = run_search(
            repo,
            {"fake_model": basis(0).reshape(1, -1)},
            SearchFilters(taxa=["orthoptera", "aves"]),
            k=1,
        )
        assert filtered[0].item_id == library["B"]

    def test_location_filter_excludes_a_better_match(self, repo, library):
        # B is the second-best match but 500 km away, so a tight radius drops it.
        results = run_search(
            repo,
            {"fake_model": basis(0).reshape(1, -1)},
            SearchFilters(latitude=-26.0, longitude=-48.0, radius_km=50.0),
        )
        assert library["B"] not in [r.item_id for r in results]
        assert library["A"] in [r.item_id for r in results]

    def test_date_filter_excludes_older_item(self, repo, library):
        results = run_search(
            repo,
            {"fake_model": basis(0).reshape(1, -1)},
            SearchFilters(date_from="2020-01-01"),
        )
        assert library["C"] not in [r.item_id for r in results]

    def test_filter_matching_nothing_returns_no_results(self, repo, library):
        results = run_search(
            repo,
            {"fake_model": basis(0).reshape(1, -1)},
            SearchFilters(taxa=["mammalia"]),
        )
        # Must be empty, not the whole library: [] and None mean different things
        # to the streaming layer.
        assert results == []

    def test_excluding_the_query_item_itself(self, repo, library):
        # Searching with an item already in the library should be able to omit it,
        # or it trivially wins its own search at score 1.0.
        results = run_search(
            repo,
            {"fake_model": basis(0).reshape(1, -1)},
            SearchFilters(exclude_item_ids=(library["A"],)),
            k=1,
        )
        assert results[0].item_id == library["B"]


class TestModelDisagreement:
    def test_max_fusion_surfaces_either_model(self, repo, library):
        # Under the second model, C is the exact match while A is orthogonal.
        repo.set_embeddings(library["A"], "other_model", [(0.0, 5.0, basis(1))])
        repo.set_embeddings(library["C"], "other_model", [(0.0, 5.0, basis(0))])

        results = run_search(
            repo,
            {
                "fake_model": basis(0).reshape(1, -1),
                "other_model": basis(0).reshape(1, -1),
            },
            fusion="max",
        )
        by_id = {r.item_id: r for r in results}
        # A wins on fake_model, C wins on other_model; both score ~1.0.
        assert by_id[library["A"]].score == pytest.approx(1.0, abs=1e-5)
        assert by_id[library["C"]].score == pytest.approx(1.0, abs=1e-5)
        assert by_id[library["A"]].best_model == "fake_model"
        assert by_id[library["C"]].best_model == "other_model"

    def test_mean_fusion_penalises_disagreement(self, repo, library):
        repo.set_embeddings(library["A"], "other_model", [(0.0, 5.0, basis(1))])
        repo.set_embeddings(library["C"], "other_model", [(0.0, 5.0, basis(1))])

        results = run_search(
            repo,
            {
                "fake_model": basis(0).reshape(1, -1),
                "other_model": basis(0).reshape(1, -1),
            },
            fusion="mean",
        )
        by_id = {r.item_id: r for r in results}
        # A: 1.0 and 0.0 -> 0.5. C: 0.0 and 0.0 -> 0.0.
        assert by_id[library["A"]].score == pytest.approx(0.5, abs=1e-5)
        assert by_id[library["C"]].score == pytest.approx(0.0, abs=1e-6)

    def test_item_without_second_model_is_not_penalised(self, repo, library):
        # B has no other_model embeddings, mimicking a pending backfill. Under
        # mean fusion it must be judged on the model that did score it.
        repo.set_embeddings(library["A"], "other_model", [(0.0, 5.0, basis(1))])

        results = run_search(
            repo,
            {
                "fake_model": basis(0).reshape(1, -1),
                "other_model": basis(0).reshape(1, -1),
            },
            fusion="mean",
        )
        by_id = {r.item_id: r for r in results}
        assert set(by_id[library["B"]].per_model) == {"fake_model"}
        assert by_id[library["B"]].score == pytest.approx(
            float(np.cos(0.3 * np.pi / 2)), abs=1e-5
        )

    def test_single_model_fusion_ignores_the_other(self, repo, library):
        repo.set_embeddings(library["C"], "other_model", [(0.0, 5.0, basis(0))])
        results = run_search(
            repo,
            {
                "fake_model": basis(0).reshape(1, -1),
                "other_model": basis(0).reshape(1, -1),
            },
            fusion="fake_model",
            k=1,
        )
        assert results[0].item_id == library["A"]
        assert set(results[0].per_model) == {"fake_model"}
