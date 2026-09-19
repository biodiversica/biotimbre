"""Shared fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from biotimbre.config import Settings
from biotimbre.db.connection import initialize
from biotimbre.db.repository import Repository
from biotimbre.domain.item import FilteredBand, ItemDraft, Location, SampleWindow
from biotimbre.models_registry import ModelSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "input_example"

# NiceGUI's simulated-user fixtures, for testing pages without a browser. pytest
# only honours pytest_plugins in the root conftest, not in a test module. The
# umbrella "nicegui.testing.plugin" is deliberately avoided: it imports selenium
# for a browser-driven fixture this suite does not use.
try:  # pragma: no cover - depends on whether the gui extra is installed
    import nicegui  # noqa: F401

    pytest_plugins = ["nicegui.testing.user_plugin"]
except ImportError:  # pragma: no cover
    pytest_plugins = []

#: The module whose ``@ui.page`` decorators define ``/`` and ``/search``.
PAGES_MODULE = "biotimbre.ui.app"


@pytest.fixture(autouse=True)
def registered_pages(request):
    """Give every page test its routes back.

    ``biotimbre.ui.app`` registers ``/`` and ``/search`` with module-level
    ``@ui.page`` decorators, which run once when the module is first imported.
    NiceGUI's ``user`` fixture resets the router around every test, so those
    routes have to be re-registered each time or ``user.open("/")`` 404s.

    NiceGUI does that itself, but only for modules it knows about: on teardown
    it drops each ``Client.page_routes`` function's module from ``sys.modules``
    (``nicegui/testing/general.py``), so the next import re-runs the
    decorators. That cycle only starts if the module was first imported
    *inside* a ``user`` fixture's lifetime — setup clears ``page_routes``, so a
    module imported before any test (``tests/test_ui.py`` imports it while
    pytest is still collecting) is never recorded, never evicted, stays cached
    for the rest of the session and never registers its pages again.

    Importing it here, after the reset and for every test that takes ``user``,
    makes that independent of who imported what first.
    """
    if "user" not in request.fixturenames:
        return
    # Force the reset to happen before the import rather than after it.
    request.getfixturevalue("user")

    import importlib
    import sys

    module = sys.modules.get(PAGES_MODULE)
    if module is None:
        importlib.import_module(PAGES_MODULE)
    else:
        # Cached from before the reset, so its decorators did not run: this is
        # what re-registers the pages.
        importlib.reload(module)


#: Models used by the ``models``-marked tests.
MODEL_IDS = ("perch_v2", "birdnet_v2.4")


@pytest.fixture(scope="session")
def shared_pool():
    """One ONNX session set for the whole test run.

    Creating a pool costs about two seconds of session setup (Perch 1.56 s,
    BirdNET 0.42 s), which dwarfs the inference in most individual tests. Sessions
    are read-only, so sharing them across tests is safe and removes that setup
    from every test but the first.

    Tests that need their own session parameters — a specific ``max_batch``, say —
    should build their own extractor instead.
    """
    from biotimbre.embeddings.extractor import ExtractorPool

    pool = ExtractorPool(MODEL_IDS)
    yield pool
    pool.close()


@pytest.fixture(scope="session")
def perch(shared_pool):
    """The shared Perch extractor."""
    return shared_pool.get("perch_v2")


@pytest.fixture(scope="session")
def birdnet(shared_pool):
    """The shared BirdNET extractor."""
    return shared_pool.get("birdnet_v2.4")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings rooted in a temporary data directory."""
    data_dir = tmp_path / "data"
    s = Settings(
        data_dir=data_dir,
        db_path=data_dir / "library.db",
        audio_dir=data_dir / "audio",
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def conn(settings: Settings):
    """An initialised, empty library database."""
    connection = initialize(settings.db_path)
    yield connection
    connection.close()


@pytest.fixture
def repo(conn: sqlite3.Connection) -> Repository:
    """A repository over an empty library."""
    return Repository(conn)


@pytest.fixture
def draft() -> ItemDraft:
    """A minimal valid draft, using only mandatory fields."""
    return ItemDraft(
        date="2020-01-14",
        time="09:40:21",
        location=Location(-20.0, -46.0),
        sample_window=SampleWindow(5.0, 10.0),
    ).validated()


@pytest.fixture
def fake_spec() -> ModelSpec:
    """A model spec that needs no download, for repository-level tests."""
    return ModelSpec(
        model_id="fake_model",
        repo_id="test/fake",
        filename="fake.onnx",
        sample_rate=32_000,
        window_duration_s=5.0,
        window_overlap=0.5,
        description="test double",
    )


def add_item(
    repo: Repository,
    draft: ItemDraft,
    *,
    sha256: str = "0" * 64,
    duration_s: float = 5.0,
    sample_rate: int = 32_000,
) -> int:
    """Insert an item with plausible audio properties, returning its ID."""
    item = repo.create_item(
        draft,
        audio_path=f"audio/pending/{sha256[:8]}.wav",
        audio_sha256=sha256,
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=1,
    )
    return item.item_id


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """``n`` random L2-normalised float32 vectors of width ``dim``."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)
