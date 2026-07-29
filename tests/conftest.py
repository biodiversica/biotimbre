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
