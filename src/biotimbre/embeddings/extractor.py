"""Embedding extraction: audio file in, per-window vectors out.

An :class:`Extractor` owns one model's session and preprocessing, and is reused
across many files — loading a 45 MB graph per item would dominate ingest time.
:class:`ExtractorPool` holds one extractor per active model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from biotimbre.embeddings import audio as audio_io
from biotimbre.embeddings.onnx_backend import (
    DEFAULT_MAX_BATCH,
    DEFAULT_THREADS,
    OnnxEmbedder,
)
from biotimbre.models_registry import ModelSpec, get_spec, resolve_model_file


@dataclass(frozen=True)
class WindowEmbeddings:
    """Embeddings for one audio file under one model.

    Attributes:
        model_id: Model that produced the vectors.
        vectors: ``(n_windows, embedding_dim)`` float32 array.
        spans: ``(start_time, end_time)`` per window, in seconds within the
            sample — already converted back from any speed shift.
        speed: Speed factor applied before inference.
    """

    model_id: str
    vectors: np.ndarray
    spans: list[tuple[float, float]]
    speed: float

    @property
    def n_windows(self) -> int:
        """Number of windows."""
        return int(self.vectors.shape[0])

    @property
    def embedding_dim(self) -> int:
        """Width of each vector."""
        return int(self.vectors.shape[1])

    def as_rows(self) -> list[tuple[float, float, np.ndarray]]:
        """Shape for :meth:`biotimbre.db.repository.Repository.set_embeddings`."""
        return [
            (start, end, self.vectors[index])
            for index, (start, end) in enumerate(self.spans)
        ]


class Extractor:
    """Computes embeddings for one model, reusing its session across files.

    Args:
        spec: The model's specification.
        model_path: Local path to the ONNX file. ``None`` resolves it via the
            registry, downloading if needed.
        cache_dir: Model cache directory passed to the registry.
        max_batch: Windows per inference call, bounding peak memory on long
            inputs. See :data:`~biotimbre.embeddings.onnx_backend.DEFAULT_MAX_BATCH`.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        model_path: str | Path | None = None,
        cache_dir: Path | None = None,
        max_batch: int = DEFAULT_MAX_BATCH,
        threads: int = DEFAULT_THREADS,
    ):
        self.spec = spec
        self._model_path = (
            Path(model_path)
            if model_path is not None
            else resolve_model_file(spec, cache_dir)
        )
        self._embedder = OnnxEmbedder(
            self._model_path,
            model_id=spec.model_id,
            max_batch=max_batch,
            threads=threads,
        )

    @property
    def model_id(self) -> str:
        """The model's ID."""
        return self.spec.model_id

    @property
    def model_path(self) -> Path:
        """Local path to the resolved ONNX file."""
        return self._model_path

    def embedding_dim(self) -> int:
        """This model's embedding width, read from the graph."""
        return self._embedder.embedding_dim(self.spec.window_samples)

    def embed_array(
        self, samples: np.ndarray, native_rate: int, *, speed: float = 1.0
    ) -> WindowEmbeddings:
        """Embed in-memory mono audio.

        Args:
            samples: 1-D float32 audio at ``native_rate``.
            native_rate: The audio's real sample rate in Hz.
            speed: Speed factor applied before resampling (PLAN.md D2b).

        Returns:
            The resulting :class:`WindowEmbeddings`.

        Raises:
            EmbeddingError: If inference fails.
            ValidationError: If the audio or speed is unusable.
        """
        windows, spans = audio_io.prepare_windows(
            samples,
            native_rate,
            model_rate=self.spec.sample_rate,
            window_samples=self.spec.window_samples,
            stride_samples=self.spec.stride_samples,
            speed=speed,
        )
        vectors = self._embedder.run(windows)
        return WindowEmbeddings(
            model_id=self.spec.model_id,
            vectors=vectors,
            spans=spans,
            speed=float(speed),
        )

    def embed_file(
        self, path: str | Path, *, speed: float = 1.0
    ) -> WindowEmbeddings:
        """Embed an audio file.

        Args:
            path: Path to the audio file.
            speed: Speed factor applied before resampling.

        Returns:
            The resulting :class:`WindowEmbeddings`.

        Raises:
            EmbeddingError: If inference fails.
            ValidationError: If the file cannot be decoded.
        """
        samples, native_rate = audio_io.load_mono(path)
        return self.embed_array(samples, native_rate, speed=speed)

    def close(self) -> None:
        """Release the model session."""
        self._embedder.close()


class ExtractorPool:
    """Lazily-built collection of extractors, one per model ID.

    Args:
        model_ids: Models to serve.
        cache_dir: Model cache directory.
        max_batch: Windows per inference call, applied to every extractor.
        threads: Threads per session, applied to every extractor.
    """

    def __init__(
        self,
        model_ids: tuple[str, ...] | list[str],
        *,
        cache_dir: Path | None = None,
        max_batch: int = DEFAULT_MAX_BATCH,
        threads: int = DEFAULT_THREADS,
    ):
        self._model_ids = tuple(model_ids)
        self._cache_dir = cache_dir
        self._max_batch = max_batch
        self._threads = threads
        self._extractors: dict[str, Extractor] = {}

    @property
    def model_ids(self) -> tuple[str, ...]:
        """IDs this pool can serve."""
        return self._model_ids

    def get(self, model_id: str) -> Extractor:
        """Return the extractor for a model, building it on first use.

        Args:
            model_id: The model's ID.

        Returns:
            A ready :class:`Extractor`.

        Raises:
            KeyError: If the model is not in this pool.
            RuntimeError: If the model file cannot be obtained.
        """
        if model_id not in self._model_ids:
            raise KeyError(
                f"Model {model_id!r} is not in this pool. "
                f"Available: {', '.join(self._model_ids)}."
            )
        if model_id not in self._extractors:
            self._extractors[model_id] = Extractor(
                get_spec(model_id),
                cache_dir=self._cache_dir,
                max_batch=self._max_batch,
                threads=self._threads,
            )
        return self._extractors[model_id]

    def embed_file(
        self,
        path: str | Path,
        *,
        speed: float = 1.0,
        models: Sequence[str] | None = None,
    ) -> dict[str, WindowEmbeddings]:
        """Embed one file with every model in the pool.

        The file is decoded once and shared, since resampling is per-model but
        decoding is not.

        Args:
            path: Path to the audio file.
            speed: Speed factor applied before resampling.
            models: Restrict to these models. ``None`` uses every model in the
                pool. An import that already carries usable vectors for one
                model names only the rest, so no inference is run for work
                someone else has already done.

        Returns:
            Mapping of model ID to its :class:`WindowEmbeddings`.

        Raises:
            KeyError: If ``models`` names a model this pool does not serve.
        """
        wanted = self._model_ids if models is None else tuple(models)
        if not wanted:
            return {}
        samples, native_rate = audio_io.load_mono(path)
        return {
            model_id: self.get(model_id).embed_array(
                samples, native_rate, speed=speed
            )
            for model_id in wanted
        }

    def embed_array_multi_speed(
        self, samples: np.ndarray, native_rate: int, speeds: list[float]
    ) -> dict[tuple[str, float], WindowEmbeddings]:
        """Embed in-memory audio at several speeds, for the query-side sweep.

        A query must be embedded at each speed present in the library to be
        comparable with the items indexed at that speed (PLAN.md D2c).

        Args:
            samples: 1-D float32 audio.
            native_rate: The audio's real sample rate in Hz.
            speeds: Speed factors to embed at.

        Returns:
            Mapping of ``(model_id, speed)`` to its :class:`WindowEmbeddings`.
        """
        result: dict[tuple[str, float], WindowEmbeddings] = {}
        for model_id in self._model_ids:
            extractor = self.get(model_id)
            for speed in speeds:
                result[(model_id, float(speed))] = extractor.embed_array(
                    samples, native_rate, speed=speed
                )
        return result

    def close(self) -> None:
        """Release every built extractor's session."""
        for extractor in self._extractors.values():
            extractor.close()
        self._extractors.clear()
