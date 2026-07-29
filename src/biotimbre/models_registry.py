"""Registry of embedding models.

A "model" here is an *(ONNX graph, preprocessing)* pair: two entries may share a
graph but differ in sample rate, window length or overlap, and they are then
distinct models as far as the library is concerned, because their embeddings are
not comparable. ``model_id`` is therefore the identity of a preprocessing recipe,
not just of a file.

``embedding_dim`` is deliberately absent from :class:`ModelSpec`: it is read from
the ONNX graph at registration time and persisted in the ``models`` table. See
PLAN.md §1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Fractional window overlap used at ingest. A vocalization straddling a window
#: boundary is split and scores poorly in both halves at 0.0; 0.5 costs little
#: (a 90 s Perch sample is ~220 KB of vectors) and materially improves recall.
DEFAULT_WINDOW_OVERLAP = 0.5


@dataclass(frozen=True)
class ModelSpec:
    """Definition of an embedding model and its required preprocessing.

    Attributes:
        model_id: Stable identifier, used as the ``models`` table primary key
            and in ``embeddings.model_id``.
        repo_id: Hugging Face repository holding the ONNX file.
        filename: ONNX file name within the repository.
        sample_rate: Rate (Hz) the graph expects its input audio at.
        window_duration_s: Length of one inference window, in seconds.
        window_overlap: Fractional overlap between consecutive windows, [0, 0.95].
        revision: Git revision to pin. ``None`` tracks the default branch.
        description: Human-readable label for UI and CLI output.
    """

    model_id: str
    repo_id: str
    filename: str
    sample_rate: int
    window_duration_s: float
    window_overlap: float = DEFAULT_WINDOW_OVERLAP
    revision: str | None = None
    description: str = ""

    @property
    def source(self) -> str:
        """Provenance string persisted in ``models.source``."""
        return f"hf://{self.repo_id}"

    @property
    def window_samples(self) -> int:
        """Window length in samples at :attr:`sample_rate`."""
        return int(self.window_duration_s * self.sample_rate)

    @property
    def stride_samples(self) -> int:
        """Hop between consecutive window starts, in samples."""
        return max(1, int(self.window_samples * (1.0 - self.window_overlap)))

    @property
    def nyquist_hz(self) -> float:
        """Highest frequency surviving the resample to :attr:`sample_rate`.

        Content above this is removed by the anti-alias filter during
        resampling, which is what ``items.speed`` exists to work around
        (PLAN.md D2b).
        """
        return self.sample_rate / 2.0

    def n_windows(self, n_samples: int) -> int:
        """Number of windows produced for an input of ``n_samples``.

        Always at least one window: a signal shorter than the window is
        zero-padded rather than dropped. A trailing partial hop is not padded
        into an extra window, so a 5 s input with a 5 s window yields exactly
        one window regardless of overlap.

        Args:
            n_samples: Input length in samples, at :attr:`sample_rate`.

        Returns:
            Window count, >= 1.
        """
        excess = max(0, n_samples - self.window_samples)
        return 1 + -(-excess // self.stride_samples)


MODEL_SPECS: dict[str, ModelSpec] = {
    "perch_v2": ModelSpec(
        model_id="perch_v2",
        repo_id="biodiversica/Perch-onnx-backbone",
        filename="perch_v2_backbone.onnx",
        sample_rate=32_000,
        window_duration_s=5.0,
        description="Perch v2 backbone (embeddings only)",
    ),
    "birdnet_v2.4": ModelSpec(
        model_id="birdnet_v2.4",
        repo_id="biodiversica/BirdNET-onnx-backbone",
        filename="model_backbone.onnx",
        sample_rate=48_000,
        window_duration_s=3.0,
        description="BirdNET 2.4 backbone (embeddings only)",
    ),
}


def get_spec(model_id: str) -> ModelSpec:
    """Look up a model spec by ID.

    Args:
        model_id: Key of :data:`MODEL_SPECS`.

    Returns:
        The matching :class:`ModelSpec`.

    Raises:
        KeyError: If no such model is registered, listing the valid IDs.
    """
    try:
        return MODEL_SPECS[model_id]
    except KeyError:
        known = ", ".join(sorted(MODEL_SPECS)) or "(none)"
        raise KeyError(
            f"Unknown model id {model_id!r}. Registered models: {known}."
        ) from None


def resolve_model_file(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    """Download the ONNX file if needed and return its local path.

    Downloads on first use and serves from the local cache afterwards, so this
    is cheap to call repeatedly and works offline once populated.

    Args:
        spec: The model to resolve.
        cache_dir: Override the Hugging Face cache location. ``None`` uses the
            HF default.

    Returns:
        Path to the cached ONNX file.

    Raises:
        RuntimeError: If the download fails, with the cause attached.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except Exception as exc:  # network, auth, missing file — all unrecoverable here
        raise RuntimeError(
            f"Could not obtain model {spec.model_id!r} "
            f"({spec.filename} from {spec.repo_id}): {exc}"
        ) from exc
    return Path(path)
