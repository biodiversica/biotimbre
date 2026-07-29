"""ONNX Runtime execution for embedding backbones.

The session is created lazily on first use, so constructing a backend is cheap
and a model is only downloaded when embeddings are actually needed.

Two properties of the graph are discovered rather than assumed:

* the **embedding dimension**, read from the output shape or, when the graph
  declares it dynamically, from one real inference;
* whether the input's **batch dimension is dynamic**, which decides between one
  batched call and a loop over windows.

Both matter because getting them wrong produces either a crash or — worse —
silently mis-shaped vectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biotimbre.domain.errors import EmbeddingError

#: Threads per inference session. ``0`` leaves the choice to onnxruntime.
#:
#: Measured on an 8-core CPU with a 16-window BirdNET batch: 3.59 s at one thread
#: versus 3.01 s at onnxruntime's default — these graphs are bandwidth-bound, so
#: threading buys about 16 %, not a multiple. That 16 % is still worth having for
#: an interactive search, hence the default. Pin this to 1 if you ever run
#: several extractors as parallel worker processes, where oversubscription would
#: cost more than the threading gains.
DEFAULT_THREADS = 0

#: Windows fed to the model per inference call.
#:
#: Peak memory scales linearly with the batch — measured at roughly 19 MB per
#: BirdNET window — so an unbounded batch is a latent out-of-memory bug: a 60 s
#: query at speed 0.25 becomes 240 s of shifted audio, about 160 windows and
#: ~3 GB in a single call, and a long recording at ingest is worse. Chunking
#: bounds that at ~300 MB while keeping nearly all the throughput benefit of
#: batching over a per-window loop.
DEFAULT_MAX_BATCH = 16


class OnnxEmbedder:
    """Runs an ONNX backbone over batches of audio windows.

    Args:
        model_path: Path to the ``.onnx`` file.
        model_id: Identifier used in error messages.
        output_name: Graph output to read embeddings from. ``None`` uses the
            first output, which is correct for a backbone-only graph.
        max_batch: Windows per inference call. See :data:`DEFAULT_MAX_BATCH`.
        threads: Threads per session. See :data:`DEFAULT_THREADS`.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str = "",
        output_name: str | None = None,
        max_batch: int = DEFAULT_MAX_BATCH,
        threads: int = DEFAULT_THREADS,
    ):
        self._model_path = Path(model_path)
        self._model_id = model_id or self._model_path.stem
        self._requested_output = output_name
        self._max_batch = max(1, int(max_batch))
        self._threads = max(0, int(threads))

        self._session = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._embedding_dim: int | None = None
        self._dynamic_batch: bool = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _ensure_session(self) -> None:
        """Create the inference session and inspect the graph, once."""
        if self._session is not None:
            return

        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            if self._threads > 0:
                options.intra_op_num_threads = self._threads
                options.inter_op_num_threads = self._threads
            session = ort.InferenceSession(
                str(self._model_path),
                options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load ONNX model {self._model_id!r} from "
                f"{self._model_path}: {exc}"
            ) from exc

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            raise EmbeddingError(
                f"Model {self._model_id!r} declares no "
                f"{'inputs' if not inputs else 'outputs'}."
            )

        self._session = session
        self._input_name = inputs[0].name

        if self._requested_output is None:
            self._output_name = outputs[0].name
        else:
            available = [o.name for o in outputs]
            if self._requested_output not in available:
                raise EmbeddingError(
                    f"Model {self._model_id!r} has no output "
                    f"{self._requested_output!r}. Available: "
                    f"{', '.join(available)}."
                )
            self._output_name = self._requested_output

        # A dimension declared as a string (e.g. "batch") or None is dynamic.
        input_shape = inputs[0].shape
        self._dynamic_batch = bool(input_shape) and not isinstance(
            input_shape[0], int
        )

        output_shape = next(
            o.shape for o in outputs if o.name == self._output_name
        )
        trailing = output_shape[-1] if output_shape else None
        self._embedding_dim = trailing if isinstance(trailing, int) else None

    @property
    def model_id(self) -> str:
        """Identifier this embedder reports itself as."""
        return self._model_id

    @property
    def input_name(self) -> str:
        """Name of the graph input audio is fed to."""
        self._ensure_session()
        return self._input_name

    @property
    def output_name(self) -> str:
        """Name of the graph output embeddings are read from."""
        self._ensure_session()
        return self._output_name

    @property
    def has_dynamic_batch(self) -> bool:
        """Whether the graph accepts more than one window per call."""
        self._ensure_session()
        return self._dynamic_batch

    def embedding_dim(self, window_samples: int) -> int:
        """The width of this model's embeddings.

        Read from the graph's declared output shape when it is static. When the
        graph declares it dynamically, one inference on silence is run to measure
        it — cheap, and done once per process.

        Args:
            window_samples: Input length the model expects, used only if a probe
                inference is needed.

        Returns:
            Embedding width in dimensions.

        Raises:
            EmbeddingError: If the dimension cannot be determined.
        """
        self._ensure_session()
        if self._embedding_dim is not None:
            return self._embedding_dim

        probe = np.zeros((1, window_samples), dtype=np.float32)
        vectors = self.run(probe)
        self._embedding_dim = int(vectors.shape[-1])
        return self._embedding_dim

    # ── inference ─────────────────────────────────────────────────────────────

    def run(self, windows: np.ndarray) -> np.ndarray:
        """Embed a batch of audio windows.

        Inference is chunked at :attr:`max_batch` windows per call so peak memory
        stays bounded regardless of how long the input is. A graph without a
        dynamic batch dimension is driven one window at a time.

        Args:
            windows: ``(n_windows, window_samples)`` float32 audio.

        Returns:
            An ``(n_windows, embedding_dim)`` float32 array.

        Raises:
            EmbeddingError: If inference fails or returns an unusable shape.
        """
        self._ensure_session()

        if windows.ndim != 2:
            raise EmbeddingError(
                f"Expected a 2-D (n_windows, samples) array, got shape "
                f"{windows.shape}."
            )
        if windows.shape[0] == 0:
            raise EmbeddingError("Cannot embed an empty batch of windows.")

        batch = np.ascontiguousarray(windows, dtype=np.float32)
        step = self._max_batch if self._dynamic_batch else 1

        outputs = [
            self._flatten_output(self._infer(batch[start : start + step]))
            for start in range(0, batch.shape[0], step)
        ]
        vectors = np.vstack(outputs)
        if vectors.shape[0] != batch.shape[0]:
            raise EmbeddingError(
                f"Model {self._model_id!r} returned {vectors.shape[0]} vectors "
                f"for {batch.shape[0]} windows."
            )
        return vectors

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        """Run one session call, translating any failure into EmbeddingError."""
        try:
            result = self._session.run(
                [self._output_name], {self._input_name: batch}
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Inference failed for model {self._model_id!r} on input of "
                f"shape {batch.shape}: {exc}"
            ) from exc
        return np.asarray(result[0])

    def _flatten_output(self, output: np.ndarray) -> np.ndarray:
        """Reduce a raw model output to ``(n, embedding_dim)`` float32.

        Backbones sometimes emit extra unit axes, or a pooled time axis. Unit
        axes are squeezed; a genuine remaining time axis is mean-pooled, which is
        the conventional way to reduce a frame-wise backbone output to one vector
        per window.

        Args:
            output: Raw array from the session.

        Returns:
            A 2-D float32 array.

        Raises:
            EmbeddingError: If the output has no usable shape.
        """
        arr = np.asarray(output, dtype=np.float32)
        if arr.ndim == 0:
            raise EmbeddingError(
                f"Model {self._model_id!r} returned a scalar, not an embedding."
            )
        if arr.ndim == 1:
            return arr.reshape(1, -1)

        # Drop unit axes between batch and features, e.g. (1, 1, 1, 1536).
        while arr.ndim > 2:
            squeezable = [
                axis
                for axis in range(1, arr.ndim - 1)
                if arr.shape[axis] == 1
            ]
            if not squeezable:
                break
            arr = np.squeeze(arr, axis=squeezable[0])

        if arr.ndim > 2:
            # A real time axis remains: average it into one vector per window.
            arr = arr.reshape(arr.shape[0], -1, arr.shape[-1]).mean(axis=1)

        return np.ascontiguousarray(arr, dtype=np.float32)

    def close(self) -> None:
        """Release the inference session."""
        self._session = None
