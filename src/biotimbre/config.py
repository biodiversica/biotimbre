"""Library configuration.

Settings come from a TOML file, with every field optional and defaulted so a
fresh checkout works with no configuration at all. Resolution order for the
config file:

1. explicit path passed to :func:`load_settings`
2. ``$BIOTIMBRE_CONFIG``
3. ``./biotimbre.toml``
4. built-in defaults

Paths in the TOML may be relative; they resolve against the config file's own
directory, so a config file can be moved together with its data.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_NAME = "biotimbre.toml"
ENV_CONFIG = "BIOTIMBRE_CONFIG"

#: Languages the UI ships translations for. Single source of truth is the
#: catalogue package, so adding a language means adding a file, not editing two
#: lists that can drift apart.
from biotimbre.i18n import LANGUAGES as SUPPORTED_LANGUAGES  # noqa: E402


@dataclass(frozen=True)
class Settings:
    """Resolved library settings.

    Attributes:
        data_dir: Root for all library state (database + audio).
        db_path: SQLite file holding metadata and embedding vectors.
        audio_dir: Root of the stored audio tree
            (``<audio_dir>/<biotimbre_id>/<item_id>.wav``).
        model_cache_dir: Where ONNX model files are cached. ``None`` uses the
            Hugging Face default (``HF_HOME`` or ``~/.cache/huggingface``).
        active_models: Model IDs to compute embeddings with on ingest. Must be
            keys of :data:`biotimbre.models_registry.MODEL_SPECS`.
        language: Default UI language, one of :data:`SUPPORTED_LANGUAGES`.
        search_batch_size: Embedding rows scored per batch during search.
        inference_batch_size: Audio windows per model inference call. 1 is both
            fastest and safest for the current backbones — batching makes Perch
            2.1x slower. See onnx_backend.DEFAULT_MAX_BATCH.
        query_cache_entries: Distinct query audios whose embeddings are retained
            in memory, so re-filtering a query does not re-run the models.
        inference_threads: Threads per ONNX session. 0 lets onnxruntime choose.
        max_query_duration_s: Longest query segment a search will embed. A longer
            file requires an explicit start/end.
        default_top_k: Default number of search hits returned.
        default_metric: Default similarity metric ("cosine" or "dot").
        default_fusion: Default cross-model fusion ("max", "mean", or a model id).
    """

    data_dir: Path
    db_path: Path
    audio_dir: Path
    model_cache_dir: Path | None = None
    active_models: tuple[str, ...] = ("perch_v2", "birdnet_v2.4")
    language: str = "en"
    search_batch_size: int = 8192
    inference_batch_size: int = 1
    inference_threads: int = 0
    query_cache_entries: int = 16
    max_query_duration_s: float = 15.0
    default_top_k: int = 10
    default_metric: str = "cosine"
    default_fusion: str = "max"

    def ensure_dirs(self) -> None:
        """Create the data and audio directories if they do not exist."""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def audio_path_for(self, biotimbre_id: str, item_id: int) -> Path:
        """Absolute path where an item's audio is stored.

        Args:
            biotimbre_id: 4-character uppercase hex biotimbre ID.
            item_id: The item's integer ID.

        Returns:
            ``<audio_dir>/<biotimbre_id>/<item_id>.wav``.
        """
        return self.audio_dir / biotimbre_id / f"{item_id}.wav"

    def relative_audio_path(self, path: Path) -> str:
        """Convert an absolute audio path to the form stored in the database.

        Paths inside :attr:`data_dir` are stored **relative** to it, so the whole
        library is portable: move or rename the directory and every reference
        still resolves.

        A path outside :attr:`data_dir` is stored absolute. That happens when
        :attr:`audio_dir` is deliberately configured elsewhere — a database on
        fast local storage with audio on a large external drive, say. Those rows
        are no longer portable, which is the trade the configuration asked for;
        the alternative was failing at ingest with an opaque error.

        Args:
            path: Absolute path to a stored audio file.

        Returns:
            POSIX-style path, relative where possible.
        """
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.data_dir.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()

    def absolute_audio_path(self, stored: str) -> Path:
        """Inverse of :meth:`relative_audio_path`.

        Works for both forms: joining an absolute stored path onto
        :attr:`data_dir` yields the stored path unchanged.

        Args:
            stored: Path as held in ``items.audio_path``.

        Returns:
            Absolute path on this machine.
        """
        return self.data_dir / stored


def _default_settings(data_dir: Path) -> Settings:
    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "library.db",
        audio_dir=data_dir / "audio",
    )


def find_config_file(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Locate the config file to use, or ``None`` for built-in defaults.

    Args:
        explicit: A path supplied directly (e.g. from ``--config``).

    Returns:
        The resolved config path, or ``None`` if no config file is present.

    Raises:
        FileNotFoundError: If ``explicit`` or ``$BIOTIMBRE_CONFIG`` names a file
            that does not exist. A missing *implicit* config is not an error;
            a missing *requested* one is.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path.resolve()

    from_env = os.environ.get(ENV_CONFIG)
    if from_env:
        path = Path(from_env).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{ENV_CONFIG} points to a missing file: {path}"
            )
        return path.resolve()

    local = Path.cwd() / DEFAULT_CONFIG_NAME
    return local.resolve() if local.is_file() else None


def load_settings(
    config_path: str | os.PathLike[str] | None = None,
    *,
    data_dir: str | os.PathLike[str] | None = None,
) -> Settings:
    """Load settings from TOML, falling back to defaults for absent fields.

    Args:
        config_path: Explicit config file. See :func:`find_config_file`.
        data_dir: Overrides the data directory from any other source. Intended
            for tests and for CLI ``--data-dir``.

    Returns:
        Fully resolved :class:`Settings`.

    Raises:
        FileNotFoundError: If a requested config file is missing.
        ValueError: If a setting has an unusable value.
    """
    path = find_config_file(config_path)
    raw: dict = {}
    if path is not None:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)

    # Relative paths resolve against the config file's directory when there is
    # one, otherwise the current directory.
    base = path.parent if path is not None else Path.cwd()

    def _resolve(value: str | os.PathLike[str]) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (base / candidate)

    library = raw.get("library", {})
    resolved_data_dir = (
        _resolve(data_dir)
        if data_dir is not None
        else _resolve(library.get("data_dir", "data"))
    )

    defaults = _default_settings(resolved_data_dir)

    db_path = (
        _resolve(library["db_path"]) if "db_path" in library else defaults.db_path
    )
    audio_dir = (
        _resolve(library["audio_dir"])
        if "audio_dir" in library
        else defaults.audio_dir
    )
    model_cache_dir = (
        _resolve(library["model_cache_dir"])
        if library.get("model_cache_dir")
        else None
    )

    models_section = raw.get("models", {})
    active = models_section.get("active", list(defaults.active_models))
    if isinstance(active, str):
        active = [active]
    if not active:
        raise ValueError("models.active must list at least one model id.")

    ui = raw.get("ui", {})
    language = ui.get("language", defaults.language)
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Expected one of {', '.join(SUPPORTED_LANGUAGES)}."
        )

    search = raw.get("search", {})
    batch_size = int(search.get("batch_size", defaults.search_batch_size))
    if batch_size < 1:
        raise ValueError("search.batch_size must be >= 1.")
    top_k = int(search.get("top_k", defaults.default_top_k))
    if top_k < 1:
        raise ValueError("search.top_k must be >= 1.")

    inference_batch = int(
        models_section.get("inference_batch_size", defaults.inference_batch_size)
    )
    if inference_batch < 1:
        raise ValueError("models.inference_batch_size must be >= 1.")
    cache_entries = int(
        search.get("query_cache_entries", defaults.query_cache_entries)
    )
    if cache_entries < 1:
        raise ValueError("search.query_cache_entries must be >= 1.")
    threads = int(models_section.get("inference_threads", defaults.inference_threads))
    if threads < 0:
        raise ValueError("models.inference_threads must be >= 0 (0 = automatic).")
    max_query = float(
        search.get("max_query_duration_s", defaults.max_query_duration_s)
    )
    if max_query <= 0:
        raise ValueError("search.max_query_duration_s must be > 0.")

    return Settings(
        data_dir=resolved_data_dir,
        db_path=db_path,
        audio_dir=audio_dir,
        model_cache_dir=model_cache_dir,
        active_models=tuple(active),
        language=language,
        search_batch_size=batch_size,
        inference_batch_size=inference_batch,
        inference_threads=threads,
        query_cache_entries=cache_entries,
        max_query_duration_s=max_query,
        default_top_k=top_k,
        default_metric=search.get("metric", defaults.default_metric),
        default_fusion=search.get("fusion", defaults.default_fusion),
    )
