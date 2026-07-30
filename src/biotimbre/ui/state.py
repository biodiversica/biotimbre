"""Application state shared by every page and request.

Two things are deliberately process-wide rather than per-request:

**The service.** :class:`~biotimbre.service.LibraryService` holds the query
embedding cache, which is what makes adjusting a filter cost 0.003 s instead of
5 s. Constructing one per request would throw that away every time.

**The extractor pool.** Creating ONNX sessions costs about two seconds, so it
happens once at startup — in the background, so the server answers immediately
and the page can say models are still loading rather than appearing to hang.

Both come with a concurrency obligation. A single SQLite connection used from
several threads needs serialising, and inference must not run on the event loop or
the interface freezes. :meth:`AppState.call` handles both: it runs the work in a
worker thread while holding a lock.
"""

from __future__ import annotations

import asyncio
import secrets
import shutil
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from biotimbre.config import Settings, load_settings
from biotimbre.db.connection import initialize
from biotimbre.embeddings.extractor import ExtractorPool
from biotimbre.service import LibraryService

T = TypeVar("T")


class QueryStore:
    """Staged query files, addressable by opaque token.

    A query is an upload, so it has no item ID, and a route is needed anyway
    because a WAV is far too large to inline as a data URI. Serving by token
    rather than by path preserves the property that a request can never name a
    filesystem location: an unknown token is simply a 404.

    Tokens are bounded in number so a long session cannot accumulate uploads
    indefinitely; the oldest is evicted and its file removed.
    """

    #: Staged queries kept before the oldest is dropped.
    MAX_ENTRIES = 8

    def __init__(self) -> None:
        self._paths: OrderedDict[str, Path] = OrderedDict()
        self._directory = Path(tempfile.mkdtemp(prefix="biotimbre-query-"))

    def add(self, name: str, data: bytes) -> str:
        """Stage a query file and return its token.

        Args:
            name: Original filename, used only for its extension and basename.
            data: File bytes.

        Returns:
            An opaque token for :meth:`path`.
        """
        token = secrets.token_urlsafe(16)
        # Basename only: an upload name is caller-controlled and must not be able
        # to escape the staging directory.
        target = self._directory / f"{token}_{Path(name).name}"
        target.write_bytes(data)
        self._paths[token] = target

        while len(self._paths) > self.MAX_ENTRIES:
            _, stale = self._paths.popitem(last=False)
            stale.unlink(missing_ok=True)
        return token

    def path(self, token: str) -> Path | None:
        """Resolve a token to its staged file, or ``None`` if unknown."""
        path = self._paths.get(token)
        return path if path is not None and path.is_file() else None

    def clear(self) -> None:
        """Remove every staged file."""
        for path in self._paths.values():
            path.unlink(missing_ok=True)
        self._paths.clear()
        shutil.rmtree(self._directory, ignore_errors=True)


@dataclass
class ModelStatus:
    """Progress of the background model warm-up.

    Attributes:
        ready: True once every active model has a live session.
        failed: Error message if warm-up failed, else ``None``. A failure is not
            fatal: browsing and editing metadata still work, only embedding and
            searching do not.
        dimensions: Embedding width per model, once known.
    """

    ready: bool = False
    failed: str | None = None
    dimensions: dict[str, int] = field(default_factory=dict)


class AppState:
    """Long-lived state for the running interface.

    Args:
        settings: Resolved settings. ``None`` loads them from the usual places.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings if settings is not None else load_settings()
        self.settings.ensure_dirs()

        # allow_threads because request handlers and the inference worker do not
        # share a thread; the lock below is what makes that safe.
        self._connection = initialize(self.settings.db_path, allow_threads=True)
        self._lock = threading.RLock()

        self.pool = ExtractorPool(
            self.settings.active_models,
            cache_dir=self.settings.model_cache_dir,
            max_batch=self.settings.inference_batch_size,
            threads=self.settings.inference_threads,
        )
        self.service = LibraryService(
            self.settings, connection=self._connection, pool=self.pool
        )
        self.models = ModelStatus()
        self.queries = QueryStore()

    # ── concurrency ───────────────────────────────────────────────────────────

    def call(self, work: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run library work under the lock, on the calling thread.

        Use from a worker thread. From async code use :meth:`run`, which offloads
        so the event loop stays responsive.

        Args:
            work: Callable to invoke.
            *args: Positional arguments for ``work``.
            **kwargs: Keyword arguments for ``work``.

        Returns:
            Whatever ``work`` returns.
        """
        with self._lock:
            return work(*args, **kwargs)

    async def run(self, work: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run library work in a worker thread, under the lock.

        Inference takes seconds; doing it on the event loop would freeze every
        open page, including the one showing the progress spinner.

        Args:
            work: Callable to invoke.
            *args: Positional arguments for ``work``.
            **kwargs: Keyword arguments for ``work``.

        Returns:
            Whatever ``work`` returns.
        """
        return await asyncio.to_thread(self.call, work, *args, **kwargs)

    # ── model warm-up ─────────────────────────────────────────────────────────

    def warm_models(self) -> None:
        """Create the ONNX sessions and register the models. Blocking.

        Safe to call more than once; the second call is a no-op.
        """
        if self.models.ready:
            return
        try:
            dimensions = self.call(self.service.ensure_models_registered)
        except Exception as exc:  # download failure, missing file, bad graph
            self.models = ModelStatus(ready=False, failed=str(exc))
            return
        self.models = ModelStatus(ready=True, dimensions=dimensions)

    def start_warming(self) -> threading.Thread:
        """Warm the models in the background and return the thread.

        The server starts serving immediately, so a first run that has to download
        70 MB shows progress instead of an unresponsive page.
        """
        thread = threading.Thread(
            target=self.warm_models, name="biotimbre-warm-models", daemon=True
        )
        thread.start()
        return thread

    # ── convenience ───────────────────────────────────────────────────────────

    def audio_url(self, item_id: int) -> str:
        """URL of an item's audio, served through the access-controlled route."""
        return f"/media/audio/{item_id}"

    def query_audio_url(self, token: str) -> str:
        """URL of a staged query file, by token."""
        return f"/media/query/{token}"

    def spectrogram_url(
        self,
        item_id: int,
        *,
        width: int | None = None,
        height: int | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> str:
        """URL of an item's spectrogram, with optional span and size."""
        query: list[str] = []
        if width is not None:
            query.append(f"w={width}")
        if height is not None:
            query.append(f"h={height}")
        if start is not None:
            query.append(f"start={start}")
        if end is not None:
            query.append(f"end={end}")
        suffix = f"?{'&'.join(query)}" if query else ""
        return f"/media/spectrogram/{item_id}{suffix}"

    def close(self) -> None:
        """Release models, staged queries and the database connection."""
        with self._lock:
            self.queries.clear()
            self.service.close()
            self.pool.close()
            self._connection.close()


#: Set by :func:`biotimbre.ui.app.create_app` so routes and pages can reach it.
_state: AppState | None = None


def set_state(state: AppState) -> None:
    """Install the process-wide state."""
    global _state
    _state = state


def get_state() -> AppState:
    """Return the process-wide state.

    Raises:
        RuntimeError: If the app has not been created yet, which would otherwise
            surface as a confusing ``NoneType`` error inside a request handler.
    """
    if _state is None:
        raise RuntimeError(
            "Application state is not initialised. Call create_app() first."
        )
    return _state
