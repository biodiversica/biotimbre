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

    def reserve(self, name: str) -> tuple[str, Path]:
        """Allocate a token and a path to write to, evicting the oldest entry.

        Split from writing so a caller can stream an upload straight to disk
        instead of holding it in memory — a query may be a long recording.

        Args:
            name: Original filename. Only its basename is used, so a
                caller-controlled name cannot escape the staging directory.

        Returns:
            The token and the path to write.
        """
        token = secrets.token_urlsafe(16)
        target = self._directory / f"{token}_{Path(name).name}"
        self._paths[token] = target

        while len(self._paths) > self.MAX_ENTRIES:
            _, stale = self._paths.popitem(last=False)
            stale.unlink(missing_ok=True)
        return token, target

    def add(self, name: str, data: bytes) -> str:
        """Stage a query file from bytes and return its token.

        Args:
            name: Original filename.
            data: File bytes.

        Returns:
            An opaque token for :meth:`path`.
        """
        token, target = self.reserve(name)
        target.write_bytes(data)
        return token

    def replace_path(self, token: str, path: Path) -> None:
        """Point a token at a different file, deleting the one it named.

        A fetched recording may have to be decoded on the way in, and what
        lands is then a WAV beside the name that was reserved. The token is
        the query's identity, so it follows the file rather than the reverse.

        Args:
            token: Token from :meth:`reserve`.
            path: The file it should name from now on.
        """
        previous = self._paths.get(token)
        if previous is not None and previous != path:
            previous.unlink(missing_ok=True)
        self._paths[token] = path

    def path(self, token: str) -> Path | None:
        """Resolve a token to its staged file, or ``None`` if unknown."""
        path = self._paths.get(token)
        return path if path is not None and path.is_file() else None

    def discard(self, token: str) -> None:
        """Remove one staged file, if the token is known.

        Args:
            token: Token returned by :meth:`reserve` or :meth:`add`. An unknown
                token is not an error: dropping a query twice is harmless.
        """
        path = self._paths.pop(token, None)
        if path is not None:
            path.unlink(missing_ok=True)

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
        self._lock = threading.RLock()

        # The models do not depend on which library is open — only on the model
        # cache — so warm-up can start while the reader is still choosing one,
        # and choosing one does not pay for it again.
        self.pool = ExtractorPool(
            self.settings.active_models,
            cache_dir=self.settings.model_cache_dir,
            max_batch=self.settings.inference_batch_size,
            threads=self.settings.inference_threads,
        )
        self.models = ModelStatus()
        self.queries = QueryStore()

        self._connection = None
        self._service: LibraryService | None = None
        if self.settings.data_dir_chosen:
            self.open_library(self.settings.data_dir)

    # ── the open library ──────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """Whether a library has been opened yet."""
        return self._service is not None

    @property
    def service(self) -> LibraryService:
        """The open library.

        Raises:
            RuntimeError: If no library has been opened. Pages guard on
                :attr:`is_open` and offer the chooser instead; reaching here
                without one is a bug, not a user error.
        """
        if self._service is None:
            raise RuntimeError("No library is open.")
        return self._service

    def open_library(
        self, data_dir: str | Path, *, name: str | None = None
    ) -> Settings:
        """Open a library directory, replacing whatever was open before.

        The directory is created if it does not exist, and initialised if it
        holds no database yet — opening a new library and opening an old one
        are the same act from the reader's side.

        Args:
            data_dir: Directory the library lives in.
            name: Name to give a library that does not have one yet. An
                existing name is never overwritten here.

        Returns:
            The settings now in force.

        Raises:
            BiotimbreError: If the directory cannot be used as a library.
        """
        with self._lock:
            settings = load_settings(data_dir=data_dir)
            settings.ensure_dirs()
            # allow_threads because request handlers and the inference worker
            # do not share a thread; the lock is what makes that safe.
            connection = initialize(settings.db_path, allow_threads=True)
            service = LibraryService(settings, connection=connection, pool=self.pool)
            if name and not service.identity().name:
                service.rename(name)

            self.close_library()
            self.settings = settings
            self._connection = connection
            self._service = service
            if self.models.ready:
                # The sessions are already built; the new library still has to
                # learn their dimensions before anything can be embedded.
                service.ensure_models_registered()
            return settings

    def close_library(self) -> None:
        """Close the open library, if any, leaving the models loaded."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self._service = None

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

        Safe to call more than once; the second call is a no-op. With no
        library open the sessions are still built — that is the slow part, and
        it does not depend on which library it is — and the models are
        registered when one is opened.
        """
        if self.models.ready:
            return
        try:
            if self.is_open:
                dimensions = self.call(self.service.ensure_models_registered)
            else:
                dimensions = self.call(self._session_dimensions)
        except Exception as exc:  # download failure, missing file, bad graph
            self.models = ModelStatus(ready=False, failed=str(exc))
            return
        self.models = ModelStatus(ready=True, dimensions=dimensions)

    def _session_dimensions(self) -> dict[str, int]:
        """Build every active model's session, without a library to record it in."""
        return {
            model_id: self.pool.get(model_id).embedding_dim()
            for model_id in self.settings.active_models
        }

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
            if self._service is not None:
                self._service.close()
            self.pool.close()
            self.close_library()


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
