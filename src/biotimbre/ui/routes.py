"""HTTP routes for audio and spectrogram images.

Audio is served through :meth:`~biotimbre.service.LibraryService.get_audio` rather
than by mounting the audio directory as static files. That keeps one chokepoint to
enforce access control if the search tool is ever exposed publicly (PLAN.md D8),
and means a request can only ever name an ``item_id`` — never a path.

Spectrograms are a route rather than inline data URIs so the browser caches them:
a result list re-rendered on a filter change reuses the images it already has.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Response
from fastapi.responses import FileResponse

from biotimbre.config import DEFAULT_DB_NAME
from biotimbre.domain.errors import BiotimbreError, NotFoundError
from biotimbre.embeddings import audio as audio_io
from biotimbre.ui.state import get_state
from biotimbre.viz import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    render_spectrogram,
)

#: Bounds on requested image size, so a crafted query cannot ask the server to
#: render something enormous.
MIN_DIMENSION = 16
MAX_WIDTH = 2400
MAX_HEIGHT = 800

#: Where the library chooser reads directories from.
BROWSE_ROUTE = "/setup/browse"


def holds_library(directory: Path) -> bool:
    """Whether a directory already holds a library.

    A directory that cannot be read is still a place to go — /lost+found is on
    the way to an external drive's contents — so an unreadable one answers "not
    that I can tell" rather than refusing the whole listing.
    """
    try:
        return (directory / DEFAULT_DB_NAME).is_file()
    except OSError:
        return False


def list_directory(path: str | None = None) -> dict:
    """What is in one directory, for the library chooser.

    Only subdirectories: a library is a directory, and listing files would
    expose more than the question asks.

    Args:
        path: Directory to list. ``None`` starts at the reader's home.

    Returns:
        The resolved path, its parent, the home directory, and the
        subdirectories — each flagged if it already holds a library.

    Raises:
        NotFoundError: If the path is not a directory.
        BiotimbreError: If it cannot be read.
    """
    target = Path(path).expanduser() if path else Path.home()
    try:
        target = target.resolve()
        if not target.is_dir():
            raise NotFoundError(f"No such directory: {target}")
        entries = sorted(target.iterdir(), key=lambda entry: entry.name.lower())
    except PermissionError as exc:
        raise BiotimbreError(f"Permission denied: {target}") from exc
    except OSError as exc:
        raise BiotimbreError(f"Could not read {target}: {exc}") from exc

    directories = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:  # a broken symlink is not a place to go
            continue
        directories.append(
            {
                "name": entry.name,
                "path": str(entry),
                # Already a library, so the chooser can say "open this" rather
                # than "make one here".
                "is_library": holds_library(entry),
            }
        )

    return {
        "path": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "home": str(Path.home()),
        "directories": directories,
        "is_library": holds_library(target),
    }


def register_browser(app) -> None:
    """Attach the directory listing the library chooser needs.

    A browser cannot hand the server a path — a file input reports names, never
    locations — so choosing where a library lives means the server listing its
    own directories. That is the one place in this interface where a request
    names a filesystem path, which is why it is mounted only while no library
    is open and withdrawn by :func:`unregister_browser` as soon as one is.

    Libraries live wherever their owner keeps them — an external drive, a
    mounted archive — so the listing is not confined to a root. The protection
    is the loopback bind, as it is for everything else here.

    Args:
        app: The application to register on.
    """

    @app.get(BROWSE_ROUTE)
    async def browse(path: str | None = None):
        """List one directory's subdirectories, for the chooser."""
        state = get_state()
        if state.is_open:
            raise HTTPException(status_code=403, detail="A library is already open.")
        try:
            return await state.run(list_directory, path)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BiotimbreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


def unregister_browser(app) -> None:
    """Withdraw the directory listing once a library is open."""
    app.remove_route(BROWSE_ROUTE)


def _normalised_wav(path: Path, start: float | None, end: float | None) -> bytes:
    """Encode a selection, normalised if it goes beyond full scale, as WAV."""
    import io

    from scipy.io import wavfile

    samples, rate = audio_io.load_mono(path, start_time=start, end_time=end)
    samples, _ = audio_io.normalise_over_range(samples)
    buffer = io.BytesIO()
    # scipy, not soundfile: soundfile reaches a buffer through cffi callbacks,
    # which fail outright where libffi does not match what cffi was built for.
    # float32 is written as IEEE-float WAV, which browsers play.
    wavfile.write(buffer, rate, samples)
    return buffer.getvalue()


def register(app) -> None:
    """Attach the media routes to a FastAPI application.

    Args:
        app: The application to register on (NiceGUI's ``app`` is one).
    """

    @app.get("/media/audio/{item_id}")
    async def audio(item_id: int):
        """Stream an item's stored audio."""
        state = get_state()
        try:
            path, mime = await state.run(state.service.get_audio, item_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BiotimbreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return FileResponse(
            path,
            media_type=mime,
            # Named after the item so a download is identifiable, and marked
            # immutable because stored audio is never modified in place (D1b).
            filename=f"biotimbre_item_{item_id}{path.suffix}",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    @app.get("/media/spectrogram/{item_id}")
    async def spectrogram(
        item_id: int,
        w: int = DEFAULT_WIDTH,
        h: int = DEFAULT_HEIGHT,
        start: float | None = None,
        end: float | None = None,
    ):
        """Render an item's spectrogram as a PNG."""
        state = get_state()
        width = max(MIN_DIMENSION, min(MAX_WIDTH, w))
        height = max(MIN_DIMENSION, min(MAX_HEIGHT, h))

        try:
            path, _ = await state.run(state.service.get_audio, item_id)
            image = await state.run(
                render_spectrogram,
                path,
                start_time=start,
                end_time=end,
                width=width,
                height=height,
                colors=state.settings.theme.spectrogram_colors,
                mark_dominant=state.settings.dominant_frequency,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BiotimbreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(
            content=image.png,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    @app.get("/media/query/{token}")
    async def query_audio(token: str):
        """Stream a staged query file by token.

        Tokens are opaque and registry-backed, so an unknown one is a 404 rather
        than a chance to name a path.
        """
        state = get_state()
        path = state.queries.path(token)
        if path is None:
            raise HTTPException(status_code=404, detail="Unknown query token.")
        return FileResponse(
            path,
            media_type="audio/wav",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/media/query/{token}/normalised")
    async def query_audio_normalised(
        token: str, start: float | None = None, end: float | None = None
    ):
        """Stream a query's selection, brought within full scale.

        For a selection that goes beyond it, which the browser would clip on
        playback. The selection alone, not the file scaled by its gain: it is at
        most the query cap long, and a whole file could be an hour. Normalised
        exactly as the search normalises it, so what is heard is what is
        searched with.
        """
        state = get_state()
        path = state.queries.path(token)
        if path is None:
            raise HTTPException(status_code=404, detail="Unknown query token.")
        try:
            wav = await state.run(_normalised_wav, path, start, end)
        except BiotimbreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/media/query/{token}/spectrogram")
    async def query_spectrogram(
        token: str,
        w: int = DEFAULT_WIDTH,
        h: int = DEFAULT_HEIGHT,
        start: float | None = None,
        end: float | None = None,
    ):
        """Render a staged query file's spectrogram."""
        state = get_state()
        path = state.queries.path(token)
        if path is None:
            raise HTTPException(status_code=404, detail="Unknown query token.")

        width = max(MIN_DIMENSION, min(MAX_WIDTH, w))
        height = max(MIN_DIMENSION, min(MAX_HEIGHT, h))
        try:
            image = await state.run(
                render_spectrogram,
                path,
                start_time=start,
                end_time=end,
                width=width,
                height=height,
                colors=state.settings.theme.spectrogram_colors,
                mark_dominant=state.settings.dominant_frequency,
            )
        except BiotimbreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(
            content=image.png,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/media/status")
    async def status():
        """Model warm-up state, polled by the page while models load.

        ``items`` is ``None`` until a library is open: a first run starts at
        the chooser, and the models warm up there all the same.
        """
        state = get_state()
        return {
            "ready": state.models.ready,
            "failed": state.models.failed,
            "dimensions": state.models.dimensions,
            "items": (
                await state.run(state.service.repo.count_items)
                if state.is_open
                else None
            ),
        }
