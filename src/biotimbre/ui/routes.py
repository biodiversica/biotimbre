"""HTTP routes for audio and spectrogram images.

Audio is served through :meth:`~biotimbre.service.LibraryService.get_audio` rather
than by mounting the audio directory as static files. That keeps one chokepoint to
enforce access control if the search tool is ever exposed publicly (PLAN.md D8),
and means a request can only ever name an ``item_id`` — never a path.

Spectrograms are a route rather than inline data URIs so the browser caches them:
a result list re-rendered on a filter change reuses the images it already has.
"""

from __future__ import annotations

from fastapi import HTTPException, Response
from fastapi.responses import FileResponse

from biotimbre.domain.errors import BiotimbreError, NotFoundError
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
        """Model warm-up state, polled by the page while models load."""
        state = get_state()
        return {
            "ready": state.models.ready,
            "failed": state.models.failed,
            "dimensions": state.models.dimensions,
            "items": await state.run(state.service.repo.count_items),
        }
