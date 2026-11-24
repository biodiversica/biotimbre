"""Downloading an item: its audio, or its embeddings as ``.npy``.

Embeddings can come with the audio each window was computed from, cut and
resampled as that model was given it, for a tool that wants to check what a
vector describes or to feed the same input to something else.

The same control serves the library list and the search results — an item is
the same thing wherever it was found, and so is the question "give me this one
as a file". Filenames match what ``biotimbre export-npy`` writes, so a download
and a command-line export of the same rows are interchangeable.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from nicegui import ui

from biotimbre.domain.errors import BiotimbreError
from biotimbre.i18n import Translator
from biotimbre.service import audio_filename, bundle_filename
from biotimbre.ui.layout import notify_error
from biotimbre.ui.state import get_state


async def _download_audio(t: Translator, item_id: int) -> None:
    """Send the stored audio to the browser."""
    state = get_state()
    try:
        path, mime = await state.run(state.service.get_audio, item_id)
    except BiotimbreError as exc:
        notify_error(t, exc)
        return
    ui.download.file(path, audio_filename(item_id, path.suffix), mime)


async def _download_bundle(t: Translator, item_id: int) -> None:
    """Send a shareable bundle of one item to the browser.

    The whole item — metadata, audio and vectors — rather than one of its
    parts, because this is the download meant for another person rather than
    for a tool.
    """
    state = get_state()
    try:
        data, name = await state.run(state.service.export_bundle_bytes, [item_id])
    except BiotimbreError as exc:
        notify_error(t, exc)
        return
    ui.download.content(data, name, "application/zip")


async def _download_embeddings(
    t: Translator, item_id: int, model_id: str, *, with_audio: bool = False
) -> None:
    """Send one model's embeddings to the browser, one file per window."""
    state = get_state()
    try:
        data, name = await state.run(
            state.service.embeddings_export,
            item_id,
            model_id,
            with_audio=with_audio,
        )
    except BiotimbreError as exc:
        notify_error(t, exc)
        return
    mime = "application/zip" if name.endswith(".zip") else "application/octet-stream"
    ui.download.content(data, name, mime)


def export_button(t: Translator, item_id: int, counts: dict[str, int]) -> ui.button:
    """A download button offering the audio and each model's embeddings.

    Args:
        t: Active translator.
        item_id: Item to export.
        counts: Embedding count per model, as
            :meth:`~biotimbre.db.repository.Repository.embedding_counts`
            returns. Models absent from it are not offered: an item that was
            never embedded with a model has nothing to hand over.

    Returns:
        The button, so a caller can place or style it further.
    """
    button = ui.button(t("export.label"), icon="download").props(
        "flat dense no-caps"
    )
    with button:
        with ui.menu():
            ui.menu_item(
                t("export.audio"),
                on_click=lambda: _download_audio(t, item_id),
            )
            # The one entry here that is for a person rather than a program:
            # everything the receiving library needs to file this sample under
            # the same identity it has here.
            ui.menu_item(
                t("export.bundle"),
                on_click=lambda: _download_bundle(t, item_id),
            )
            ui.separator()
            for model, count in sorted(counts.items()):
                # The label names the artefact: a window per file means several
                # windows arrive as an archive, not as one .npy.
                key = "export.embeddings_one" if count == 1 else "export.embeddings_many"
                ui.menu_item(
                    t(key, model=model, count=count),
                    on_click=lambda model=model: _download_embeddings(
                        t, item_id, model
                    ),
                )
                # Only for a model whose preprocessing is known: the clips are
                # cut again from the audio, the way that model cuts them.
                try:
                    spec = get_state().service.settings.model_spec(model)
                except KeyError:
                    spec = None
                if spec is not None:
                    ui.menu_item(
                        t(
                            "export.embeddings_with_audio",
                            model=model,
                            rate=spec.sample_rate,
                        ),
                        on_click=lambda model=model: _download_embeddings(
                            t, item_id, model, with_audio=True
                        ),
                    )
            if not counts:
                ui.menu_item(t("export.no_embeddings")).props("disable")
    return button


# ── several items as one bundle ───────────────────────────────────────────────

#: Where multi-item bundles are written before the browser fetches them, made
#: on first use. A bundle of many items can be hundreds of megabytes, which is
#: too much to build in memory and hand over as content.
_EXPORT_DIR: Path | None = None

#: How long a written bundle is kept for the browser to fetch.
_EXPORT_TTL_S = 600.0


def _export_path(name: str) -> Path:
    """A fresh path for a bundle about to be written, clearing stale ones."""
    global _EXPORT_DIR
    if _EXPORT_DIR is None:
        _EXPORT_DIR = Path(tempfile.mkdtemp(prefix="biotimbre-exports-"))
        atexit.register(shutil.rmtree, _EXPORT_DIR, True)
    now = time.time()
    for old in _EXPORT_DIR.iterdir():
        try:
            if now - old.stat().st_mtime > _EXPORT_TTL_S:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    folder = _EXPORT_DIR / uuid4().hex[:12]
    folder.mkdir()
    return folder / name


async def download_bundle(
    t: Translator, item_ids: list[int], *, include_embeddings: bool = True
) -> bool:
    """Send a bundle of several items to the browser.

    Written to a file rather than built in memory, then served from it.

    Returns:
        Whether it was sent.
    """
    state = get_state()
    target = _export_path(bundle_filename(item_ids))
    try:
        path = await state.run(
            state.service.export_bundle,
            item_ids,
            target,
            include_embeddings=include_embeddings,
        )
    except BiotimbreError as exc:
        notify_error(t, exc)
        return False
    ui.download.file(path, path.name, "application/zip")
    return True


class SelectionBar:
    """What to do with the items ticked in a list: for now, share them.

    Hidden while nothing is ticked, and showing how many are, so what is about
    to be exported can be seen before it is.

    Args:
        t: Active translator.
        on_clear: Unticks everything in the list the bar belongs to.
    """

    def __init__(self, t: Translator, *, on_clear: Callable[[], None]):
        self._t = t
        self._ids: list[int] = []
        self._on_clear = on_clear
        self.row = ui.row().classes(
            "items-center gap-2 w-full px-3 py-1 rounded bg-surface"
        ).mark("selection-bar")
        with self.row:
            self._count = ui.label().classes("text-sm")
            self._export = ui.button(
                icon="inventory_2", on_click=self._export_clicked
            ).props("flat dense no-caps").mark("export-selected-bundle")
            self._embeddings = ui.switch(
                t("export.with_embeddings"), value=True
            ).props("dense").tooltip(t("export.with_embeddings_help"))
            ui.button(
                t("export.clear_selection"), icon="clear", on_click=self.clear
            ).props("flat dense no-caps")
        self.update([])

    @property
    def item_ids(self) -> list[int]:
        """The items ticked, in the list's order."""
        return list(self._ids)

    def update(self, item_ids: list[int]) -> None:
        """Show the bar for these items, or hide it for none."""
        self._ids = list(item_ids)
        count = len(self._ids)
        self._count.set_text(self._t("export.selected", count=count))
        self._export.set_text(self._t("export.bundle_selected", count=count))
        self.row.set_visibility(bool(count))

    def clear(self) -> None:
        """Untick everything."""
        self._on_clear()
        self.update([])

    async def _export_clicked(self) -> None:
        if not self._ids:
            return
        if await download_bundle(
            self._t, self._ids, include_embeddings=bool(self._embeddings.value)
        ):
            self.clear()
