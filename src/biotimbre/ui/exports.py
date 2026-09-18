"""Downloading an item: its audio, or its embeddings as ``.npy``.

The same control serves the library list and the search results — an item is
the same thing wherever it was found, and so is the question "give me this one
as a file". Filenames match what ``biotimbre export-npy`` writes, so a download
and a command-line export of the same rows are interchangeable.
"""

from __future__ import annotations

from nicegui import ui

from biotimbre.domain.errors import BiotimbreError
from biotimbre.i18n import Translator
from biotimbre.service import audio_filename
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


async def _download_embeddings(t: Translator, item_id: int, model_id: str) -> None:
    """Send one model's embeddings to the browser, one file per window."""
    state = get_state()
    try:
        data, name = await state.run(
            state.service.embeddings_export, item_id, model_id
        )
    except BiotimbreError as exc:
        notify_error(t, exc)
        return
    ui.download.content(data, name, "application/octet-stream")


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
            if not counts:
                ui.menu_item(t("export.no_embeddings")).props("disable")
    return button
