"""Upload handling against NiceGUI's real event contract.

These exist because of a bug that reached the browser: the handlers were written
for a NiceGUI 2.x-style ``event.content``, but 3.x passes ``event.file``, a
:class:`FileUpload` with *async* ``read()``/``save()``. The handler raised
``AttributeError`` on every upload, which was logged server-side and never shown,
so the page looked like it had simply ignored the file.

Nothing here mocks the contract loosely. The stub subclasses NiceGUI's own
abstract :class:`FileUpload`, so if a future version changes those methods the
stub fails to instantiate and these tests break — which is the point.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui.elements.upload_files import FileUpload  # noqa: E402

from biotimbre.ui.admin import AddState  # noqa: E402
from biotimbre.ui.state import QueryStore  # noqa: E402
from conftest import EXAMPLES_DIR  # noqa: E402

GUYVIR = EXAMPLES_DIR / "GUYVIR.wav"


class StubFileUpload(FileUpload):
    """A FileUpload backed by bytes, implementing the real abstract interface."""

    def __init__(self, name: str, data: bytes, content_type: str = "audio/wav"):
        self.name = name
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data

    async def text(self, encoding: str = "utf-8") -> str:
        return self._data.decode(encoding)

    def iterate(self, *, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
        async def generator() -> AsyncIterator[bytes]:
            for start in range(0, len(self._data), chunk_size):
                yield self._data[start : start + chunk_size]

        return generator()

    async def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self._data)

    def size(self) -> int:
        return len(self._data)


class TestContractHolds:
    def test_stub_satisfies_nicegui_abstract_interface(self):
        # If NiceGUI renames or re-signatures these, instantiation fails here
        # rather than silently at runtime in a browser.
        upload = StubFileUpload("q.wav", b"abc")
        assert isinstance(upload, FileUpload)

    def test_event_exposes_file_not_content(self):
        import dataclasses

        from nicegui.events import UploadEventArguments

        names = {f.name for f in dataclasses.fields(UploadEventArguments)}
        # The original bug in one assertion.
        assert "file" in names
        assert "content" not in names

    async def test_file_name_and_read_are_the_accessors(self):
        upload = StubFileUpload("GUYVIR.wav", GUYVIR.read_bytes())
        assert upload.name == "GUYVIR.wav"
        assert await upload.read() == GUYVIR.read_bytes()


class TestAddStateStreaming:
    async def test_begin_then_save_then_finish(self):
        add_state = AddState()
        upload = StubFileUpload("GUYVIR.wav", GUYVIR.read_bytes())

        target = add_state.begin(upload.name)
        await upload.save(target)
        staged = add_state.finish(target, upload.name)

        assert staged.path.read_bytes() == GUYVIR.read_bytes()
        assert staged.properties.sample_rate == 32_000
        assert add_state.upload is staged
        add_state.discard()

    async def test_begin_discards_the_previous_upload(self):
        add_state = AddState()
        first = add_state.begin("a.wav")
        await StubFileUpload("a.wav", GUYVIR.read_bytes()).save(first)
        add_state.finish(first, "a.wav")

        second = add_state.begin("b.wav")
        # The previous file goes as soon as a new upload starts, so a staging
        # directory cannot accumulate abandoned files.
        assert not first.exists()
        await StubFileUpload("b.wav", GUYVIR.read_bytes()).save(second)
        add_state.finish(second, "b.wav")
        add_state.discard()

    async def test_undecodable_upload_is_rejected_at_finish(self):
        add_state = AddState()
        upload = StubFileUpload("bad.wav", b"not audio")
        target = add_state.begin(upload.name)
        await upload.save(target)

        from biotimbre.domain.errors import ValidationError

        with pytest.raises(ValidationError):
            add_state.finish(target, upload.name)
        assert add_state.upload is None

    async def test_begin_uses_the_basename_only(self):
        add_state = AddState()
        target = add_state.begin("../../escape.wav")
        assert target.parent == add_state.directory
        assert target.name == "escape.wav"

    def test_replace_still_works_for_byte_callers(self):
        add_state = AddState()
        staged = add_state.replace("GUYVIR.wav", GUYVIR.read_bytes())
        assert staged.properties.duration_s == pytest.approx(5.0)
        add_state.discard()


class TestQueryStoreStreaming:
    async def test_reserve_then_save(self):
        store = QueryStore()
        upload = StubFileUpload("q.wav", GUYVIR.read_bytes())

        token, target = store.reserve(upload.name)
        await upload.save(target)

        assert store.path(token) == target
        assert target.read_bytes() == GUYVIR.read_bytes()
        store.clear()

    async def test_reserved_but_unwritten_token_resolves_to_none(self):
        store = QueryStore()
        token, _ = store.reserve("q.wav")
        # Reserving does not create the file, so a failed upload cannot leave a
        # token pointing at nothing usable.
        assert store.path(token) is None
        store.clear()

    async def test_reserve_evicts_the_oldest(self):
        store = QueryStore()
        tokens = []
        for index in range(QueryStore.MAX_ENTRIES + 1):
            token, target = store.reserve(f"q{index}.wav")
            await StubFileUpload(f"q{index}.wav", GUYVIR.read_bytes()).save(target)
            tokens.append(token)
        assert store.path(tokens[0]) is None
        assert store.path(tokens[-1]) is not None
        store.clear()

    def test_add_still_works_for_byte_callers(self):
        store = QueryStore()
        token = store.add("q.wav", GUYVIR.read_bytes())
        assert store.path(token) is not None
        store.clear()
