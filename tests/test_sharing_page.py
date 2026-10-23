"""The bundle and library importers in the interface.

The dialog's job is to make four decisions visible before they are acted on,
so most of what is tested here is that flipping a control changes what the
list says — and that what the list said is what the import did.

Driven through NiceGUI's simulated ``user`` fixture, so no browser is needed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.config import Settings  # noqa: E402
from biotimbre.domain.item import (  # noqa: E402
    ItemDraft,
    Location,
    generate_biotimbre_id,
)
from biotimbre.models_registry import ModelSpec  # noqa: E402
from biotimbre.service import LibraryService  # noqa: E402
from biotimbre.ui.state import AppState, get_state, set_state  # noqa: E402
from conftest import EXAMPLES_DIR, unit_vectors  # noqa: E402

GUYVIR = EXAMPLES_DIR / "GUYVIR.wav"
ZAMTYM = EXAMPLES_DIR / "ZAMTYM.wav"

FAKE_MODEL = "fake_model"
FAKE_DIM = 16
FAKE_SPEC = ModelSpec(
    model_id=FAKE_MODEL,
    repo_id="test/fake",
    filename="fake.onnx",
    sample_rate=32_000,
    window_duration_s=5.0,
    window_overlap=0.5,
    description="test double",
)


def _offline_settings(data_dir) -> Settings:
    """Settings for a library that never runs inference.

    ``active_models`` is empty, so nothing is ever computed and no model file
    is ever wanted. The fake model is registered by hand instead, which is all
    the fingerprint comparison needs — and the decisions this page is about are
    the ones that happen around the vectors, not the arithmetic inside them.
    """
    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "library.db",
        audio_dir=data_dir / "audio",
        active_models=(),
    )


@pytest.fixture
def receiver(tmp_path):
    """Install app state for the library that receives bundles."""
    settings = _offline_settings(tmp_path / "receiver")
    state = AppState(settings)
    # AppState.open_library reloads settings from the usual places, which puts
    # the default active_models back; the pool was built from the ones passed
    # in. Pin both halves to the same answer.
    state.settings = settings
    state.service.settings = settings
    state.service.rename("Receiver")
    with state.service._transaction():
        state.service.repo.register_model(FAKE_SPEC, FAKE_DIM)

    previous = None
    try:
        previous = get_state()
    except RuntimeError:
        pass
    set_state(state)
    yield state
    state.close()
    if previous is not None:
        set_state(previous)


@pytest.fixture
def sender(tmp_path):
    """An independent library to build bundles from."""
    settings = _offline_settings(tmp_path / "sender")
    service = LibraryService(settings)
    service.rename("Sender's cicadas")
    with service._transaction():
        service.repo.register_model(FAKE_SPEC, FAKE_DIM)
    yield service
    service.close()


def add(service: LibraryService, audio=GUYVIR, **fields):
    """Add one item with no embeddings, returning it."""
    draft = ItemDraft(
        date=fields.pop("date", "2020-01-12"),
        time=fields.pop("time", "09:40:21"),
        location=fields.pop("location", Location(-26.0, -48.0)),
        **fields,
    )
    return service.add_item(audio, draft, compute_embeddings=False).item


def embed(service: LibraryService, item_id: int, n: int = 3):
    """Place vectors for the fake model, as an inference pass would have."""
    vectors = unit_vectors(n, FAKE_DIM)
    rows = [(i * 2.5, i * 2.5 + 5.0, vectors[i]) for i in range(n)]
    with service._transaction():
        service.repo.set_embeddings(item_id, FAKE_MODEL, rows)


def _control(user: User, marker: str):
    """One of the dialog's policy controls, named by the decision it makes.

    By marker rather than by caption: the captions are translated, and a test
    that breaks when someone improves a sentence is testing the wrong thing.
    """
    matches = list(user.find(marker=marker).elements)
    assert len(matches) == 1, f"expected one {marker!r}, found {len(matches)}"
    return matches[0]


def _import_button(user: User):
    """The dialog's Import button, as an element filter that can be clicked."""
    return user.find(marker="bundle-import")


async def upload_bundle(user: User, path) -> None:
    """Hand a bundle to the page's importer, as a person choosing a file would."""
    from nicegui.elements.upload_files import SmallFileUpload

    uploader = next(iter(user.find(marker="upload-bundle").elements))
    await uploader.handle_uploads(
        [SmallFileUpload(path.name, "application/zip", path.read_bytes())]
    )


@pytest.mark.usefixtures("receiver")
class TestImportCard:
    async def test_the_page_offers_the_importer(self, user: User):
        await user.open("/")
        await user.should_see("Import items")
        await user.should_see("Choose a bundle")

    async def test_the_tab_is_titled_with_the_library_name(self, user: User):
        await user.open("/")
        assert user.client.title == "Receiver"

    async def test_the_importer_accepts_only_bundles(self, user: User):
        await user.open("/")
        uploader = next(iter(user.find(marker="upload-bundle").elements))
        assert uploader._props["accept"] == ".btz"

    async def test_the_two_uploaders_are_distinguishable(self, user: User):
        # The audio form and the importer both take files; each says which.
        await user.open("/")
        assert len(user.find(marker="upload-audio").elements) == 1
        assert len(user.find(marker="upload-bundle").elements) == 1


@pytest.mark.usefixtures("receiver")
class TestPreviewDialog:
    async def test_it_says_where_the_bundle_came_from(
        self, user: User, sender, tmp_path
    ):
        item = add(sender, taxa="cicadidae")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        await user.open("/")
        await upload_bundle(user, path)

        await user.should_see("From Sender's cicadas")
        await user.should_see("1 item(s)")
        await user.should_see("cicadidae")

    async def test_it_shows_what_would_happen_and_does_nothing(
        self, user: User, sender, receiver, tmp_path
    ):
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        await user.open("/")
        await upload_bundle(user, path)
        await user.should_see("Would add 1, skip 0.")

        # Looking is not importing.
        assert receiver.service.stats()["items"] == 0

    async def test_it_offers_all_four_decisions(self, user: User, sender, tmp_path):
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        await user.open("/")
        await upload_bundle(user, path)

        await user.should_see("What to do")
        await user.should_see("Use the sender's embeddings")
        await user.should_see("Compute what the bundle does not supply")
        await user.should_see("A recording already here")
        await user.should_see("A species already here")

    async def test_importing_adds_the_item(
        self, user: User, sender, receiver, tmp_path
    ):
        item = add(sender, taxa="cicadidae")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        await user.open("/")
        await upload_bundle(user, path)
        await user.should_see("Would add 1, skip 0.")
        _import_button(user).click()
        await user.should_see("Added 1, skipped 0.", retries=20)

        assert receiver.service.stats()["items"] == 1
        stored = receiver.service.list_items()[0]
        # The identity travelled, which is the point of the whole feature.
        assert stored.biotimbre_id == item.biotimbre_id

    async def test_a_bundle_with_nothing_new_cannot_be_imported(
        self, user: User, sender, receiver, tmp_path
    ):
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")
        receiver.call(receiver.service.import_bundle, path)  # nothing to embed

        await user.open("/")
        await upload_bundle(user, path)

        await user.should_see("Would add 0, skip 1.")
        await user.should_see("already in this library")
        button = next(iter(_import_button(user).elements))
        assert not button.enabled

    async def test_a_bad_file_is_reported(self, user: User, tmp_path):
        broken = tmp_path / "broken.btz"
        broken.write_bytes(b"not a zip")

        await user.open("/")
        await upload_bundle(user, broken)
        await user.should_see("Not a readable bundle", retries=20)


async def preview_library_import(user: User, data_dir) -> None:
    """Name another library and ask for the import preview."""
    path_input = _control(user, "library-path")
    path_input.set_value(str(data_dir))
    user.find(marker="library-preview").click()


@pytest.mark.usefixtures("receiver")
class TestImportLibrary:
    async def test_the_page_offers_the_library_import(self, user: User):
        await user.open("/")
        await user.should_see("Import from another library")
        await user.should_see("Preview import")

    async def test_it_previews_every_item_and_does_nothing(
        self, user: User, sender, receiver
    ):
        add(sender, taxa="cicadidae")
        add(sender, ZAMTYM, taxa="orthoptera")

        await user.open("/")
        await preview_library_import(user, sender.settings.data_dir)

        await user.should_see("From Sender's cicadas", retries=20)
        await user.should_see("2 item(s)")
        await user.should_see("Would add 2, skip 0.")
        assert receiver.service.stats()["items"] == 0

    async def test_importing_adds_every_item(self, user: User, sender, receiver):
        first = add(sender, taxa="cicadidae")
        add(sender, ZAMTYM, taxa="orthoptera")

        await user.open("/")
        await preview_library_import(user, sender.settings.data_dir)
        await user.should_see("Would add 2, skip 0.", retries=20)
        _import_button(user).click()
        await user.should_see("Added 2, skipped 0.", retries=20)

        assert receiver.service.stats()["items"] == 2
        assert first.biotimbre_id in {
            item.biotimbre_id for item in receiver.service.list_items()
        }
        # The other library is untouched.
        assert sender.stats()["items"] == 2

    async def test_missing_audio_is_reported(self, user: User, sender):
        gone = add(sender)
        add(sender, ZAMTYM)
        sender.settings.absolute_audio_path(gone.audio_path).unlink()

        await user.open("/")
        await preview_library_import(user, sender.settings.data_dir)
        await user.should_see("have no audio on disk", retries=20)
        await user.should_see("Would add 1, skip 0.")

    async def test_a_directory_without_a_library_is_reported(
        self, user: User, tmp_path
    ):
        (tmp_path / "nothing").mkdir()

        await user.open("/")
        await preview_library_import(user, tmp_path / "nothing")
        await user.should_see("No biotimbre library", retries=20)

    async def test_browsing_lists_libraries(self, user: User, sender, tmp_path):
        await user.open("/")
        _control(user, "library-path").set_value(str(tmp_path))
        user.find("Browse").click()

        await user.should_see("sender", retries=20)
        await user.should_see("library")


@pytest.mark.usefixtures("receiver")
class TestPolicyControlsChangeThePlan:
    """Flipping a control re-plans, so the list is the consequence."""

    async def test_keeping_the_senders_biotimbre_changes_the_outcome(
        self, user: User, sender, receiver, tmp_path
    ):
        item = add(sender, taxa="guyalna bonaerensis")
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")
        mine = add(receiver.service, ZAMTYM, taxa="guyalna bonaerensis")

        await user.open("/")
        await upload_bundle(user, path)

        # By default one species keeps one biotimbre: the local one.
        await user.should_see("joins it rather than adding a second identity")
        await user.should_see(mine.biotimbre_id[:8])

        _control(user, "policy-species").set_value("keep_incoming")
        await user.should_see("two identities for one animal", retries=20)

    async def test_distrusting_the_sender_refuses_their_vectors(
        self, user: User, sender, tmp_path
    ):
        item = add(sender)
        embed(sender, item.item_id, n=3)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")

        await user.open("/")
        await upload_bundle(user, path)
        await user.should_see("3 window(s), reusable")

        _control(user, "policy-trust").set_value(False)
        await user.should_see("not reused", retries=20)

    async def test_a_duplicate_can_be_let_in(
        self, user: User, sender, receiver, tmp_path
    ):
        item = add(sender)
        path = sender.export_bundle([item.item_id], tmp_path / "share.btz")
        add(receiver.service)  # the same recording, catalogued independently

        await user.open("/")
        await upload_bundle(user, path)
        await user.should_see("Would add 0, skip 1.")

        _control(user, "policy-duplicates").set_value("add")
        await user.should_see("Would add 1, skip 0.", retries=20)
