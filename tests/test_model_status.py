"""How the page tells the reader about the models: passing notices, not the header."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.ui.state import AppState, ModelStatus, get_state, set_state  # noqa: E402

#: Longer than the page's poll interval, so one poll has run.
POLL_WAIT = 2.0


@pytest.fixture
def installed_state(settings, shared_pool):
    """Install app state backed by a temporary library, models not yet loaded."""
    state = AppState(settings)
    state.pool = shared_pool
    state.service._pool = shared_pool
    state.service._owns_pool = False

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


def notices(user: User) -> list[ui.notification]:
    """The notices on the page. Empty when there are none, where find() raises."""
    try:
        return list(user.find(ui.notification).elements)
    except AssertionError:
        return []


def options(notice: ui.notification) -> dict:
    return notice.props["options"]


async def test_nothing_is_said_when_the_models_are_ready(
    user: User, installed_state: AppState
):
    installed_state.models = ModelStatus(ready=True, dimensions={"perch_v2": 1536})
    await user.open("/search")
    await asyncio.sleep(POLL_WAIT)
    assert notices(user) == []


async def test_loading_turns_into_ready_and_leaves(
    user: User, installed_state: AppState
):
    await user.open("/search")
    [notice] = notices(user)
    assert options(notice)["message"] == "Loading models…"
    assert options(notice)["spinner"] is True
    assert options(notice)["timeout"] == 0  # stays while loading

    installed_state.models = ModelStatus(ready=True, dimensions={"perch_v2": 1536})
    await asyncio.sleep(POLL_WAIT)
    assert options(notice)["message"] == "Models ready"
    assert options(notice)["caption"] == "perch_v2 1536d"
    assert options(notice)["spinner"] is False
    assert options(notice)["type"] == "positive"
    assert options(notice)["timeout"] > 0  # leaves by itself


async def test_a_failure_stays_until_closed(user: User, installed_state: AppState):
    await user.open("/search")
    [notice] = notices(user)
    installed_state.models = ModelStatus(failed="no network")
    await asyncio.sleep(POLL_WAIT)
    assert "no network" in options(notice)["message"]
    assert options(notice)["type"] == "negative"
    assert options(notice)["spinner"] is False
    assert options(notice)["closeBtn"] is True
    assert options(notice)["timeout"] == 0
    assert "caption" not in options(notice)


async def test_a_failure_before_the_page_opens_is_shown(
    user: User, installed_state: AppState
):
    installed_state.models = ModelStatus(failed="no network")
    await user.open("/search")
    [notice] = notices(user)
    assert "no network" in options(notice)["message"]
    assert options(notice)["timeout"] == 0


async def test_the_header_has_no_model_status(user: User, installed_state: AppState):
    installed_state.models = ModelStatus(ready=True, dimensions={"perch_v2": 1536})
    await user.open("/search")
    await user.should_see("((-)) biotimbre")
    await user.should_not_see("Models ready")
