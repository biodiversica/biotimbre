"""Playing a sample at the speed the models hear it.

The models reinterpret the samples at ``speed`` times their rate, so time and
pitch move together: ultrasound slowed to 0.25 drops two octaves into hearing.
A browser preserves pitch by default when its rate changes, which is the
opposite, so every player here switches that off. The browser's own speed
menu is hidden for the same reason — it would bring pitch preservation back,
and give the player a second speed that disagrees with the page.
"""

from __future__ import annotations

from nicegui import ui

from biotimbre.i18n import Translator

#: Rates a browser will play at. Chromium refuses anything outside this, so a
#: speed beyond it is heard at the nearest end rather than not at all.
PLAYABLE_SPEEDS = (0.0625, 16.0)


def speed_audio(source: str) -> ui.audio:
    """A player without the browser's own speed menu, for a speed to steer."""
    return ui.audio(source).props('controlslist="noplaybackrate"')


def play_at_speed(player: ui.audio, speed: float) -> float:
    """Set the player's rate to ``speed``, with the pitch moving along.

    The default rate is set as well: loading a new source resets the rate to
    it.

    Returns:
        The rate actually played at, which differs from ``speed`` only when
        the browser cannot go that far.
    """
    low, high = PLAYABLE_SPEEDS
    rate = min(max(speed, low), high)
    ui.run_javascript(
        f"""
        const el = getHtmlElement({player.id});
        if (el) {{
            el.preservesPitch = false;
            el.mozPreservesPitch = false;
            el.webkitPreservesPitch = false;
            el.defaultPlaybackRate = {rate};
            el.playbackRate = {rate};
        }}
        """
    )
    return rate


def play_described(
    t: Translator, player: ui.audio, note: ui.label, speed: float
) -> None:
    """Play at ``speed`` and say so, or say how far short the browser falls."""
    rate = play_at_speed(player, speed)
    if rate != speed:
        note.set_text(
            t("playback.clamped", rate=f"{rate:g}", speed=f"{speed:g}")
        )
    elif speed != 1.0:
        note.set_text(t("playback.speed", speed=f"{speed:g}"))
    else:
        note.set_text("")


def item_player(t: Translator, source: str, speed: float) -> ui.audio:
    """A stored item's player, heard at its embedded speed or as recorded.

    The choice is offered only for an item embedded at a speed other than 1:
    otherwise both would sound the same.
    """
    player = speed_audio(source)
    heard_at = None
    if speed != 1.0:
        heard_at = ui.toggle(
            {
                speed: t("playback.embedded", speed=f"{speed:g}"),
                1.0: t("playback.original"),
            },
            value=speed,
        ).props("dense no-caps")
    note = ui.label().classes("text-xs text-muted")

    def play_as_chosen() -> None:
        play_described(t, player, note, heard_at.value if heard_at else speed)

    player.on("loadedmetadata", lambda _: play_as_chosen())
    if heard_at is not None:
        heard_at.on_value_change(lambda _: play_as_chosen())
    return player
