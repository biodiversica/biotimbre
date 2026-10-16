"""Copying text to the clipboard from a click.

The copy runs in the browser, inside the click itself: browsers only let a page
write to the clipboard in answer to a gesture, and a round trip to the server
first can outlast that. The clipboard API exists only in a secure context
(HTTPS or localhost), and the app is just as often opened over plain HTTP from
another machine on the network, so the old ``execCommand("copy")`` route is
tried when the API is missing or refuses. The outcome comes back to the server
only to say so: a copy that failed shows the text instead, to copy by hand.
"""

from __future__ import annotations

import json

from nicegui import ui
from nicegui.events import GenericEventArguments

from biotimbre.i18n import Translator

#: Copies ``text`` and emits whether it worked. ``%s`` is the text as a JSON
#: string literal.
_COPY_JS = """() => {
    const text = %s;
    const fallback = () => {
        const area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        let copied = false;
        try { copied = document.execCommand("copy"); } catch (e) {}
        area.remove();
        emit(copied);
    };
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => emit(true), fallback);
    } else {
        fallback();
    }
}"""


def copy_on_click(t: Translator, element: ui.element, text: str) -> ui.element:
    """Make a click on ``element`` copy ``text``, and say whether it did.

    Args:
        t: Active translator.
        element: What is clicked.
        text: What is copied.

    Returns:
        ``element``, for chaining.
    """

    def report(event: GenericEventArguments) -> None:
        if event.args is True:
            ui.notify(t("common.copied", text=text), type="positive", position="top")
        else:
            ui.notify(
                t("common.copy_failed", text=text),
                type="warning",
                position="top",
                multi_line=True,
                timeout=0,
                close_button=t("common.close"),
            )

    return element.classes("cursor-pointer").on(
        "click", report, js_handler=_COPY_JS % json.dumps(text)
    )
