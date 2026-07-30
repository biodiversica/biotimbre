"""Web interface.

Requires the ``gui`` extra::

    uv sync --extra gui

Start it with ``biotimbre gui``, or programmatically::

    from biotimbre.ui.app import run
    run()
"""

from biotimbre.ui.state import AppState, get_state, set_state

__all__ = ["AppState", "get_state", "set_state"]
