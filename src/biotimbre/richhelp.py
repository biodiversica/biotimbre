"""Typer-styled ``--help`` output and error reporting for the argparse CLI.

Ported from spectro-calendar's ``richhelp`` module so the two tools' ``-h``
look the same: a padded usage line, the description, and one rounded panel per
group of parameters, in the palette of ``typer.rich_utils``. On top of that
port, this one knows about what a multi-command CLI needs:

* subcommands, listed in a *Commands* panel — or in several, when a command's
  parser was made with ``panel="..."``;
* argparse argument groups, each drawn as a panel of its own, so a command with
  many options can show them in related sets;
* an epilog, printed below the panels with its line breaks kept, which is where
  the examples go.

argparse's ``formatter_class`` hook cannot express panels, so the entry point
is :class:`RichHelpParser` instead: it overrides ``format_help`` and leaves
argument parsing untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import IO, Any, NoReturn

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, RenderableType
from rich.highlighter import RegexHighlighter
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Styles, copied from typer.rich_utils so the CLIs read identically.
STYLE_OPTION = "bold cyan"
STYLE_SWITCH = "bold green"
STYLE_NEGATIVE_OPTION = "bold magenta"
STYLE_NEGATIVE_SWITCH = "bold red"
STYLE_TYPES = "bold yellow"
STYLE_TYPES_SEPARATOR = "dim"
STYLE_USAGE = "yellow"
STYLE_USAGE_COMMAND = "bold"
STYLE_HELPTEXT_FIRST_LINE = ""
STYLE_HELPTEXT = "dim"
STYLE_OPTION_HELP = ""
STYLE_OPTION_DEFAULT = "dim"
STYLE_REQUIRED_SHORT = "red"
STYLE_REQUIRED_LONG = "dim red"
STYLE_OPTIONS_PANEL_BORDER = "dim"
ALIGN_OPTIONS_PANEL = "left"
STYLE_OPTIONS_TABLE_PADDING = (0, 1)
STYLE_COMMANDS_TABLE_FIRST_COLUMN = "bold cyan"
STYLE_EPILOG = "dim"
STYLE_ERROR = "red"
STYLE_WARNING = "yellow"
STYLE_ERRORS_PANEL_BORDER = "red"
ALIGN_ERRORS_PANEL = "left"
STYLE_ERRORS_SUGGESTION = "dim"

DEFAULT_STRING = "[default: {}]"
REQUIRED_SHORT_STRING = "*"
REQUIRED_LONG_STRING = "[required]"
ARGUMENTS_PANEL_TITLE = "Arguments"
OPTIONS_PANEL_TITLE = "Options"
COMMANDS_PANEL_TITLE = "Commands"
ERRORS_PANEL_TITLE = "Error"
RICH_HELP = "Try [blue]'{command_path} {help_option}'[/] for help."

#: Titles argparse gives its two built-in groups, across Python versions.
_POSITIONAL_GROUP_TITLES = ("positional arguments",)
_OPTIONAL_GROUP_TITLES = ("options", "optional arguments")


class OptionHighlighter(RegexHighlighter):
    """Colours option flags and ``<type>`` markers wherever they appear."""

    highlights = [
        r"(^|\W)(?P<switch>\-\w+)(?![a-zA-Z0-9])",
        r"(^|\W)(?P<option>\-\-[\w\-]+)(?![a-zA-Z0-9])",
        r"(?P<types>\<[^\>]+\>)",
        r"(?P<usage>Usage: )",
    ]


class NegativeOptionHighlighter(RegexHighlighter):
    """Colours the ``--no-x`` half of a paired boolean flag."""

    highlights = [
        r"(^|\W)(?P<negative_switch>\-\w+)(?![a-zA-Z0-9])",
        r"(^|\W)(?P<negative_option>\-\-[\w\-]+)(?![a-zA-Z0-9])",
    ]


class TypesHighlighter(RegexHighlighter):
    """Dims the brackets around a type so only the type name stands out."""

    highlights = [
        r"^(?P<types_sep>(\[|<))",
        r"(?P<types_sep>\|)",
        r"(?P<types_sep>(\]|>))(\.\.\.)?$",
    ]


highlighter = OptionHighlighter()
negative_highlighter = NegativeOptionHighlighter()
types_highlighter = TypesHighlighter()


def _console(stderr: bool = False, file: IO[str] | None = None) -> Console:
    return Console(
        file=file,
        theme=Theme(
            {
                "option": STYLE_OPTION,
                "switch": STYLE_SWITCH,
                "negative_option": STYLE_NEGATIVE_OPTION,
                "negative_switch": STYLE_NEGATIVE_SWITCH,
                "types": STYLE_TYPES,
                "types_sep": STYLE_TYPES_SEPARATOR,
                "usage": STYLE_USAGE,
            },
        ),
        highlighter=highlighter,
        stderr=stderr,
    )


def _is_subcommands(action: argparse.Action) -> bool:
    return isinstance(action, argparse._SubParsersAction)


def _takes_value(action: argparse.Action) -> bool:
    """False for flags like ``--yes`` or ``-h``, which consume no argument."""
    return action.nargs != 0


def _type_name(action: argparse.Action) -> str:
    """The type marker for an action: ``[a|b]`` for choices, else ``<type>``.

    An explicit metavar wins over the type's name, since it is the more
    specific of the two: ``<YYYY-MM-DD>`` says more than ``<str>``.
    """
    if action.choices is not None:
        return "[" + "|".join(str(choice) for choice in action.choices) + "]"
    if action.metavar and action.option_strings:
        name = str(action.metavar)
    else:
        t = action.type
        if t is None:
            name = "text"
        elif t is Path:
            name = "path"
        else:
            name = getattr(t, "__name__", str(t))
    marker = f"<{name}>"
    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        marker += "..."
    return marker


def _metavar(action: argparse.Action) -> str:
    if action.metavar and not action.option_strings:
        return str(action.metavar)
    return (action.dest or "").upper()


def _is_required(action: argparse.Action) -> bool:
    if action.option_strings:
        return bool(action.required)
    return action.nargs not in ("?", "*")


def _default_text(action: argparse.Action) -> str:
    """Typer shows a ``[default: ...]`` tag, but not for empty or boolean ones."""
    default = action.default
    if default is None or default is argparse.SUPPRESS or isinstance(default, bool):
        return ""
    if isinstance(default, (list, tuple)):
        return ", ".join(str(x) for x in default)
    return str(default)


def _expand_help(parser: argparse.ArgumentParser, action: argparse.Action) -> str:
    """Apply argparse's own ``%(default)s``-style substitutions to a help string."""
    if not action.help or action.help is argparse.SUPPRESS:
        return ""
    params = {k: v for k, v in vars(action).items() if v is not argparse.SUPPRESS}
    params["prog"] = parser.prog
    for name, value in list(params.items()):
        if hasattr(value, "__name__"):
            params[name] = value.__name__
    if params.get("choices") is not None:
        params["choices"] = ", ".join(str(c) for c in params["choices"])
    try:
        return action.help % params
    except (TypeError, ValueError, KeyError):
        return action.help


def _prose(text: str) -> str:
    """Reflow a help string: one line per paragraph, paragraphs kept apart."""
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
    return "\n".join(p for p in paragraphs if p)


def _help_cell(parser: argparse.ArgumentParser, action: argparse.Action) -> Columns:
    """The right-hand cell: help prose, then the default and required tags."""
    items: list[RenderableType] = []
    text = _expand_help(parser, action)
    if text:
        items.append(highlighter(Text(_prose(text), style=STYLE_OPTION_HELP)))
    default = _default_text(action)
    if default:
        items.append(Text(DEFAULT_STRING.format(default), style=STYLE_OPTION_DEFAULT))
    if _is_required(action):
        items.append(Text(REQUIRED_LONG_STRING, style=STYLE_REQUIRED_LONG))
    return Columns(items)


def _table() -> Table:
    return Table(
        highlight=True,
        show_header=False,
        expand=True,
        box=None,
        border_style=None,
        row_styles=None,
        pad_edge=False,
        padding=STYLE_OPTIONS_TABLE_PADDING,
        leading=0,
        show_lines=False,
    )


def _print_panel(console: Console, title: str, body: RenderableType) -> None:
    console.print(
        Panel(
            body,
            border_style=STYLE_OPTIONS_PANEL_BORDER,
            title=title,
            title_align=ALIGN_OPTIONS_PANEL,
            box=box.ROUNDED,
        )
    )


def _panel(
    parser: argparse.ArgumentParser,
    name: str,
    actions: list[argparse.Action],
    console: Console,
) -> None:
    """Render one group of parameters as a rounded panel, Typer-style."""
    rows: list[list[RenderableType]] = []
    required_flags: list[RenderableType] = []
    for action in actions:
        long_strs = [o for o in action.option_strings if o.startswith("--")]
        short_strs = [o for o in action.option_strings if not o.startswith("--")]
        secondary_long: list[str] = []
        if not action.option_strings:  # a positional: its name goes in the short column
            short_strs = [_metavar(action)]
        else:
            # BooleanOptionalAction pairs --flag with --no-flag; show the negative
            # half in its own column, as Typer does for `--x/--no-x`.
            negatives = [o for o in long_strs if o.startswith("--no-")]
            if negatives and len(long_strs) > len(negatives):
                secondary_long = negatives
                long_strs = [o for o in long_strs if o not in negatives]

        types = Text(style=STYLE_TYPES, overflow="fold")
        if _takes_value(action):
            types.append(_type_name(action))

        required_flags.append(
            Text(REQUIRED_SHORT_STRING, style=STYLE_REQUIRED_SHORT)
            if _is_required(action)
            else ""
        )
        rows.append(
            [
                highlighter(",".join(long_strs)),
                highlighter(",".join(short_strs)),
                negative_highlighter(",".join(secondary_long)),
                types_highlighter(types),
                _help_cell(parser, action),
            ]
        )

    if not rows:
        return
    if any(bool(flag) for flag in required_flags):
        rows = [[flag, *row] for flag, row in zip(required_flags, rows)]

    table = _table()
    for row in rows:
        table.add_row(*row)
    _print_panel(console, name, table)


def _command_panels(
    action: argparse._SubParsersAction,
) -> dict[str, list[tuple[str, str]]]:
    """Subcommands as ``{panel title: [(name, help), ...]}``, in order given."""
    panels: dict[str, list[tuple[str, str]]] = {}
    for choice in action._choices_actions:
        name = choice.dest
        subparser = action.choices.get(name)
        title = getattr(subparser, "panel", None) or COMMANDS_PANEL_TITLE
        panels.setdefault(title, []).append((name, choice.help or ""))
    return panels


def _commands(action: argparse._SubParsersAction, console: Console) -> None:
    """Render the subcommands, one panel per ``panel=`` title."""
    for title, commands in _command_panels(action).items():
        table = _table()
        table.add_column(style=STYLE_COMMANDS_TABLE_FIRST_COLUMN, no_wrap=True)
        table.add_column(ratio=1)
        for name, help_text in commands:
            table.add_row(name, highlighter(Text(_prose(help_text))))
        _print_panel(console, title, table)


def _usage(parser: argparse.ArgumentParser, positionals: list[argparse.Action]) -> str:
    """``Usage: prog [OPTIONS] REQUIRED [OPTIONAL]``, as Click writes it."""
    pieces = [parser.prog, "[OPTIONS]"]
    for action in positionals:
        if _is_subcommands(action):
            pieces.append("COMMAND [ARGS]...")
            continue
        metavar = _metavar(action)
        pieces.append(metavar if _is_required(action) else f"[{metavar}]")
        if action.nargs in ("*", "+"):
            pieces[-1] += "..."
    return "Usage: " + " ".join(pieces)


def _description(parser: argparse.ArgumentParser) -> Align:
    """First paragraph plain, the rest dim — Typer's treatment of the docstring."""
    text = (parser.description or "").strip()
    first, *rest = text.split("\n\n")
    parts: list[RenderableType] = [
        highlighter(Text(" ".join(first.split()), style=STYLE_HELPTEXT_FIRST_LINE))
    ]
    if rest:
        parts.append(Text(""))
        body = "\n\n".join(" ".join(p.split()) for p in rest)
        parts.append(highlighter(Text(body, style=STYLE_HELPTEXT)))
    return Align(_stack(parts), pad=False)


def _stack(parts: list[RenderableType]) -> Table:
    """Lay renderables out one per line, the way Rich's @group decorator does."""
    grid = Table.grid()
    grid.add_column()
    for part in parts:
        grid.add_row(part)
    return grid


class RichHelpParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose ``-h`` prints Typer's Rich help screen.

    Args:
        panel: For a subcommand's parser, the title of the panel its parent's
            help lists it under. Commands without one go under "Commands".
    """

    def __init__(self, *args: Any, panel: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.panel = panel
        for action in self._actions:
            if isinstance(action, argparse._HelpAction):
                action.help = "Show this message and exit."  # Typer's wording

    def format_help(self) -> str:
        console = _console()
        with console.capture() as capture:
            self._render_help(console)
        return capture.get()

    def print_help(self, file: IO[str] | None = None) -> None:
        # Not argparse's default of writing format_help() to the file: a
        # captured string is raw ANSI, which the classic Windows console
        # (cmd/PowerShell in conhost) prints as "←[1m" junk. Printing through
        # Rich lets it drive that console with Win32 calls instead.
        self._render_help(_console(file=file))

    def _groups(self) -> list[tuple[str, list[argparse.Action]]]:
        """Each argument group as ``(panel title, actions)``, in display order.

        Arguments first, then the groups a command defined for itself, then
        the remaining options — with ``--help`` last, where Typer puts it.
        """
        arguments: list[argparse.Action] = []
        options: list[argparse.Action] = []
        named: list[tuple[str, list[argparse.Action]]] = []
        for group in self._action_groups:
            actions = [
                a
                for a in group._group_actions
                if not _is_subcommands(a) and a.help is not argparse.SUPPRESS
            ]
            if group.title in _POSITIONAL_GROUP_TITLES:
                arguments.extend(actions)
            elif group.title in _OPTIONAL_GROUP_TITLES:
                options.extend(actions)
            elif actions:
                named.append((group.title or OPTIONS_PANEL_TITLE, actions))
        options.sort(key=lambda a: isinstance(a, argparse._HelpAction))
        return [
            (ARGUMENTS_PANEL_TITLE, arguments),
            *named,
            (OPTIONS_PANEL_TITLE, options),
        ]

    def _render_help(self, console: Console) -> None:
        positionals = [a for a in self._actions if not a.option_strings]
        console.print(
            Padding(highlighter(_usage(self, positionals)), 1), style=STYLE_USAGE_COMMAND
        )
        if self.description:
            console.print(Padding(_description(self), (0, 1, 1, 1)))
        for title, actions in self._groups():
            _panel(self, title, actions, console)
        for action in positionals:
            if _is_subcommands(action):
                _commands(action, console)
        if self.epilog:
            # Kept line for line: the epilog holds examples, which must not be
            # reflowed into a paragraph.
            console.print(
                Padding(highlighter(Text(self.epilog.strip("\n"), style=STYLE_EPILOG)), 1)
            )

    def error(self, message: str) -> NoReturn:
        """Report a usage error the way Typer does: usage, a hint, a red panel."""
        console = _console(stderr=True)
        positionals = [a for a in self._actions if not a.option_strings]
        console.print(highlighter(_usage(self, positionals)), style=STYLE_USAGE_COMMAND)
        help_option = self._help_option()
        if help_option:
            console.print(
                RICH_HELP.format(
                    command_path=escape(self.prog), help_option=help_option
                ),
                style=STYLE_ERRORS_SUGGESTION,
            )
        console.print(
            Panel(
                highlighter(message),
                border_style=STYLE_ERRORS_PANEL_BORDER,
                title=ERRORS_PANEL_TITLE,
                title_align=ALIGN_ERRORS_PANEL,
                box=box.ROUNDED,
            )
        )
        self.exit(2)

    def _help_option(self) -> str:
        """The flag to suggest in the hint line, or "" when help is disabled."""
        for action in self._actions:
            if isinstance(action, argparse._HelpAction) and action.option_strings:
                return action.option_strings[0]
        return ""


def print_error(message: str) -> None:
    """Write ``error: message`` to stderr, in red.

    The message is passed as ``Text`` so brackets in a path are not read as
    markup, and printed with ``soft_wrap`` so Rich does not break it into lines
    of its own: a terminal wraps it anyway, and a log keeps it whole.
    """
    _console(stderr=True).print(
        Text(f"error: {message}", style=STYLE_ERROR), soft_wrap=True, highlight=False
    )


def print_warning(message: str) -> None:
    """Write ``warning: message`` to stderr, in yellow."""
    _console(stderr=True).print(
        Text(f"warning: {message}", style=STYLE_WARNING), soft_wrap=True, highlight=False
    )
