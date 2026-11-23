#!/bin/sh
# Install biotimbre and a launcher for its browser interface, on macOS or Linux.
#
#   curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | sh
#
# Installs uv if it is missing (uv brings its own Python, so none is needed
# beforehand), installs biotimbre as an isolated tool, and adds a launcher that
# opens the interface from a work folder. Run it again to upgrade.
#
# It says what it is about to do and asks before doing any of it.
#
# Environment overrides:
#   BIOTIMBRE_PACKAGE   what to install                 (default: biotimbre[gui])
#   BIOTIMBRE_WORKDIR   folder the interface starts in  (default: ~/biotimbre)
#   BIOTIMBRE_YES=1     install without asking, e.g. from a script or CI
set -eu

PACKAGE="${BIOTIMBRE_PACKAGE:-biotimbre[gui]}"
WORKDIR="${BIOTIMBRE_WORKDIR:-$HOME/biotimbre}"
# The version CI tests on. Pinning it keeps a newer system Python, which
# onnxruntime may not publish wheels for yet, from being picked.
PYTHON=3.12

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  Darwin) OS=macos ;;
  Linux) OS=linux ;;
  *) die "unsupported system $(uname -s); on Windows use install.ps1." ;;
esac

if command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
else
  UV=
fi

# ── confirmation ─────────────────────────────────────────────────────────────
if [ "$OS" = macos ]; then
  LAUNCHER_TEXT="biotimbre.command on your Desktop"
else
  LAUNCHER_TEXT="a biotimbre entry in your applications menu, with its icon"
fi
if [ -n "$UV" ]; then
  UV_TEXT="use uv, already installed at $UV"
else
  UV_TEXT="install uv, the tool that installs biotimbre, into ~/.local/bin"
fi
cat <<EOF
This installs biotimbre, a tool for building and searching libraries of
animal sounds, with its interface in the browser. It will:

  - $UV_TEXT
  - install $PACKAGE with uv, isolated, along with a Python $PYTHON
    of its own; your other Pythons are not touched (if biotimbre is already
    installed, it is upgraded)
  - add uv's tools folder to your PATH, in your shell's startup file
  - create the folder $WORKDIR, where the interface starts
  - add $LAUNCHER_TEXT

The first start of the interface downloads the models (about 70 MB), once.

EOF
if [ "${BIOTIMBRE_YES:-}" != 1 ]; then
  # Asked of the terminal: under "curl ... | sh" standard input is the script.
  if ! (: </dev/tty) 2>/dev/null; then
    die "there is no terminal to ask in; set BIOTIMBRE_YES=1 to install without asking."
  fi
  printf 'Continue? [y/N] '
  read -r ANSWER </dev/tty || ANSWER=
  case "$ANSWER" in
    [yY] | [yY][eE][sS]) ;;
    *) echo "Nothing was installed."; exit 0 ;;
  esac
fi

# ── uv ───────────────────────────────────────────────────────────────────────
if [ -z "$UV" ]; then
  say "Installing uv"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
  else
    die "neither curl nor wget is available to download uv."
  fi
  UV="$HOME/.local/bin/uv"
fi

# ── biotimbre ────────────────────────────────────────────────────────────────
say "Installing $PACKAGE"
"$UV" tool install --python "$PYTHON" --upgrade "$PACKAGE"
"$UV" tool update-shell >/dev/null 2>&1 || true
BIOTIMBRE="$("$UV" tool dir --bin)/biotimbre"
[ -x "$BIOTIMBRE" ] || die "biotimbre was not installed where uv said it would be ($BIOTIMBRE)."

mkdir -p "$WORKDIR"

# The logo, from the tool's environment. A package from before the logo has
# none, and the launcher then keeps the default icon.
ASSETS=
for dir in "$("$UV" tool dir)"/biotimbre/lib/python*/site-packages/biotimbre/assets; do
  if [ -f "$dir/biotimbre.png" ]; then ASSETS="$dir"; fi
done

# ── launcher ─────────────────────────────────────────────────────────────────
# Both open a terminal: it shows the address and is where Ctrl+C stops the
# interface.
if [ "$OS" = macos ]; then
  LAUNCHER="$HOME/Desktop/biotimbre.command"
  mkdir -p "$HOME/Desktop"
  cat > "$LAUNCHER" <<EOF
#!/bin/sh
cd "$WORKDIR" && exec "$BIOTIMBRE" gui
EOF
  chmod +x "$LAUNCHER"
  # A .command file has no icon setting of its own; Finder shows the one
  # attached to the file, which keeps its own copy of the image.
  if [ -n "$ASSETS" ]; then
    osascript -l JavaScript -e '
      function run(argv) {
        ObjC.import("AppKit");
        var image = $.NSImage.alloc.initWithContentsOfFile(argv[0]);
        $.NSWorkspace.sharedWorkspace.setIconForFileOptions(image, argv[1], 0);
      }' "$ASSETS/biotimbre.png" "$LAUNCHER" >/dev/null 2>&1 || true
  fi
  WHERE="double-click biotimbre on your Desktop"
else
  APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  LAUNCHER="$APPS/biotimbre.desktop"
  mkdir -p "$APPS"
  # Copied into the user's icon theme, which the launcher names: the tool's
  # environment may be rebuilt on an upgrade.
  ICON=audio-x-generic
  if [ -n "$ASSETS" ]; then
    ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"
    mkdir -p "$ICONS/scalable/apps" "$ICONS/512x512/apps"
    cp "$ASSETS/biotimbre.svg" "$ICONS/scalable/apps/biotimbre.svg"
    cp "$ASSETS/biotimbre.png" "$ICONS/512x512/apps/biotimbre.png"
    touch "$ICONS"  # so menus notice the theme changed
    ICON=biotimbre
  fi
  cat > "$LAUNCHER" <<EOF
[Desktop Entry]
Type=Application
Name=biotimbre
Comment=Build and search libraries of animal sounds in the browser
Exec="$BIOTIMBRE" gui
Path=$WORKDIR
Terminal=true
Icon=$ICON
Categories=Science;
EOF
  WHERE="open biotimbre from your applications menu"
fi

say "Done. $("$BIOTIMBRE" --version 2>/dev/null || echo biotimbre) is installed."
cat <<EOF

  To start: $WHERE,
  or run 'biotimbre gui' in a new terminal.

  The interface starts in $WORKDIR, where it offers to create or open a
  library. The first start downloads the models (about 70 MB), once.
  To upgrade later, run this installer again.
EOF
