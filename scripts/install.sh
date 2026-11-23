#!/bin/sh
# Install biotimbre and a launcher for its browser interface, on macOS or Linux.
#
#   curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | sh
#
# Installs uv if it is missing (uv brings its own Python, so none is needed
# beforehand), installs biotimbre as an isolated tool, and adds a launcher that
# opens the interface from a work folder. Run it again to upgrade.
#
# To uninstall, run it with BIOTIMBRE_UNINSTALL set:
#
#   curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | BIOTIMBRE_UNINSTALL=1 sh
#
# Environment overrides:
#   BIOTIMBRE_PACKAGE    what to install                 (default: biotimbre[gui])
#   BIOTIMBRE_WORKDIR    folder the interface starts in  (default: ~/biotimbre)
#   BIOTIMBRE_UNINSTALL  1 removes biotimbre and its launcher; all also removes
#                        the downloaded models and the interface's remembered
#                        settings. Libraries are never removed, nor is uv.
set -eu

PACKAGE="${BIOTIMBRE_PACKAGE:-biotimbre[gui]}"
WORKDIR="${BIOTIMBRE_WORKDIR:-$HOME/biotimbre}"
UNINSTALL="${BIOTIMBRE_UNINSTALL:-}"
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

# Where the launcher goes, and where the interface remembers its settings
# (biotimbre.ui.app.user_state_dir).
if [ "$OS" = macos ]; then
  LAUNCHER="$HOME/Desktop/biotimbre.command"
  STATE="$HOME/Library/Application Support/biotimbre"
else
  APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  LAUNCHER="$APPS/biotimbre.desktop"
  STATE="${XDG_STATE_HOME:-$HOME/.local/state}/biotimbre"
fi

# ── uninstall ────────────────────────────────────────────────────────────────
remove() {
  if [ -e "$1" ]; then
    rm -rf "$1"
    say "Removed $2 ($1)"
  fi
}

if [ -n "$UNINSTALL" ]; then
  case "$UNINSTALL" in
    1|all) ;;
    *) die "BIOTIMBRE_UNINSTALL must be 1 or all, not '$UNINSTALL'." ;;
  esac
  if command -v uv >/dev/null 2>&1; then
    UV=$(command -v uv)
  else
    UV="$HOME/.local/bin/uv"
  fi
  if [ -x "$UV" ] && "$UV" tool list 2>/dev/null | grep -q '^biotimbre '; then
    say "Uninstalling biotimbre"
    "$UV" tool uninstall biotimbre
  fi
  remove "$LAUNCHER" launcher
  if [ "$UNINSTALL" = all ]; then
    remove "$STATE" "remembered settings"
    # Only biotimbre's own models: the cache is shared with other programs.
    HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
    for name in models--biodiversica--Perch-onnx-backbone models--biodiversica--BirdNET-onnx-backbone; do
      remove "$HUB/$name" model
      remove "$HUB/.locks/$name" "model lock"
    done
  fi
  say "Done. biotimbre is uninstalled."
  cat <<EOF

  Libraries were left where they are, including any in $WORKDIR.
  Delete them yourself if you no longer want them.
EOF
  exit 0
fi

# ── uv ───────────────────────────────────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
else
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

# ── launcher ─────────────────────────────────────────────────────────────────
# Both open a terminal: it shows the address and is where Ctrl+C stops the
# interface.
if [ "$OS" = macos ]; then
  mkdir -p "$HOME/Desktop"
  cat > "$LAUNCHER" <<EOF
#!/bin/sh
cd "$WORKDIR" && exec "$BIOTIMBRE" gui
EOF
  chmod +x "$LAUNCHER"
  WHERE="double-click biotimbre on your Desktop"
else
  mkdir -p "$APPS"
  cat > "$LAUNCHER" <<EOF
[Desktop Entry]
Type=Application
Name=biotimbre
Comment=Build and search libraries of animal sounds in the browser
Exec="$BIOTIMBRE" gui
Path=$WORKDIR
Terminal=true
Icon=audio-x-generic
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
