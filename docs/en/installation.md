# Installation

*English · [Português (Brasil)](../pt-BR/installation.md)*

**Without Python.** The installer sets up everything required and adds a
**biotimbre** shortcut that opens the [web interface](web-interface.md).
Running it again upgrades the installation.

On Windows, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex"
```

On macOS and Linux, open a terminal and run:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | sh
```

The installer first lists what it is going to do and asks for confirmation
before changing anything. It then performs the following steps:

1. Installs [uv](https://docs.astral.sh/uv/) into `~/.local/bin` if it is not
   already present. uv downloads its own Python, so no prior installation is
   required, and any existing Python installation is left unchanged.
2. On Windows, installs the current Microsoft Visual C++ Redistributable (x64)
   if the one present is older than 14.40, the version the models need.
   Windows asks for an administrator's permission for this step.
3. Runs `uv tool install --python 3.12 "biotimbre[gui]"`, which installs
   biotimbre with its web interface and adds `biotimbre` to the PATH.
4. Creates a working folder, `~/biotimbre`, in which the interface starts and
   offers to create the first library.
5. Adds a **biotimbre** launcher that runs `biotimbre gui` in that folder: a
   Desktop and Start menu shortcut on Windows, `biotimbre.command` on the
   macOS Desktop, and an applications menu entry on Linux. The launcher opens a
   terminal window next to the browser. This window runs the server, and
   closing it (or pressing Ctrl+C) stops the server.

Two environment variables change the installer's defaults: `BIOTIMBRE_PACKAGE`
(the package to install, for example `biotimbre` without the interface) and
`BIOTIMBRE_WORKDIR` (the working folder). `BIOTIMBRE_YES=1` skips the
confirmation, for installing or uninstalling from a script.

**Uninstalling.** Run the installer again with `BIOTIMBRE_UNINSTALL` set. It
lists what it is going to remove and asks for confirmation first. On Windows,
close the biotimbre window first, then run in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c '$env:BIOTIMBRE_UNINSTALL=1; irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex'
```

On macOS and Linux:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | BIOTIMBRE_UNINSTALL=1 sh
```

This removes biotimbre and its launcher. With `BIOTIMBRE_UNINSTALL=all`
instead of `1`, it also removes the downloaded models and what the interface
remembers (language, theme, recent libraries). Libraries are never removed,
and neither is uv, which other programs may use. A plain
`uv tool uninstall biotimbre` removes only the program, leaving a launcher
that no longer works: delete it by hand (on Windows, **biotimbre** on the
Desktop and in the Start menu).

**With uv or pip.** The web interface is an optional extra, so a headless or
command-line-only installation stays small:

```bash
uv tool install --python 3.12 "biotimbre[gui]"   # with the web interface
uv tool install --python 3.12 biotimbre          # command line only
pip install "biotimbre[gui]"                     # or with pip, into the current environment
```

On Windows, the models need a current
[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
(14.40 or newer). Without it, biotimbre still runs, but the models fail to
load and the interface and command line say why.

`uv tool upgrade biotimbre` upgrades and `uv tool uninstall biotimbre`
uninstalls. To work on biotimbre itself, see [Development](development.md); from
a source checkout, prefix the commands in this document with `uv run`.

**First run.** The first start downloads the Perch and BirdNET models
(approximately 70 MB) and caches them; no network connection is required
afterwards. A library can be created from the interface, or from the command
line:

```bash
biotimbre init --name "My Library"
```

The `--name` option is required the first time a library is initialised. Every
library has a name and a UUID, generated once and stable for its lifetime, so
that it remains identifiable after being shared: copying the data directory
transfers both. The name and UUID are shown in the interface header and by
`biotimbre stats`. Running `init --name "..."` again renames the library.
