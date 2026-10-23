## Install

No Python needed: the installer sets everything up and adds a **biotimbre** shortcut that opens the interface in your browser. Run it again to upgrade.

**Windows**: open PowerShell and paste:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | iex"
```

**macOS and Linux**: open a terminal and paste:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | sh
```

Already using Python? `uv tool install --python 3.12 "biotimbre[gui]"` or `pip install "biotimbre[gui]"`. See the [README](https://github.com/biodiversica/biotimbre#installation) ([português](https://github.com/biodiversica/biotimbre/blob/main/README_pt-BR.md#instalação)).

The first start downloads the Perch and BirdNET models (about 70 MB), once.

## What's in this release

<!-- Write the highlights here before publishing. -->
