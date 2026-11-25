# ((-)) biotimbre

*English · [Português (Brasil)](https://github.com/biodiversica/biotimbre/blob/main/README_pt-BR.md)*

Biotimbre is a tool for building and managing libraries of animal sound
templates, with similarity search based on audio embeddings. Embeddings are
computed with ONNX backbones through `onnxruntime`; Perch v2 and BirdNET 2.4 are
supported by default.

Each library stores the recordings, their metadata and their embedding vectors
in a single folder. Libraries can be used through a web interface or from the
command line, searched by sound and by metadata, exported for analysis, and
shared item by item with other libraries.

## Installation

**Without Python.** The installer sets up everything required and adds a
**biotimbre** shortcut that opens the web interface. Running it again upgrades
the installation.

On Windows, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex"
```

On macOS and Linux, open a terminal and run:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | sh
```

The installer lists what it is going to do and asks for confirmation before
changing anything.

**Uninstalling.** Run the installer again with `BIOTIMBRE_UNINSTALL` set. On
Windows, close the biotimbre window first, then run in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c '$env:BIOTIMBRE_UNINSTALL=1; irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex'
```

On macOS and Linux:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | BIOTIMBRE_UNINSTALL=1 sh
```

Libraries are never removed.

**With uv or pip.** The web interface is an optional extra, so a headless or
command-line-only installation stays small:

```bash
uv tool install --python 3.12 "biotimbre[gui]"   # with the web interface
uv tool install --python 3.12 biotimbre          # command line only
pip install "biotimbre[gui]"                     # or with pip, into the current environment
```

[Installation](https://github.com/biodiversica/biotimbre/blob/main/docs/en/installation.md)
describes what the installer does, its environment variables, what
uninstalling removes, and the Visual C++ Redistributable the models need on
Windows.

## Quick Start

```bash
biotimbre init --name "My Library"           # create a library in ./data
biotimbre gui                                # the web interface, at http://127.0.0.1:8080
biotimbre add call.wav --sidecar call.txt    # add a recording with its metadata
biotimbre search query.wav -k 5              # the five most similar items
```

The first start downloads the Perch and BirdNET models (approximately 70 MB)
and caches them; no network connection is required afterwards. `biotimbre
--help` lists every command, and `biotimbre COMMAND --help` explains each of
its options in full, with examples.

## Documentation

| Page | Contents |
|---|---|
| [Installation](https://github.com/biodiversica/biotimbre/blob/main/docs/en/installation.md) | The installer, uninstalling, uv and pip, the first run |
| [Libraries](https://github.com/biodiversica/biotimbre/blob/main/docs/en/libraries.md) | Where a library is stored, opening one, the data layout |
| [Web Interface](https://github.com/biodiversica/biotimbre/blob/main/docs/en/web-interface.md) | The Library and Search pages, search-only mode, choosing a library, the map, display options |
| [Command-Line Interface](https://github.com/biodiversica/biotimbre/blob/main/docs/en/cli.md) | Every command, with examples |
| [Metadata](https://github.com/biodiversica/biotimbre/blob/main/docs/en/metadata.md) | Fields, identifiers, one biotimbre per species, sidecar files, tables |
| [Audio Preparation](https://github.com/biodiversica/biotimbre/blob/main/docs/en/audio-preparation.md) | Trimming, filtering and normalisation; ultrasonic recordings and the speed setting |
| [Search Behaviour](https://github.com/biodiversica/biotimbre/blob/main/docs/en/search.md) | The query limit, filters, ranking by one model |
| [Exporting Data](https://github.com/biodiversica/biotimbre/blob/main/docs/en/exporting.md) | Audio, embeddings and metadata tables |
| [Sharing Items Between Libraries](https://github.com/biodiversica/biotimbre/blob/main/docs/en/sharing.md) | Bundles, import rules, importing from another library |
| [iNaturalist Integration](https://github.com/biodiversica/biotimbre/blob/main/docs/en/inaturalist.md) | Adding and searching with an observation's recording |
| [Configuration](https://github.com/biodiversica/biotimbre/blob/main/docs/en/configuration.md) | `biotimbre.toml`, with every setting and its default |
| [Adding a Model](https://github.com/biodiversica/biotimbre/blob/main/docs/en/adding-a-model.md) | Custom ONNX embedding models |
| [Development](https://github.com/biodiversica/biotimbre/blob/main/docs/en/development.md) | Working on biotimbre itself, tests and CI |

## License

[MIT](https://github.com/biodiversica/biotimbre/blob/main/LICENSE)
