## Install

No Python needed: the installer sets everything up and adds a **biotimbre** shortcut that opens the interface in your browser. Run it again to upgrade.

**Windows**: open PowerShell and paste:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex"
```

**macOS and Linux**: open a terminal and paste:

```sh
curl -LsSf https://github.com/biodiversica/biotimbre/releases/latest/download/install.sh | sh
```

Already using Python? `uv tool install --python 3.12 "biotimbre[gui]"` or `pip install "biotimbre[gui]"`. See the [README](https://github.com/biodiversica/biotimbre#installation) ([português](https://github.com/biodiversica/biotimbre/blob/main/README_pt-BR.md#instalação)).

The first start downloads the Perch and BirdNET models (about 70 MB), once. To uninstall, run the installer with `BIOTIMBRE_UNINSTALL=1` (see [Uninstalling](https://github.com/biodiversica/biotimbre#installation)); your libraries are kept.

## What's in this release

### Changes to check before upgrading

- **Search ranks by one model.** Each model's similarity scores have a baseline and spread of their own, so combining them let the model whose scores run higher decide almost every ranking. Results are now ranked by one model, **Perch v2 by default**, and each result also shows the other models' scores and the rank each gave it. `search --fusion` is now `--rank-by MODEL_ID`, and `search.fusion` in `biotimbre.toml` is now `search.rank_by`: a configuration file that still sets `fusion` is refused at startup, with the replacement named. In the results CSV, `best_model` is now `ranked_by`, with a `rank_<model>` column beside each `score_<model>`.
- **The Search page searches every speed by default.** *Query speed* starts empty, which embeds the query at every speed in the library; before, it was always 1, so items stored at another speed were never found from the page. A query speed, on the page or with `search --query-speed`, now searches only the items stored at it, in both interfaces, and the command line says when no item is stored at it.
- **Recordings are always stored as WAV.** A FLAC, OGG, MP3 or AIFF recording is converted when it is added, keeping every sample of a lossless one. Items already in a library keep their files.
- **Renamed options:** `--link` is now `--link-to-full-recording` and `--apply-order` is now `--apply-filter-order`, as sidecars and tables name them. The old names still work.
- **Dependencies:** `rich` and `httpx2` are now required, so a command-line-only install has its help screens and the iNaturalist source. `httpx` is no longer a dependency of the `gui` extra.

### New

- **Your own models.** Any ONNX embedding model can be added under `[models.custom.<id>]`, from Hugging Face or an http(s) URL. The library remembers it, so it still works when opened elsewhere without that configuration; a URL model's file is checked against the SHA-256 recorded on first download, and a definition cannot change once vectors exist under it.
- **Adding a model to a library with items** now says how many have no vectors from it and that `biotimbre backfill <id>` computes them. A model whose window length does not fit its graph is refused with the length it takes, and one that looks like the wrong kind (several outputs, class scores) is flagged.
- **Find similar.** A library item can be the search query, from its dialog or a search result, or with `biotimbre search --item ITEM_ID`. The item is left out of its own results unless asked for (`--include-self`, or *Include the item itself*).
- **Share several items at once.** Tick items on the Library or Search page and export them as one bundle, with or without embeddings. On the command line, `search --csv` writes the results, and `export-bundle --items-from` bundles the items left in that file once wrong matches are deleted.
- **Import from a table.** `biotimbre import-table` adds many items from one CSV (comma, semicolon or tab separated), with a row per recording or per iNaturalist observation. iNaturalist is asked at most once a second and for under 5 GB an hour, as its recommended practices ask. A recording already stored is skipped, so an interrupted import can be run again, and `--dry-run` checks every row first.
- **iNaturalist on the command line:** `add --inat` fills the metadata from an observation, as the Library page does, and `search --inat` searches with its recording; `--inat-filters` also narrows by its taxon and place, and `--inat-sound` picks another of its recordings.
- **The command line can do what the interface does:** trim, filter and normalise a recording when adding it, with options or sidecar lines; `edit` an item's metadata; filter a query before searching; search by metadata alone; and list the items stored at given speeds (`search --speed`). A sidecar may give the location as `latitude` and `longitude` lines.
- **Switch library** from the interface header, without restarting: the library is closed and the chooser opens, with the models still loaded.
- **Clearer command-line output.** Search results are laid out in labelled blocks, and an item's audio file and full recording are links, which play or open in terminals that support them. Every command's `--help` explains each option in full, with examples, and usage errors say what to change.
- **Documentation** is split into pages, in English and Portuguese, linked from the README, including what uninstalling leaves behind.

### Fixes

- A trimmed, filtered, normalised or converted recording is recognised as already stored when it is added again. Each processed WAV used to be written with a timestamp in it, so the same audio never had the same bytes twice.
- The credit added to `notes` names the licence of the observation's recording that was used, not always its first one.
- The library chooser shows the browsed folder once, in its path field.
- Fields in the item details dialog are ruled off from each other, so a long value is not read as the next field.
