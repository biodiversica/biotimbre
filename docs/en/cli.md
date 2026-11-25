# Command-Line Interface

*English · [Português (Brasil)](../pt-BR/cli.md)*

## Command Reference

```bash
biotimbre init [--name NAME] [--skip-models]   # --name is required until one is set
biotimbre add AUDIO [--sidecar FILE | metadata options] [processing options]
biotimbre add --inat OBSERVATION [metadata options] [processing options]
biotimbre import-dir DIR                       # every file paired with a sidecar
biotimbre import-table TABLE.csv [--dry-run]   # one row per recording or observation
biotimbre list [--biotimbre-id ID] [--limit N]
biotimbre show ITEM_ID
biotimbre edit ITEM_ID [metadata options] [--clear FIELD...]
biotimbre delete ITEM_ID [-y] [--keep-audio]
biotimbre search [AUDIO | --inat OBSERVATION | --item ITEM_ID] [-k N] [filters]   # none: by filters
biotimbre models                               # registered models and vector counts
biotimbre backfill MODEL_ID                    # embed items missing this model
biotimbre export-npy ITEM_ID MODEL_ID OUT [--with-audio]
biotimbre export-csv OUT                       # metadata of every item as CSV
biotimbre export-bundle [ITEM_ID...] [--items-from CSV] -o FILE [--no-embeddings]
biotimbre import-bundle FILE [--dry-run] [options]
biotimbre import-library DIR [--dry-run] [options]   # add another library's items to this one
biotimbre stats
biotimbre gui [--host H] [--port P] [--no-browser] [--no-map] [--search-only]
```

Global options: `--data-dir`, `--config`, `--models`. They go before the
command, e.g. `biotimbre --data-dir ~/my-library list`.

`biotimbre --help` lists every command, and `biotimbre COMMAND --help`
explains each of its arguments and options in full, with examples.

## Examples

```bash
# Add a recording with explicit metadata. Coordinates are given as separate
# options, since argparse cannot accept "--location -26,-48" (the value
# begins with a minus sign).
biotimbre add katydid.wav \
    --date 2021-03-05 --time 22:10:00 \
    --latitude -26.9 --longitude -48.7 \
    --window-start 12 --window-end 17 \
    --taxa orthoptera insecta --device Audiomoth \
    --highpass 8000 --lowpass 40000 --speed 0.25

# Search, with filters applied before scoring
biotimbre search query.wav -k 5 \
    --taxa cicadidae --latitude -26 --longitude -48 --radius-km 50 \
    --date-from 2020-01-01

# A query longer than the limit requires an explicit segment
biotimbre search dawn_chorus.wav --start-time 92 --end-time 99

# Filter the query before embedding it (the library is not changed)
biotimbre search noisy.wav --apply-highpass 1500

# Search with an item already in the library: what else sounds like item 12
biotimbre search --item 12

# Search by metadata alone: no query recording, results newest first
biotimbre search --taxa cicadidae --date-from 2024-01-01 -k 50

# Store only part of a recording, filtered and normalised, as the add form does
biotimbre add dawn.wav --sidecar dawn.txt --trim 92 99 --apply-highpass 1500 --normalise

# Change stored metadata; fields not given are kept
biotimbre edit 12 --taxa "guyalna bonaerensis" --clear device
```

Further `search` options: `--metric {cosine,dot,euclidean}`,
`--rank-by MODEL_ID` (default `perch_v2`), `--min-score`, `--query-speed`, `--taxa-all`,
`--date-to`, `--biotimbre-id`, `--exclude`, `--apply-lowpass`, `--apply-filter-order`,
and, in a search by filters alone, `--speed` (items stored at these speeds).

`--highpass`/`--lowpass` on `add` and `edit` *record* filtering already done to
a recording; `--apply-highpass`/`--apply-lowpass` *filter* it. Every option is
explained in full by `biotimbre COMMAND --help`.

Search results and `show` give each item's audio as the full path of its stored
file. In a terminal that supports links (GNOME Terminal, Konsole, Windows
Terminal, iTerm2, kitty, WezTerm, VS Code), the path can be clicked to play the
recording in the system's audio player; elsewhere it can be copied into one.
