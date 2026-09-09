# Biotimbre: A vocalization library management tool

A tool for creating and managing libraries of animal vocalization templates with embedding-based similarity search, using ONNX backbones via `onnxruntime` (e.g. Perch v2, BirdNET 2.4).

## Setup

```bash
uv venv --python 3.12
uv sync --extra dev --extra gui
uv run biotimbre init --name "My Library"   # downloads ~70 MB of models, once
```

Models are cached after the first download, so everything afterwards works with
no network.

`--name` is required the first time: every library gets a name and a UUID
(generated once, stable for its lifetime) so it can be identified after being
shared — copy the `data_dir` to hand a library to someone else, and both the
name and UUID travel with it. Both are shown in the GUI header and in
`biotimbre stats`. Rerun `init --name "..."` any time to rename.

## Library path

By default, `./data` relative to the current directory. Three ways to change it:

```bash
# 1. per invocation
uv run biotimbre --data-dir /media/drive/biotimbre list

# 2. a config file in the working directory (or --config /path/to/it)
echo '[library]
data_dir = "/media/drive/biotimbre"' > biotimbre.toml

# 3. an environment variable pointing at a config file
export BIOTIMBRE_CONFIG=/media/drive/biotimbre/biotimbre.toml
```

`--data-dir` overrides the config file. Relative paths inside a config file
resolve against **that file's own directory**, not the working directory, so a
config file and its library can be moved together.

**Using an existing library elsewhere:** just point at it. Every command opens
the database and creates anything missing, so no separate step is needed:

```bash
uv run biotimbre --data-dir /media/drive/biotimbre list
```

`init` is safe to re-run on a populated library — it only creates what is absent
and downloads the models. Stored audio paths are relative to the data directory,
so **moving or renaming the whole folder keeps every reference valid.**

## Web interface

```bash
uv run biotimbre gui        # http://127.0.0.1:8080
```

Loopback by default; exposing the library takes an explicit `--host`. Models load
in the background, so the page is usable while the header reports progress.

- **Library** — add, inspect, edit, delete. Choosing a file shows its spectrogram
  and sample rate before you commit, with each model's frequency limit noted.
  Validation errors mark the offending field. Editing warns before saving if you
  changed `speed`, since that recomputes embeddings.
- **Search** — query recording in, ranked matches out. Each result has its own
  player and marks the window that produced its score, plus the per-model
  breakdown. The query stays playable at the top for A/B comparison. For a file
  over 15 s, a slider over the spectrogram picks the segment.

## Commands

```bash
biotimbre init [--name NAME] [--skip-models]   # --name required until one is set
biotimbre add AUDIO [--sidecar FILE | metadata flags]
biotimbre import-dir DIR              # every sidecar-paired file
biotimbre list [--biotimbre-id ID] [--limit N]
biotimbre show ITEM_ID
biotimbre delete ITEM_ID [-y] [--keep-audio]
biotimbre search AUDIO [-k N] [filters]
biotimbre models                      # registered models and vector counts
biotimbre backfill MODEL_ID           # embed items missing this model
biotimbre export-npy ITEM_ID MODEL_ID OUT
biotimbre stats
biotimbre gui [--host H] [--port P] [--no-browser]
```

Global options: `--data-dir`, `--config`, `--models`.

```bash
# add with explicit metadata (scalar coordinates: argparse cannot take
# "--location -26,-48", since the value starts with a minus)
biotimbre add katydid.wav \
    --date 2021-03-05 --time 22:10:00 \
    --latitude -26.9 --longitude -48.7 \
    --window-start 12 --window-end 17 \
    --taxa orthoptera insecta --device Audiomoth \
    --highpass 8000 --lowpass 40000 --speed 0.25

# search, with filters applied before scoring
biotimbre search query.wav -k 5 \
    --taxa cicadidae --latitude -26 --longitude -48 --radius-km 50 \
    --date-from 2020-01-01

# a longer query needs an explicit segment
biotimbre search dawn_chorus.wav --start-time 92 --end-time 99
```

Other `search` options: `--metric {cosine,dot,euclidean}`, `--fusion {max,mean,MODEL_ID}`,
`--min-score`, `--query-speed`, `--taxa-all`, `--date-to`, `--exclude`.

## Metadata

**Required:** `date` (YYYY-MM-DD), `time` (HH:MM:SS), location.

**Optional:** `biotimbre_id` (omit to allocate a new one; give an existing one to
file the sample under the same biotimbre), `taxa` (empty means unknown),
`sample_window` (start and end in seconds *within the full recording* — it only
places the sample if there is a full recording to place it in, so it pairs with
`link_to_full_recording`; give both ends or neither), `device`, `microphone`,
`recorded_by`, `filtered_band`, `filter_type`, `filter_order`, `temperature`,
`humidity`, `speed`, `link_to_full_recording`, `notes`.

`item_id` is assigned automatically, never reused, and library-specific — just a
row identifier. A `biotimbre` is a sound-being: a sonotype whose species identity
may be unknown, so `biotimbre_id` is a UUID rather than a library-local counter —
it's meant to identify the same sonotype consistently if libraries are ever
shared or merged. Several items may share one `biotimbre_id` — same species or
individual, different sound content.

### Sidecar files

One field per line, name then value, separated by tabs or spaces. A file named
`GUYVIR.txt` pairs with `GUYVIR.wav`.

```
taxa	cicadidae
date	2020-01-14
time	09:40:21
sample_window 5,10
location -20,-46
device	Audiomoth
filtered_band	2000,10000
```

`filtered_band` accepts `-` for an open bound, e.g. `2000,-`. `#` and `;` start
comments. An unrecognised field name is an error rather than being ignored, so a
typo cannot silently drop a required value.

## Ultrasonic samples: `speed`

Embedding resamples to each model's rate — 32 kHz for Perch, 48 kHz for BirdNET —
which filters away everything above 16 kHz and 24 kHz. A katydid at 35 kHz would
be embedded as near-silence and never match anything.

`--speed` reinterprets the source rate before resampling. At `0.25`, content at
40 kHz lands at 10 kHz, inside both models' range:

```bash
biotimbre add katydid.wav --sidecar katydid.txt --speed 0.25
```

The stored audio is never modified, and playback is unaffected.

**You do not need to remember an item's speed when searching.** The query is
automatically embedded at every speed present in the library, so a
natively-recorded query still finds a slowed-down item. Results report the speed
that matched.

If a declared `filtered_band` exceeds a model's limit, `add` prints an advisory
suggesting a speed value. Pick one that brings your highest frequency of interest
under 16 kHz; dividing by 4 or 8 suits most insect and bat material. Consistency
across similar recordings matters more than the exact value.

## Search notes

A search embeds at most **15 s** of query audio. Scoring takes the best-matching
window pair, so a long query lets an arbitrary fragment decide the result. Files
within the limit are used whole; longer ones need `--start-time`/`--end-time`,
which is refused rather than guessed. Only those frames are decoded.

Filters run before scoring, so results are ranked within the filtered set rather
than filtered after the fact. Re-running the same query with different filters
reuses cached embeddings and returns immediately.

## Adding a model later

Register it in `models_registry.py`, then:

```bash
biotimbre backfill MODEL_ID
```

Existing items are re-read from their stored audio; nothing needs re-importing.

## Data layout

```
data/
├── library.db                        # metadata AND embedding vectors
└── audio/<biotimbre_id>/<item_id>.wav
```

Back up by copying the folder. Vectors live in the database so a metadata filter
can narrow the candidate set before any vector is scored; use `export-npy` when
you need a `.npy` for interop.

## Configuration

Optional `biotimbre.toml`. Every field has a default.

```toml
[library]
data_dir = "data"
# db_path   = "/fast/library.db"      # may sit apart from the audio
# audio_dir = "/bulk/audio"           # stored absolute if outside data_dir
# model_cache_dir = "/path/to/cache"  # default: the Hugging Face cache

[models]
active = ["perch_v2", "birdnet_v2.4"]
inference_batch_size = 1   # windows per model call; 1 is fastest here
inference_threads = 0      # 0 lets onnxruntime choose

[search]
top_k = 10
metric = "cosine"          # cosine | dot | euclidean
fusion = "max"             # max | mean | a model id
batch_size = 8192
query_cache_entries = 16
max_query_duration_s = 15

[ui]
language = "en"            # en | pt_BR

[ui.theme]
# Every key optional; shown values are the defaults. Colours are CSS values
# (hex, rgb(a), named) — primary..warning feed Quasar's palette, the rest
# style the header directly.
primary = "#118b50"
secondary = "#5ba98c"
accent = "#e9b44c"
dark = "#1d1d1d"
positive = "#118b50"
negative = "#c62828"
info = "#2f6f9f"
warning = "#e9b44c"
button_text = "#ffffff"        # label colour on filled color=primary buttons
page_bg = "#ffffff"            # body background below the header
text = "#1a1a1a"                # default label colour (unstyled ui.label)
header_bg = "#118b50"
header_text = "#ffffff"
header_inactive_text = "rgba(255, 255, 255, 0.65)"
header_active_bg = "rgba(255, 255, 255, 0.2)"
header_status_ready = "#a8e6a1"    # "Models ready" badge
header_status_loading = "#ffd166"  # "Loading models…" badge
header_status_failed = "#ff8a8a"   # model-load-error badge

# The rest theme widgets Quasar/Tailwind hardcode regardless of ui.colors():
card = "#ffffff"               # card backgrounds (item detail, add/search forms)
link = "#3b82f6"                # hyperlinks (e.g. "open full recording")
muted = "#64748b"              # secondary/hint text (labels, timestamps, IDs)
notice = "#92400e"             # inline warnings (speed-shifted, query cap, etc.)
surface = "#f1f5f9"            # stat-tile backgrounds on the admin page
chip = "#e2e8f0"                # small id-badge backgrounds
marker_border = "rgba(255, 255, 255, 0.8)"  # matched-window box on a spectrogram
marker_fill = "rgba(255, 255, 255, 0.25)"

# Recolours the spectrogram images themselves, not a GUI widget. Evenly spaced
# #rrggbb stops from quiet (first) to loud (last); at least 2 required. Keep
# the ramp monotonically brighter so intensity still reads for colour-blind
# viewers. Shown values are the default dark-blue -> magenta -> orange -> pale
# yellow ramp.
spectrogram_colors = ["#0c0826", "#400c60", "#941a60", "#de5234", "#fca83c", "#fff4c8"]
```

## Development

```bash
uv run pytest                    # ~110 s
uv run pytest -m "not models"    # ~3 s: skips the 66 tests that run real models
```
