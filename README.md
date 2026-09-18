# Biotimbre: A management tool for animal sound libraries

A tool for creating and managing libraries of animal sound templates with embedding-based similarity search, using ONNX backbones via `onnxruntime` (e.g. Perch v2, BirdNET 2.4).

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
`--no-map` leaves out the location map, which is the only part that fetches
anything from the network on its own.

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
biotimbre export-npy ITEM_ID MODEL_ID OUT   # .npy per window; .zip if several
biotimbre export-bundle ITEM_ID... -o FILE  # package items to send to someone
biotimbre import-bundle FILE [--dry-run]    # load a bundle you were sent
biotimbre stats
biotimbre gui [--host H] [--port P] [--no-browser]
```

Global options: `--data-dir`, `--config`, `--models`.

Started with no `--data-dir` and no configured one, `biotimbre gui` does not
quietly make a library in whatever directory you launched it from: it asks.
The first page is a chooser — type a path or browse to one, make a folder if
you need it, name the library, open it. An existing library opens as it is;
directories that already hold one are marked. Libraries opened from that
browser are offered again next time.

Browsing is not confined to `$HOME` — libraries live on external drives and
mounted archives. That listing is the one place in the interface where a
request names a filesystem path, so it exists only while no library is open
and is withdrawn the moment one is.

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

### One species, one biotimbre

A species is where an identification is complete, so the library gives it a
single identity: add a sample tagged with a species already in the library —
`Guyalna bonaerensis` — and it is filed under that species' existing
`biotimbre_id` instead of starting a second one for the same animal. Tags are
stored lowercase, so the capitalisation you type does not matter.

This applies to binomials only. A rank above species (`cicadidae`) says nothing
about which being was recorded, and open nomenclature (`Guyalna sp.`, `cf.`,
`aff.`) is precisely an admission that the species is unknown — both keep
starting their own biotimbre. Naming a `biotimbre_id` yourself always wins.

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

## Sharing items between libraries

`biotimbre_id` is a UUID so that one sound-being means the same thing in every
library that holds it. A **bundle** is how one travels: a single `.btz` file
carrying an item's metadata, its stored audio, and its embedding vectors.

```bash
biotimbre export-bundle 12 -o guyalna.btz          # one item, with its vectors
biotimbre export-bundle 12 13 14 -o batch.btz      # several
biotimbre export-bundle 12 -o small.btz --no-embeddings
```

Send the file however you like. On the other side, look before you leap:

```bash
biotimbre import-bundle guyalna.btz --dry-run      # says what would happen
biotimbre import-bundle guyalna.btz
```

A `.btz` is a plain zip — open it and you will find `item.json` (the
authoritative metadata), `item.txt` (the same fields as a sidecar, for
reading), the audio as a mono WAV, and the vectors as `.npz` arrays.

### What an import decides, and why

**The same recording, already here.** Matched on the stored file's digest, so
two copies recognise each other whatever either library called them. Under the
same biotimbre it is the same sample and is skipped. Under a *different* one it
is two libraries disagreeing about which being it is, which is reported and
skipped — `--duplicates add` overrides.

**A species that already has a biotimbre here.** You and the sender each
catalogued *Guyalna bonaerensis* and have two UUIDs for one animal. One species,
one biotimbre: the incoming sample joins yours, and the sender's id is reported
so nothing vanishes quietly. `--species keep-incoming` keeps theirs instead, and
leaves you with two identities for one species until you merge them.

**Embeddings.** Vectors are comparable only when the whole preprocessing recipe
matches, so the bundle carries each model's full fingerprint — source, file,
revision, sample rate, window length, overlap, dimension — and your library
compares it field for field. A match is stored as it is, saving an inference
pass; a mismatch is refused, naming the field that differs, and recomputed. A
model you do not run is refused whatever it says. `--distrust-embeddings`
throws them all away and recomputes from the audio, which is the only part of a
bundle that verifies itself.

Anything the bundle did not supply is computed on import, so an imported item is
searchable straight away. `--no-embeddings` leaves those gaps for `backfill`
instead — for a bulk import on a machine without the model files.

A bundle from another person is treated as untrusted: only members its manifest
names are read, paths that would escape the archive are refused, sizes and
compression ratios are capped, arrays load with pickling off, every audio digest
is verified, and every field goes through the same validation a typed-in form
would.

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
you need the vectors as files for interop.

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

[map]
# The location map on both pages. This is the only part of the interface that
# reaches the network simply by being looked at — Leaflet itself is served
# from this process, but the basemap tiles come from the URL below — so an
# offline library sets enabled = false. Coordinates are still filtered on and
# still shown as text either way.
enabled = true
# tile_url    = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
# attribution = "&copy; OpenStreetMap contributors"   # must credit the source
# max_zoom    = 19        # match what your tile server actually has

[ui.theme]
# The light palette. Every key optional; shown values are the defaults. Colours
# are CSS values (hex, rgb(a), named) — primary..warning feed Quasar's palette,
# the rest style the header directly.
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

[ui.dark_theme]
# The dark palette: same keys, same meanings. Omitted keys fall back to the
# built-in dark defaults, never to the light values above, so overriding one
# colour does not drag the light palette into the dark theme.
# primary = "#5a9c3e"
# page_bg = "#111509"
# text = "#e7ecdc"
# card = "#171d10"
# spectrogram_colors is not read here: the images are rendered and cached
# server-side, where the reader's theme is not known, so [ui.theme]'s ramp
# serves both.
```

### Trimming and filtering on the way in

The Library page's add form works on the recording you upload before it is
stored: drag over the spectrogram to keep only part of it, and give a highpass
and/or lowpass cutoff to filter it (Butterworth, zero-phase). A second
spectrogram shows exactly what will be stored, and the player follows the
chosen segment.

What the library keeps is the processed audio, so the metadata is filled in to
match: the segment becomes `sample_window` — where the sample sits in the
recording it came from — and the cutoffs become `filtered_band` with
`filter_type` "Butterworth" and the order used. Both stay editable, since only
you know whether the file you uploaded was itself already an excerpt.

What the library keeps is always mono — that is what every model is fed and
what a spectrogram shows — so a multi-channel submission is averaged down, by
the add form and by `biotimbre add` alike, and the player previews that mono
copy. A mono file you neither trim nor filter is stored exactly as it arrived.

### Taking an item away

Both the library list and the search results carry an **Export** button on each
item: the stored audio file, or one model's embeddings.

A window is the unit that means something — one vector over one span of seconds
— so no exported file holds more than one of them. A model with a single window
(Perch, on a short sample) exports as a bare `item_12_perch_v2.npy` holding one
1-D vector; several windows export as `item_12_birdnet_v2.4.zip`, one `.npy`
inside per window, each named with its index and span
(`item_12_birdnet_v2.4_w00_0.00-3.00s.npy`). `biotimbre export-npy` writes
exactly the same files, so a download and a command-line export of the same
rows are interchangeable.

### Fetching from iNaturalist

Both pages take an iNaturalist observation ID or link, in a field under the
file uploader, and stage the recording exactly as an upload — same preview,
same controls.

On the **Library** page it also fills the metadata from the observation: date
and time in the observation's own timezone, coordinates, taxon, the observer as
`recorded_by`, the observation page as `link_to_full_recording` — which is what
a trimmed sample's `sample_window` then refers to — and the licence and credit
appended to `notes`. Every field stays editable, and notes you have already
typed are added to rather than replaced.

On the **Search** page it becomes the query recording, and what the observation
knows — its coordinates with a 25 km radius, and the taxon — goes into the
filter panel, which it opens so the narrowing is visible and can be cleared.
Not the date: an observation's day says when that one recording was made, not
when the animal sings.

Either way it says what you should know before trusting it: an obscured
location is left blank rather than filled with iNaturalist's randomised point,
an all-rights-reserved recording is flagged, and anything short of research
grade is noted, since the identification may still change. A progress bar runs
while the recording arrives.

Most iNaturalist audio is AAC in an `.m4a`, which libsndfile cannot read, so
decoding it needs **ffmpeg**. Without ffmpeg the fetch fails with a message
saying so, and everything else still works. Note that a typical 128 kbps upload
carries nothing above about 16 kHz whatever its sample rate claims — fine for
Perch, below BirdNET's 24 kHz, and useless for ultrasonic work.

Turn the whole thing off for an offline library:

```toml
[sources.inaturalist]
enabled = false
```

### The location map

Every item carries coordinates — the columns are `NOT NULL` — so both pages
show **one map**: the library page plots every item, the search page plots the
current results. One dot per recording; click a dot to see which one it is.
There is deliberately no mini-map per row, which would mean one map instance
and one set of tile requests per result.

On the search page the map also works the other way round: clicking it fills
in the location filter's centre (opening the filter panel so the narrowing is
visible), and the centre and radius are drawn as a dot and a circle that
follow whatever the three filter fields say.

Leaflet ships inside NiceGUI and is served from this process, so no script is
fetched from a CDN. The **basemap tiles are the exception** and the only reason
the map touches the network. Turn it off for a library kept offline — the
coordinates are still filtered on and still printed on every result:

```toml
[map]
enabled = false
```

or, for a single run, `biotimbre gui --no-map`. Command-line flags outrank the
config file and survive opening a different library from the browser.

Pointing `tile_url` at a tile server of your own keeps the map off the public
one; whatever you point it at, `attribution` has to credit it, and `max_zoom`
should match the deepest zoom it actually serves or you get blank tiles.

### Spectrogram or waveform

Every picture of a sample — in the item detail, the add form, the search query
and each result — can be shown either way. The small toggle beside it switches
both at once and the choice is remembered per browser, alongside the language
and the theme.

The waveform reports two readings per pixel column: the **peak** pair (how far
the signal swung, so a click or a wingbeat survives) as a light band, and the
**RMS** (how loud that column was, which is what the ear reports) drawn inside
it. Amplitudes are scaled to the loudest column, as the spectrogram's dB range
is scaled to its loudest bin, so a quiet recording is a picture of its own
signal rather than a flat line.

It is drawn as inline SVG rather than a rendered image, which is what lets it
take its colours from the theme — a PNG is drawn server-side, where the
reader's light or dark choice is not known — and stretch to any width without
another round trip.

### Light, dark, or whatever the system says

The header carries a three-state control: **follow the system** (the default),
**light**, **dark**. The choice is remembered per browser, alongside the
language, so two people sharing one local instance can read it differently, and
neither rewrites the config file.

Each colour is declared once, as `light-dark(<light>, <dark>)` against a CSS
`color-scheme`, so the two palettes cannot drift apart — and native controls no
stylesheet reaches (scrollbars, the audio player, date pickers) follow the
theme too. Switching needs no reload.

## Development

```bash
uv run pytest                    # ~110 s
uv run pytest -m "not models"    # ~3 s: skips the 66 tests that run real models
```
