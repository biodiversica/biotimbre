# Biotimbre

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

```bash
uv venv --python 3.12
uv sync --extra dev --extra gui
uv run biotimbre init --name "My Library"   # downloads approximately 70 MB of models, once
```

Models are cached after the first download; no network connection is required
afterwards.

The `--name` option is required the first time a library is initialised. Every
library has a name and a UUID, generated once and stable for its lifetime, so
that it remains identifiable after being shared: copying the data directory
transfers both. The name and UUID are shown in the interface header and by
`biotimbre stats`. Running `init --name "..."` again renames the library.

## Library Location

By default, the library is stored in `./data`, relative to the current
directory. The location can be set in three ways:

```bash
# 1. For a single invocation
uv run biotimbre --data-dir /media/drive/biotimbre list

# 2. With a configuration file in the working directory (or --config /path/to/it)
echo '[library]
data_dir = "/media/drive/biotimbre"' > biotimbre.toml

# 3. With an environment variable pointing to a configuration file
export BIOTIMBRE_CONFIG=/media/drive/biotimbre/biotimbre.toml
```

`--data-dir` takes precedence over the configuration file. Relative paths in a
configuration file are resolved against the directory containing that file,
not the working directory, so a configuration file and its library can be
moved together.

**Opening an existing library.** No separate step is required: every command
opens the database and creates anything that is missing.

```bash
uv run biotimbre --data-dir /media/drive/biotimbre list
```

`init` may be run safely on a populated library; it only creates what is absent
and downloads the models. Stored audio paths are relative to the data
directory, so **the whole folder can be moved or renamed without breaking any
reference.**

## Web Interface

```bash
uv run biotimbre gui        # http://127.0.0.1:8080
```

The server listens on the loopback interface by default; serving it to other
machines requires an explicit `--host`. Models are loaded in the background, so
the interface is usable while the header reports their progress. `--no-map`
disables the location map, which is the only component that accesses the
network on its own.

The interface has two pages, **Library** and **Search**, presented as tabs.
Both remain loaded while switching between them, so entered values, staged
recordings and search results are preserved. The address bar follows the
active page, so reloading, bookmarks and the browser's back and forward
buttons behave as expected.

### Library Page

The Library page is divided into three collapsible sections.

- **Library items** — library totals, a map of every item and the item list.
  The totals are followed by the speeds present in the library and by its
  taxa, each with its number of items; when there are many taxa, the most
  frequent are shown first and the rest can be expanded. Selecting an item
  opens a dialog with its full metadata, its spectrogram or
  waveform, a player, and buttons to export, edit or delete it. Editing warns
  before saving a change of `speed`, since that requires recomputing the
  embeddings. **Export CSV** downloads the metadata of every item as a table
  (see [Exporting Data](#exporting-data)).
- **Add an item** — uploads a recording and its metadata. The spectrogram and
  sample rate are shown before anything is stored, with each model's frequency
  limit indicated. The recording can be trimmed, filtered and normalised before
  it is stored (see [Audio Preparation](#audio-preparation)). Validation errors
  are shown on the corresponding field.
- **Import items** — loads a `.btz` bundle from another library, or adds
  every item of another library on the same computer, with a preview of the
  outcome that can be adjusted before anything is added (see
  [Sharing Items Between Libraries](#sharing-items-between-libraries)).

The recording location is entered in a single field as
`latitude, longitude` in decimal degrees. A comma, a semicolon or spaces are
accepted as separators, so coordinates copied from mapping services, GPS
applications or field notes can be pasted directly. **Pick on map** sets the
location by clicking on a map instead.

### Search Page

The Search page takes a query recording and returns the most similar library
items, ranked by score.

- The query is prepared with the same controls as the Library page's add form
  (trimming, filtering, normalisation and speed), and the prepared copy is
  what is searched. The selection is limited to 15 s; for a longer file, a
  segment of at most 15 s is selected on the spectrogram.
- **Filters** restrict the search by taxa, date range, distance from a centre
  point, and biotimbre ID. A search may also be run with filters only and no
  recording, in which case the matching items are listed without scores,
  newest first.
- **Options** set the query speed, the maximum number of results, the
  similarity metric, how the models' scores are combined, and a minimum score.

Each result shows its score, the per-model breakdown, its own player, and a
spectrogram on which the window that produced the score is marked. The query
remains playable at the top of the page for comparison by ear. Each result
also provides:

- **Details** — a read-only dialog with the item's full metadata, as on the
  Library page but without the edit and delete options.
- **Export** — the item's audio, its embeddings or a shareable bundle.
- A shortened biotimbre ID which, when clicked, copies the full UUID to the
  clipboard, for use in the biotimbre ID filter or in another library.

**Export CSV**, next to the result count, downloads the displayed results as a
table. **Add to library…** opens the Library page's add form with the query
recording and all of its settings (and, for a recording fetched from
iNaturalist, the observation's metadata) so that a new recording can be added
to the library. Nothing is stored until the form is saved. This option is not
available in search-only mode.

### Search-Only Mode

```bash
uv run biotimbre gui --data-dir ./library --search-only
```

This mode allows other people to search a library without being able to modify
it. The Library page is not served (`/` redirects, so the page and its handlers
are never created), the library chooser is not available, and **the database
is opened read-only**, which makes this a guarantee rather than a hidden tab.
The header displays `Search only` to indicate the mode.

The library must already exist: `--search-only` requires one named with
`--data-dir` (or `library.data_dir`) and creates nothing, so a mistyped path
is reported as an error rather than producing a new, empty library. A library
whose schema requires an upgrade is also refused, since migrating it is a
write; such a library must first be opened once without the flag.

Every feature of the Search page remains available, including downloading a
result's audio or exporting it as a bundle. The mode prevents the library from
being *changed*; it does not restrict reading its contents. To make it
reachable from another machine, combine it with `--host`:

```bash
uv run biotimbre gui --data-dir ./library --search-only --host 0.0.0.0
```

The equivalent configuration, for a machine that should only serve searches:

```toml
[ui]
search_only = true
```

### Selecting a Library at Startup

When `biotimbre gui` is started without `--data-dir`, it does not open or
create a library on its own. The first page asks, with two choices:

- **Create a new library** takes a name and a location. The library is
  created in a new folder of its own inside that location, named after it
  (*Atlantic Forest Cicadas* becomes `atlantic-forest-cicadas`), and the full
  path is shown before anything is created. A folder that already exists and
  is not empty is refused.
- **Open an existing library** browses to a library; folders that contain one
  are marked. A folder without a library is reported rather than turned into
  a new one, so a mistyped path cannot produce an empty library by accident.

Libraries opened from the same browser are offered again on the next visit.
A `data_dir` in a configuration file does not skip this page: the library it
names is offered first, or, if it does not exist yet, offered for creation at
that path. Only `--data-dir` opens a library directly. The other commands, and
`gui --search-only`, which has no chooser, still use the configured
`data_dir`.

Browsing is not restricted to `$HOME`, since libraries are often kept on
external drives and mounted archives. Because this listing is the only part
of the interface where a request names a filesystem path, it exists only while
no library is open and is withdrawn as soon as one is.

## Command-Line Interface

### Command Reference

```bash
biotimbre init [--name NAME] [--skip-models]   # --name is required until one is set
biotimbre add AUDIO [--sidecar FILE | metadata options]
biotimbre import-dir DIR                       # every file paired with a sidecar
biotimbre list [--biotimbre-id ID] [--limit N]
biotimbre show ITEM_ID
biotimbre delete ITEM_ID [-y] [--keep-audio]
biotimbre search AUDIO [-k N] [filters]
biotimbre models                               # registered models and vector counts
biotimbre backfill MODEL_ID                    # embed items missing this model
biotimbre export-npy ITEM_ID MODEL_ID OUT [--with-audio]
biotimbre export-csv OUT                       # metadata of every item as CSV
biotimbre export-bundle ITEM_ID... -o FILE [--no-embeddings]
biotimbre import-bundle FILE [--dry-run] [options]
biotimbre import-library DIR [--dry-run] [options]   # add another library's items to this one
biotimbre stats
biotimbre gui [--host H] [--port P] [--no-browser] [--no-map] [--search-only]
```

Global options: `--data-dir`, `--config`, `--models`.

### Examples

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
```

Further `search` options: `--metric {cosine,dot,euclidean}`,
`--fusion {max,mean,MODEL_ID}`, `--min-score`, `--query-speed`, `--taxa-all`,
`--date-to`, `--biotimbre-id`, `--exclude`.

## Metadata

### Fields

**Required:** `date` (YYYY-MM-DD), `time` (HH:MM:SS) and location (latitude and
longitude in decimal degrees).

**Optional:**

| Field | Description |
|---|---|
| `biotimbre_id` | Omit to allocate a new biotimbre; give an existing ID to file the item under it. |
| `taxa` | Taxon tags. Empty means unknown. |
| `sample_window` | Start and end, in seconds, of the item *within the full recording*. It is meaningful only together with `link_to_full_recording`. Both ends or neither must be given. |
| `device`, `microphone`, `recorded_by` | Recording equipment and recordist. |
| `filtered_band`, `filter_type`, `filter_order` | Filtering applied to the recording. |
| `temperature`, `humidity` | Conditions at the time of recording. |
| `speed` | Playback speed used for embedding (see [Ultrasonic Recordings](#ultrasonic-recordings-and-the-speed-setting)). |
| `link_to_full_recording` | URL of the complete recording. |
| `notes` | Free text. |

### Identifiers

`item_id` is assigned automatically, never reused, and specific to one library;
it is a row identifier only.

A **biotimbre** is a sound-being: a sonotype whose species may be unknown.
`biotimbre_id` is therefore a UUID rather than a library-local counter, so that
the same sonotype is identified consistently when libraries are shared or
merged. Several items may share one `biotimbre_id` — the same species or
individual, with different sound content.

### One Species, One Biotimbre

A species is a complete identification, so the library assigns it a single
identity. An item tagged with a species already present in the library — for
example `Guyalna bonaerensis` — is filed under that species' existing
`biotimbre_id` instead of starting a second one for the same animal. Tags are
stored in lowercase, so capitalisation does not matter.

This rule applies to binomials only. A rank above species (`cicadidae`) does
not establish which being was recorded, and open nomenclature (`Guyalna sp.`,
`cf.`, `aff.`) explicitly states that the species is unknown; both start their
own biotimbre. An explicitly given `biotimbre_id` always takes precedence.

### Sidecar Files

A sidecar file holds one field per line: the field name followed by its value,
separated by tabs or spaces. A file named `GUYVIR.txt` is paired with
`GUYVIR.wav`.

```
taxa	cicadidae
date	2020-01-14
time	09:40:21
sample_window 5,10
location -20,-46
device	Audiomoth
filtered_band	2000,10000
```

`filtered_band` accepts `-` for an open bound, e.g. `2000,-`. Lines beginning
with `#` or `;` are comments. An unrecognised field name is reported as an
error rather than ignored, so a misspelling cannot silently discard a required
value.

## Audio Preparation

### Trimming, Filtering and Normalisation

The add form processes the uploaded recording before storing it. A segment is
selected by dragging over the spectrogram; a highpass and/or lowpass cutoff
applies a zero-phase Butterworth filter; and normalisation scales the
recording so that its loudest sample reaches a chosen level. A second
spectrogram shows exactly what will be stored, and the player follows the
selected segment. The Search page offers the same controls for the query, with
the selection limited to 15 s.

The library stores the processed audio, and the metadata is filled in
accordingly: the segment becomes `sample_window` (the position of the item in
the original recording), and the cutoffs become `filtered_band`, with
`filter_type` set to "Butterworth" and the order used. Both remain editable,
since only the user knows whether the uploaded file was itself an excerpt.
Normalisation is recorded in the notes.

Stored audio is always mono, as that is what every model receives and what a
spectrogram shows. A multi-channel recording is therefore averaged to mono,
both by the add form and by `biotimbre add`, and the player previews the mono
version. A mono file that is neither trimmed nor filtered is stored exactly as
received.

### Ultrasonic Recordings and the Speed Setting

Before embedding, audio is resampled to each model's rate — 32 kHz for Perch
and 48 kHz for BirdNET — which removes all content above 16 kHz and 24 kHz
respectively. A katydid at 35 kHz would be embedded as near-silence and would
never match anything.

`--speed` reinterprets the source sample rate before resampling. At `0.25`,
content at 40 kHz is moved to 10 kHz, within the range of both models:

```bash
biotimbre add katydid.wav --sidecar katydid.txt --speed 0.25
```

The stored audio is never modified, and playback is unaffected.

**An item's speed does not need to be known when searching.** The query is
embedded automatically at every speed present in the library, so a recording
at its natural speed still finds a slowed-down item. Results report the speed
at which they matched.

If a declared `filtered_band` exceeds a model's limit, `add` suggests a speed
value. A suitable speed brings the highest frequency of interest below 16 kHz;
a factor of 1/4 or 1/8 suits most insect and bat recordings. Using the same
value across similar recordings matters more than the exact value chosen.

## Search Behaviour

A search embeds at most **15 s** of query audio. Scoring uses the
best-matching pair of windows, so a long query would allow an arbitrary
fragment to determine the result. Files within the limit are used whole;
longer files require `--start-time` and `--end-time` (or a selection in the
interface), and are refused rather than truncated. Only the selected frames
are decoded.

Filters are applied before scoring, so results are ranked within the filtered
set rather than filtered afterwards. Running the same query again with
different filters reuses the cached embeddings and returns immediately.

## Exporting Data

### Audio and Embeddings

The item dialog and each search result provide an **Export** menu with the
stored audio file, each model's embeddings, and a shareable bundle.

Each exported file holds at most one window, since a window — one vector over
one span of time — is the meaningful unit. A model with a single window (for
example Perch on a short item) is exported as a single `item_12_perch_v2.npy`
containing one 1-D vector. Several windows are exported as
`item_12_birdnet_v2.4.zip`, with one `.npy` per window, each named with its
index and time span (`item_12_birdnet_v2.4_w00_0.00-3.00s.npy`).
`biotimbre export-npy` writes exactly the same files, so a download from the
interface and a command-line export are interchangeable.

### Embeddings with Model Input Audio

The **embeddings + audio clips** entry of the Export menu, or
`biotimbre export-npy --with-audio`, also includes the audio of each window as
the model received it: speed-adjusted, resampled to the model's sample rate
and padded to the window length. The result is always a zip archive
(`item_12_birdnet_v2.4_with_audio.zip`) in which each `.npy` is accompanied by
a WAV file with the same name (`item_12_birdnet_v2.4_w00_0.00-3.00s.wav`). If
an item's stored windows no longer correspond to its audio, the export is
refused and the item must be re-embedded first.

### Metadata Tables (CSV)

Item metadata can be exported as a CSV table for use in a spreadsheet or a
dataframe, with one row per item and one column per stored value. Compound
values are split into separate columns (for example `latitude` and
`longitude`), missing values are left empty, and one column per registered
model gives the number of embedding windows. The file is UTF-8 encoded and
comma-separated.

- **Library page:** **Export CSV** in the *Library items* section, or
  `biotimbre export-csv OUT`, exports every item (`biotimbre_items.csv`).
- **Search page:** **Export CSV** next to the result count exports the
  displayed results only, in rank order (`biotimbre_search_results.csv`).
  Each row is preceded by the rank, the combined score, the best model, the
  matched window, the distance from the filter's centre, the query speed and
  the score of each model. For a search without a recording, the score
  columns are left empty.

## Sharing Items Between Libraries

`biotimbre_id` is a UUID so that a sound-being has the same identity in every
library that holds it. Items are transferred as **bundles**: a single `.btz`
file containing an item's metadata, its stored audio and its embedding vectors.

### Creating and Importing Bundles

Bundles can be created from the Export menu in the interface or from the
command line:

```bash
biotimbre export-bundle 12 -o guyalna.btz          # one item, with its vectors
biotimbre export-bundle 12 13 14 -o batch.btz      # several items
biotimbre export-bundle 12 -o small.btz --no-embeddings
```

The recipient can import the bundle in the *Import items* section of
the Library page, which shows a preview first, or from the command line. A
dry run reports what would happen without changing anything:

```bash
biotimbre import-bundle guyalna.btz --dry-run
biotimbre import-bundle guyalna.btz
```

A `.btz` file is a standard zip archive. It contains `item.json` (the
authoritative metadata), `item.txt` (the same fields in sidecar format, for
reading), the audio as a mono WAV file, and the vectors as `.npz` arrays.

### Import Rules

**Recordings already in the library.** Recordings are matched by the digest of
the stored file, so two copies are recognised regardless of what either
library calls them. Under the same biotimbre, the incoming item is the same
item and is skipped. Under a *different* biotimbre, the two libraries disagree
about which being it is; this is reported and the item is skipped.
`--duplicates add` overrides this.

**Species that already have a biotimbre.** When two libraries have each
catalogued *Guyalna bonaerensis*, there are two UUIDs for one animal. Following
the one-species-one-biotimbre rule, the incoming item joins the local
biotimbre, and the sender's ID is reported. `--species keep-incoming` keeps
the sender's ID instead, leaving two identities for one species until they are
merged.

**Embeddings.** Vectors are comparable only when the entire preprocessing
configuration matches. The bundle therefore carries each model's full
fingerprint — source, file, revision, sample rate, window length, overlap and
dimension — and the receiving library compares it field by field. Matching
vectors are stored as they are, saving an inference pass; mismatching vectors
are refused, the differing field is named, and the embeddings are recomputed.
Vectors from a model the receiving library does not use are always refused.
`--distrust-embeddings` discards all incoming vectors and recomputes them from
the audio, which is the only part of a bundle that can be verified.

Anything the bundle does not supply is computed on import, so an imported item
is immediately searchable. `--no-embeddings` leaves these gaps to be filled
later with `backfill`, which is useful for bulk imports on a machine without
the model files.

### Importing From Another Library

Every item of another library on the same computer can be imported at once,
without creating a bundle first. In the *Import items* section of the Library
page, **Import from another library** takes the other library's directory (the one
containing its `library.db`), typed in or chosen with **Browse**, and
**Preview import** opens the same preview and settings as for a bundle. From
the command line, `import-library` accepts the same options as
`import-bundle`:

```bash
biotimbre import-library ~/field-2024 --dry-run
biotimbre import-library ~/field-2024
```

The other library is opened read-only and is not changed. Each of its items is
treated as an entry of a bundle from that library, so the
[import rules](#import-rules) above apply unchanged: recordings already present
are skipped, which makes repeating an import safe; species join their local
biotimbre; and vectors are reused only where the model fingerprints match.
Items whose stored audio is missing from disk are left out and listed. The
other library must be at the current schema version; an older one is upgraded
by opening it once.

### Security

A bundle from another person is treated as untrusted. Only the members named
in its manifest are read, paths that would escape the archive are refused,
sizes and compression ratios are limited, arrays are loaded with pickling
disabled, every audio digest is verified, and every field undergoes the same
validation as a value entered in the form.

## iNaturalist Integration

Both pages accept an iNaturalist observation ID or URL, in a field below the
file uploader, and stage its recording exactly as an uploaded file, with the
same preview and controls.

On the **Library** page, the metadata is also filled in from the observation:
the date and time in the observation's own time zone, the coordinates, the
taxon, the observer as `recorded_by`, and the observation page as
`link_to_full_recording` (to which a trimmed item's `sample_window` then
refers). The licence and attribution are appended to `notes`. Every field
remains editable, and existing notes are extended rather than replaced.

On the **Search** page, the recording becomes the query, and the observation's
coordinates (with a 25 km radius) and taxon are entered in the filter panel,
which is opened so that the restriction is visible and can be cleared. The
date is not used, since an observation's date records when that recording was
made rather than when the animal vocalises.

In both cases, relevant caveats are reported. An obscured location is left
blank rather than filled with iNaturalist's randomised point, an
all-rights-reserved recording is flagged, and observations below research
grade are noted, since their identification may still change. A progress bar
is shown while the recording is downloaded.

Most iNaturalist audio is AAC in an `.m4a` container, which libsndfile cannot
read, so decoding requires **ffmpeg**. Without ffmpeg, the download fails with
an explanatory message and all other features continue to work. A typical
128 kbps upload contains nothing above approximately 16 kHz, whatever its
nominal sample rate: adequate for Perch, below BirdNET's 24 kHz, and
unsuitable for ultrasonic work.

The integration can be disabled for an offline library:

```toml
[sources.inaturalist]
enabled = false
```

## Location Map

Every item has coordinates (the columns are `NOT NULL`), so each page displays
**a single map**: the Library page plots every item, and the Search page plots
the current results. Each point represents one item; selecting it identifies
the item and offers a link to its details. A separate map per item is
deliberately avoided, since it would require one map instance and one set of
tile requests per result.

On the Search page, the map can also be used to set the location filter:
clicking elsewhere on the map sets the centre (opening the filter panel so
that the restriction is visible), and the centre and radius are drawn as a
point and a circle that follow the filter fields.

Leaflet is bundled with NiceGUI and served by the application, so no script is
loaded from a CDN. **Basemap tiles are the only exception** and the only
reason the map accesses the network. For a library kept offline, the map can
be disabled; coordinates are still used for filtering and still displayed on
every result:

```toml
[map]
enabled = false
```

For a single run, `biotimbre gui --no-map` has the same effect. Command-line
options take precedence over the configuration file and remain in effect when
a different library is opened from the browser.

Setting `tile_url` to a private tile server keeps map requests away from the
public one. Whichever server is used, `attribution` must credit it, and
`max_zoom` should match the deepest zoom level it serves to avoid blank tiles.

## Display Options

### Spectrogram and Waveform

Every image of a recording — in the item dialog, the add form, the search
query and each search result — can be displayed as a spectrogram or a
waveform. The toggle beside each image changes that image only. The most
recent choice is remembered per browser, together with the language and the
theme, and is used for images displayed afterwards.

The waveform shows two values per pixel column: the **peak** range (how far
the signal swings, so that clicks and wingbeats remain visible) as a light
band, and the **RMS** level (how loud the column is, which corresponds to
perceived loudness) drawn inside it. Amplitudes are scaled to the loudest
column, just as the spectrogram's decibel range is scaled to its loudest bin,
so a quiet recording is still displayed clearly rather than as a flat line.

The waveform is drawn as inline SVG rather than as a rendered image. This
allows it to use the theme's colours (an image is rendered on the server,
where the viewer's light or dark preference is not known) and to scale to any
width without another request.

### Colour Theme

The header provides a three-way control: **follow the system** (the default),
**light** and **dark**. The choice is remembered per browser, together with the
language, so that several people using one local instance can each choose
their own; the configuration file is not modified.

Each colour is declared once, as `light-dark(<light>, <dark>)` together with a
CSS `color-scheme`, so the two palettes cannot diverge, and native controls
that stylesheets cannot reach (scroll bars, the audio player, date pickers)
also follow the theme. Switching takes effect without reloading the page.

## Configuration

The configuration file `biotimbre.toml` is optional, and every setting has a
default value.

```toml
[library]
data_dir = "data"
# db_path   = "/fast/library.db"      # may be stored separately from the audio
# audio_dir = "/bulk/audio"           # stored as an absolute path if outside data_dir
# model_cache_dir = "/path/to/cache"  # default: the Hugging Face cache

[models]
active = ["perch_v2", "birdnet_v2.4"]
inference_batch_size = 1   # windows per model call; 1 is fastest here
inference_threads = 0      # 0 lets onnxruntime decide

[search]
top_k = 10
metric = "cosine"          # cosine | dot | euclidean
fusion = "max"             # max | mean | a model ID
batch_size = 8192
query_cache_entries = 16
max_query_duration_s = 15

[ui]
language = "en"            # en | pt_BR
search_only = false        # serve the Search page only, with a read-only library
dominant_frequency = true  # dashed, labelled line at the loudest frequency

[map]
# The location map on both pages. This is the only part of the interface that
# accesses the network merely by being displayed: Leaflet is served by this
# process, but the basemap tiles come from the URL below. An offline library
# should set enabled = false. Coordinates are still filtered on and displayed
# as text in either case.
enabled = true
# tile_url    = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
# attribution = "&copy; OpenStreetMap contributors"   # must credit the source
# max_zoom    = 19        # must match the tile server's deepest zoom level

[ui.theme]
# The light palette. Every key is optional; the values shown are the defaults.
# Colours are CSS values (hex, rgb(a) or named). primary..warning set Quasar's
# palette; the remaining keys style the header directly.
primary = "#118b50"
secondary = "#5ba98c"
accent = "#e9b44c"
dark = "#1d1d1d"
positive = "#118b50"
negative = "#c62828"
info = "#2f6f9f"
warning = "#e9b44c"
button_text = "#ffffff"        # label colour on filled primary buttons
page_bg = "#ffffff"            # page background below the header
text = "#1a1a1a"               # default label colour
header_bg = "#118b50"
header_text = "#ffffff"
header_inactive_text = "rgba(255, 255, 255, 0.65)"
header_active_bg = "rgba(255, 255, 255, 0.2)"
header_status_ready = "#a8e6a1"    # "Models ready" badge
header_status_loading = "#ffd166"  # "Loading models…" badge
header_status_failed = "#ff8a8a"   # model loading error badge

# The following keys style components whose colours Quasar and Tailwind
# otherwise fix regardless of ui.colors():
card = "#ffffff"               # card backgrounds (item dialog, forms)
link = "#3b82f6"               # hyperlinks (e.g. "open full recording")
muted = "#64748b"              # secondary text (hints, timestamps, IDs)
notice = "#92400e"             # inline warnings (speed shift, query limit, etc.)
surface = "#f1f5f9"            # statistics tile backgrounds on the Library page
chip = "#e2e8f0"               # small ID badge backgrounds
marker_border = "rgba(255, 255, 255, 0.8)"  # matched-window outline on a spectrogram
marker_fill = "rgba(255, 255, 255, 0.25)"

# Colours of the spectrogram images themselves. Evenly spaced #rrggbb stops
# from quiet (first) to loud (last); at least two are required. The ramp
# should increase monotonically in brightness so that intensity remains
# readable for colour-blind viewers. The default ramp runs from dark blue
# through magenta and orange to pale yellow.
spectrogram_colors = ["#0c0826", "#400c60", "#941a60", "#de5234", "#fca83c", "#fff4c8"]

[ui.dark_theme]
# The dark palette, with the same keys and meanings. Omitted keys fall back to
# the built-in dark defaults, never to the light values above, so overriding
# one colour does not bring the light palette into the dark theme.
# primary = "#5a9c3e"
# page_bg = "#111509"
# text = "#e7ecdc"
# card = "#171d10"
# spectrogram_colors is not read here: the images are rendered and cached on
# the server, where the viewer's theme is not known, so the [ui.theme] ramp
# is used for both.
```

## Data Layout

```
data/
├── library.db                        # metadata and embedding vectors
└── audio/<biotimbre_id>/<item_id>.wav
```

A library is backed up by copying its folder. Vectors are stored in the
database so that metadata filters can narrow the candidate set before any
vector is scored; `export-npy` provides the vectors as files when needed for
other tools.

## Adding a Model

Register the model in `models_registry.py`, then compute its embeddings for the
existing items:

```bash
biotimbre backfill MODEL_ID
```

Existing items are read again from their stored audio; no item needs to be
imported again.

## Development

```bash
uv run pytest                    # full suite, approximately 100 s
uv run pytest -m "not models"    # approximately 75 s: skips the 98 tests that run real models
```
