# Biotimbre Library

A local library of animal vocalization samples with embedding-based similarity
search, using the Perch v2 and BirdNET 2.4 ONNX backbones via `onnxruntime`.

See [PLAN.md](PLAN.md) for the design and the reasoning behind it.

## Status

Phases 1–5 complete: the library core and a CLI. 387 tests pass.

Not built yet: the two GUI pages (phases 6–7), the translated interface
(phase 8), and the public read-only search endpoint (phase 9).

## Setup

```bash
uv venv --python 3.12
uv sync --extra dev
uv run biotimbre init          # downloads ~70 MB of models on first run
```

Models are downloaded once and then resolved from the local cache, so everything
after `init` works with no network.

## Usage

```bash
# Add samples. Metadata comes from a sidecar .txt beside the audio…
uv run biotimbre add recording.wav

# …or from flags. Note the scalar coordinate options: argparse cannot accept
# "--location -26,-48" because the value starts with a minus.
uv run biotimbre add recording.wav \
    --date 2021-03-05 --time 22:10:00 \
    --latitude -26.9 --longitude -48.7 \
    --window-start 12 --window-end 17 \
    --taxa orthoptera insecta --device Audiomoth \
    --highpass 8000 --lowpass 40000

# Bulk import every sidecar-paired file in a directory
uv run biotimbre import-dir input_example

# Search with a short query recording (15 s limit, see below)
uv run biotimbre search query.wav -k 5

# For a longer file, pick the segment to search with
uv run biotimbre search dawn_chorus.wav --start-time 92 --end-time 99

# Filter before scoring — cheaper and more correct than filtering afterwards
uv run biotimbre search query.wav --taxa cicadidae \
    --latitude -26 --longitude -48 --radius-km 50 \
    --date-from 2020-01-01

uv run biotimbre list
uv run biotimbre show 1
uv run biotimbre stats
uv run biotimbre models
uv run biotimbre delete 1
```

## Ultrasonic samples: the `speed` field

Embedding resamples audio to each model's rate — 32 kHz for Perch, 48 kHz for
BirdNET — which anti-alias-filters away everything above 16 kHz and 24 kHz
respectively. A katydid calling at 35 kHz would therefore be embedded as
near-silence, and would never match anything.

`--speed` fixes this by reinterpreting the source rate before resampling.
At `--speed 0.25`, content at 40 kHz lands at 10 kHz, comfortably inside both
models' range:

```bash
uv run biotimbre add katydid.wav --sidecar katydid.txt --speed 0.25
```

The stored audio is never modified — speed applies only at inference time — and
playback is unaffected.

**You do not need to remember an item's speed when searching.** The query is
automatically embedded at every speed present in the library, and each item is
scored against the query embedded at its own speed. A natively-recorded query
still finds a slowed-down library item. Hits report the speed that produced them.

If you declare a `filtered_band` whose upper bound exceeds a model's limit,
`add` prints an advisory suggesting a speed value rather than silently producing
a useless embedding.

### Choosing a speed

Pick a value that brings your highest frequency of interest below the *lower* of
the two model limits (16 kHz). Dividing by 4 or 8 covers most insect and bat
material. Being consistent across similar recordings matters more than the exact
value — two curators indexing the same species at 0.25 and 0.1 will get
less-comparable embeddings, though both stay findable.

## Adding a model later

Register it in `models_registry.py`, then:

```bash
uv run biotimbre backfill <model_id>
```

Existing items are re-read from their stored audio; nothing needs re-importing,
because the stored files were never model-specific.

## Data layout

```
data/
├── library.db                       # metadata AND embedding vectors
└── audio/<biotimbre_id>/<item_id>.wav
```

Vectors live in the database rather than in per-item files so that a metadata
filter can narrow the candidate set before any vector is scored. Use
`export-npy` when you need a `.npy` for interop:

```bash
uv run biotimbre export-npy 1 perch_v2 ./vectors/
```

## Configuration

Optional `biotimbre.toml` in the working directory, or `$BIOTIMBRE_CONFIG`:

```toml
[library]
data_dir = "data"
# model_cache_dir = "/path/to/onnx/cache"   # default: the Hugging Face cache

[models]
active = ["perch_v2", "birdnet_v2.4"]
inference_batch_size = 1   # windows per model call; 1 is fastest here
inference_threads = 0      # 0 lets onnxruntime choose

[search]
top_k = 10
metric = "cosine"          # cosine | dot | euclidean
fusion = "max"             # max | mean | a model id
batch_size = 8192          # embedding rows scored per batch
query_cache_entries = 16   # query audios whose embeddings stay in memory
max_query_duration_s = 15  # longest segment a search will embed

[ui]
language = "en"        # en | pt_BR
```

## Search queries are short by design

A search is limited to **15 seconds** of query audio. This library matches short
vocalization samples, and a long query is not just slow — it is analytically
weaker, because scoring takes the best-matching window pair, so a whole recording
lets an arbitrary fragment decide the result.

A file within the limit is used whole. A longer one needs an explicit segment,
which is refused rather than guessed:

```
$ biotimbre search dawn_chorus.wav
error: query_segment: Query audio is 60.0 s, but a search is limited to 15 s.
Choose a start and end time at most 15 s apart — this library matches short
vocalizations, so searching a whole recording dilutes the query rather than
improving it.

$ biotimbre search dawn_chorus.wav --start-time 92 --end-time 99
```

Only those frames are decoded, so a long recording is never loaded in full. The
limit is checked from the file header, before any decoding or model loading.

Raise `max_query_duration_s` if you really need to, knowing the trade-off.

## Search performance

Embedding the query dominates a search — milliseconds go to the vector scoring
itself. Query embeddings are therefore **cached in memory per
(audio, segment, model, speed)**, so adjusting filters and re-running the same
query is effectively free:

| | first search | re-run with a different filter |
|---|---|---|
| 5 s query | 5.2 s | **0.003 s** |
| 15 s segment | 16.2 s | **0.003 s** |

Keying per model *and* per speed means a filter that newly admits items at a
previously unused speed computes just that one entry rather than discarding the
sweep. A fully cached query does not re-read the file at all. The cache lives for
the life of a `LibraryService`, holds `query_cache_entries` distinct queries
(LRU), and is not used during ingest, where every item is distinct and caching
would only grow memory.

### Why inference runs one window at a time

Measured on 16 windows, best of 3, threads on automatic:

| model | batch 1 | batch 16 |
|---|---|---|
| `perch_v2` | **1.59 s** | 3.41 s |
| `birdnet_v2.4` | 2.50 s | 2.28 s |

Batching makes Perch **2.1× slower** and helps BirdNET only within run-to-run
noise. These backbones do a lot of compute per window — 5 s at 32 kHz is 160 000
samples — so a batch mainly enlarges the working set past cache instead of
exposing parallelism onnxruntime was not already using.

Per-window is therefore both fastest and the tightest memory bound, which still
matters: peak memory is roughly 19 MB per BirdNET window, and `speed` can go as
low as 0.01, multiplying window count by up to 100× despite the query length cap.

Threading, by contrast, is worth leaving on (`inference_threads = 0`): automatic
beat one thread in every configuration measured, by 2.5× for Perch.

## Development

```bash
uv run pytest                    # everything, ~85 s
uv run pytest -m "not models"    # ~1 s: skips the 50 tests that run real models
```

Only 50 of the 371 tests load models; the rest run in about a second. ONNX
sessions are shared across the whole run via a session-scoped fixture (creating
one costs ~2 s: Perch 1.56 s, BirdNET 0.42 s), so the remaining time is genuine
inference rather than repeated setup.

## Metadata sidecar format

One field per line, name then value, separated by tabs or spaces. Values may
contain spaces. Absent `taxa` means unknown.

```
taxa	cicadidae
date	2020-01-14
time	09:40:21
sample_window 5,10
location -20,-46
device	Audiomoth
filtered_band	2000,10000
```

`filtered_band` accepts `-` for an open bound, e.g. `2000,-`. Also recognised:
`biotimbre_id` (to file a sample under an existing biotimbre),
`noise_reduction`, `speed`, `link_to_full_recording`, `notes`. Lines starting
with `#` or `;` are comments. An unrecognised field name is an error rather than
being ignored, so a typo cannot silently drop a mandatory value.
