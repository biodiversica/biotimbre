# Biotimbre Library

A local library of animal vocalization samples with embedding-based similarity
search, using the Perch v2 and BirdNET 2.4 ONNX backbones via `onnxruntime`.

See [PLAN.md](PLAN.md) for the design and the reasoning behind it.

## Status

Phases 1–5 complete: the library core and a CLI. 325 tests pass.

Not built yet: the two GUI pages (phases 6–7), the translated interface
(phase 8), and the public read-only search endpoint (phase 9).

## Setup

```bash
uv venv --python 3.12
uv sync --extra dev
uv run biotimbre init          # downloads ~70 MB of models on first run
```

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

# Search with a query recording
uv run biotimbre search query.wav -k 5

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

[search]
top_k = 10
metric = "cosine"      # cosine | dot | euclidean
fusion = "max"         # max | mean | a model id
batch_size = 8192

[ui]
language = "en"        # en | pt_BR
```

## Development

```bash
uv run pytest                    # everything
uv run pytest -m "not models"    # skip tests needing the ONNX files (offline)
```

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
