# Biotimbre Library — Implementation Plan

Working document. Derived from `biotimbre_prompt.txt` plus review of the two
sample files in `input_example/`, the legacy `../templates/` library, and the
Auricularia prototype (used as a reference only — **no dependency on its
packages**).

Status: plan agreed, implementation not started.

---

## 1. Established facts

These were measured, not assumed, and they constrain the design.

| Thing | Value |
|---|---|
| Example samples | 5.0 s, 32 kHz, mono, 16-bit PCM |
| **Sample length in general** | **Arbitrary.** The 5 s examples are incidental, not a constraint. |
| Perch v2 backbone | 45 MB ONNX, 32 kHz input, 5.0 s window |
| BirdNET 2.4 backbone | 25 MB ONNX, 48 kHz input, 3.0 s window |
| Legacy biotimbre IDs | 4-digit decimal dirs (`0000`…`0040`) with JSON sidecars |
| Sidecar metadata quirks | `filtered_band 2000,-` (open-ended high cut), `taxa` absent in `GUYVIR.txt`, `location -26,-48` |
| System Python | 3.8.10 — unusable; uv must manage a 3.12 toolchain |

Consequence of variable-length samples: **one sample does not produce one
vector.** A 20 s sample yields 4 Perch windows and 7 BirdNET windows. Window
count differs per item *and* per model. Every downstream decision follows from
this.

Embedding dimensions (Perch ≈1536, BirdNET ≈1024) are **detected from the ONNX
graph at model registration**, never hardcoded.

---

## 2. Decisions

### D1 — Storage: single SQLite database, no per-item files

Metadata *and* embedding vectors live in one `library.db`. One row per
`(item_id, model_id, window_index)` with the vector as a float32 BLOB.

Rationale:

- **Pre-filtering.** Filters (taxa, location, date) and vectors are queryable
  together, so a selective filter reduces the set *before* scoring. "Cicadidae
  within 50 km" may be 200 windows out of 50,000. If vectors sat in files, a
  metadata query couldn't reach them: you would either score the whole library
  and filter afterwards (wasteful, and the top-k was computed over the wrong
  population) or gather hundreds of scattered small files — the slow path
  exactly when the filter is most selective.
- **Atomicity.** Add/edit/delete is one transaction. No orphaned vector files,
  no cache to invalidate.
- **Ragged shapes.** Variable `n_windows` per item and model can't be memmapped
  as one contiguous array; it needs a flat matrix plus index arrays. A flat
  table keyed on `(item_id, model_id, window_index)` *is* that structure, and
  SQLite already implements it correctly.
- **Upgrade path.** `sqlite-vec` adds ANN to this schema in place; the same
  table shape ports to Postgres + pgvector for a public deployment.

`.npy` is provided as an **`export-npy` / `import-npy` command** for interop and
hand-off, not as the on-disk source of truth. (The prompt mentioned `.npy`;
confirmed as descriptive rather than a hard requirement.)

> Superseded: an earlier draft made `.npy`-per-item the source of truth with a
> rebuilt consolidated matrix cache for search speed. That kept two copies of
> the same data plus an invalidation path — strictly more complexity than the
> database alone, in exchange for transparency that the export command
> delivers just as well.

Audio files stay on the filesystem (`data/audio/<biotimbre_id>/<item_id>.wav`),
referenced by relative path plus a SHA-256 for integrity and duplicate
detection. Only the vectors move into the DB.

### D1b — Audio is stored exactly as submitted

The WAV goes into `data/audio/` byte-for-byte as the user added it. Sample rate,
bit depth and channel count are recorded in `items` but never altered.

**Resampling happens only inside the embedding extractor**, in memory, per model
— Perch at 32 kHz, BirdNET at 48 kHz from the same untouched source file. The
library's copy is the archival original; model input is a derived, transient
view of it. This also means re-registering a model with different parameters, or
adding a third model later, needs no re-import.

### D2 — Windowing: keep every window

Store all windows per item per model. Do **not** mean-pool.

Mean-pooling would make a 4 s and a 90 s sample both collapse to one vector,
discarding everything that distinguishes them, and any partly zero-padded
trailing window drags the mean toward silence.

**Ingest overlap defaults to 50 %.** At 0 % a vocalization straddling a window
boundary is split and scores poorly in both halves. Storage cost is trivial
(a 90 s Perch sample ≈ 220 KB at 50 % overlap). Configurable per model.

Samples shorter than a model's window are zero-padded — an edge case rather
than the norm.

### D2b — Optional speed shift, stored as item metadata

`items.speed` — REAL, default `1.0`. `<1.0` slows down, `>1.0` speeds up. It is
metadata *and* an embedding-time parameter: it is applied before resampling, so
it determines the vectors that item is indexed by.

Implementation is a reinterpretation, not a filter: playing a file at speed `s`
means its effective sample rate is `sr_native × s`, so the extractor resamples
from `sr_native × s` to the model rate instead of from `sr_native`. Frequencies
scale by `s`, duration by `1/s`. No extra DSP stage, and only one resampling
step — so no compounded interpolation loss.

This is what makes ultrasonic material searchable: at `speed = 0.25`, energy at
40 kHz lands at 10 kHz, comfortably inside Perch's 16 kHz band, instead of being
erased by the anti-alias filter. It is the remedy for the Nyquist risk in §7,
which downgrades that risk from "silently wrong embeddings" to "set the field".

Because `sr_native × s` is generally not an integer, the resample ratio is
derived with `Fraction(...).limit_denominator()` rather than integer `gcd`.

Three consequences that are easy to get wrong:

1. **Stored audio and `duration_s` are unaffected.** Speed never touches the
   archived file (D1b) and never affects playback (D8). It is an inference-time
   view only.
2. **Window times are stored in the original sample's time base**, i.e. the
   extractor's shifted-domain times multiplied back by `s`. Otherwise the
   matched-window marker in the player (D8) would point at the wrong region for
   every item with `speed ≠ 1.0`.
3. **Editing `speed` invalidates that item's embeddings** and must trigger
   re-computation. Unlike `notes` or `taxa`, it is not a free-text edit. The
   repository enforces this; the phase 6 edit form warns about it.

Validation: `speed > 0`, bounded to `[0.01, 100]`.

### D2c — Query-side speed sweep

An item embedded at `speed = 0.25` sits in a different region of embedding space
than the same species embedded at `1.0`. So a naive query of natively-recorded
ultrasound would **never** match a slowed-down library item — the feature would
appear to work at ingest and silently fail at search.

Rather than making the user guess the right speed, search embeds the query at
`1.0` **and at every distinct speed present in the library**
(`SELECT DISTINCT speed FROM items`), scores each, and keeps the best per item.
Library items stay single-speed — whatever the curator chose — while the query
side adapts automatically. The cost is a handful of extra query inferences on one
short file, which is negligible; the library side, which is the expensive side,
is untouched.

The speed that produced each winning score is returned with the result, so an
ultrasonic match is visible as such.

Migration note: if per-item multi-speed indexing is ever wanted, `speed` moves
from `items` into the `embeddings` primary key. Keeping it on `items` for now is
the simpler correct thing, not a shortcut that has to be undone.

### D3 — Scoring: max over window pairs

Query audio is also multi-window. Per model, scoring is a
`(q_windows × db_windows)` cosine matrix reduced to **one score per item by max
over all pairs** — a 60 s query should match a 3 s library item that shares one
syllable.

The winning `(query_window, library_window)` pair is retained and surfaced in
results: when curating, knowing *where* the match occurred matters as much as
the score.

### D4 — Cross-model fusion

Score each model independently, then fuse. Default **max of per-model cosine**;
`mean` and single-model modes selectable. The per-model breakdown is always
returned so a lopsided fusion is visible rather than hidden.

Concatenating the two embedding spaces is not viable: different dims, different
scales, and it breaks for any item missing one model's embeddings.

Caveat to revisit empirically: cosine from two different models is only
*roughly* comparable. If one model's scores prove systematically higher,
per-model score normalisation (z-score against the library distribution) is the
fix. Not doing it upfront — it needs real data to calibrate.

### D5 — Layered architecture, thin frontend

All logic behind a `LibraryService` class. The GUI, the CLI, and any future HTTP
API are peers that call it and hold no logic of their own. This is what makes
the eventual public search tool a frontend swap rather than a rewrite.

### D6 — GUI: NiceGUI (deferred to phase 6)

Runs on FastAPI, so exposing a public read-only search endpoint later is a
config change in the same process. Real event-driven forms and tables, which the
admin CRUD page needs. Revisitable at phase 6 — D5 is what actually protects
this choice, not the framework itself.

### D7 — i18n: JSON catalogues

`en.json` / `pt_BR.json` plus a small `t(key)` helper. No gettext compile step
for two languages, and the catalogues port verbatim to i18next if a JS frontend
ever happens.

### D8 — Search results are audible

Every result row carries an audio player. Listening to the top hits *is* the
verification step — a similarity score is a claim the user has to check by ear.

This is not purely a UI concern; it puts two things in the service layer:

- **`LibraryService.get_audio(item_id)`** returns the stored file (path or byte
  stream) plus its MIME type. Both the GUI and any future HTTP API serve audio
  *through* this call rather than by exposing `data/audio/` as a static
  directory — which is also the single place to enforce access control if the
  search tool goes public.
- **Search results carry the matched window's `start_time`/`end_time`** (already
  retained per D3), so the player can mark or jump to the segment that actually
  produced the score, rather than making the user hunt through a long sample.

The query audio stays playable alongside the results, since the real workflow is
A/B: hear the query, then hear candidate 1, 2, 3. Served as the original file
per D1b — no transcoding.

### D9 — IDs

- `item_id`: SQLite `INTEGER PRIMARY KEY AUTOINCREMENT`. Never reused, so a
  deleted item's ID does not come back.
- `biotimbre_id`: 4 uppercase hex chars (`0000`–`FFFF`, 16-bit as specified).
  Allocated as lowest unused value; overridable so an item can join an existing
  biotimbre. Tracked in its own table so an ID can exist with an optional
  human-readable label independent of its items.

---

## 3. Schema

Single file `data/library.db`, WAL mode, foreign keys on.

```sql
CREATE TABLE biotimbres (
    biotimbre_id  TEXT PRIMARY KEY,          -- '0000'..'FFFF'
    label         TEXT,                      -- optional human name
    created_at    TEXT NOT NULL
);

CREATE TABLE items (
    item_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    biotimbre_id        TEXT    NOT NULL REFERENCES biotimbres(biotimbre_id),
    date                TEXT    NOT NULL,    -- YYYY-MM-DD
    time                TEXT    NOT NULL,    -- HH:MM:SS
    latitude            REAL    NOT NULL,
    longitude           REAL    NOT NULL,
    sample_window_start REAL    NOT NULL,    -- seconds into the FULL recording
    sample_window_end   REAL    NOT NULL,
    device              TEXT,
    filtered_band_low   REAL,                -- NULL = open (the '-' in sidecars)
    filtered_band_high  REAL,
    noise_reduction     INTEGER NOT NULL DEFAULT 0,
    speed               REAL    NOT NULL DEFAULT 1.0,  -- <1 slower, >1 faster
    link_to_full_recording TEXT,
    notes               TEXT,
    audio_path          TEXT    NOT NULL,    -- relative to data dir
    audio_sha256        TEXT    NOT NULL,
    duration_s          REAL    NOT NULL,
    sample_rate         INTEGER NOT NULL,
    channels            INTEGER NOT NULL,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL
);
CREATE INDEX idx_items_biotimbre ON items(biotimbre_id);
CREATE INDEX idx_items_date      ON items(date);
CREATE INDEX idx_items_latlon    ON items(latitude, longitude);

CREATE TABLE item_taxa (
    item_id INTEGER NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
    tag     TEXT    NOT NULL,                -- normalised lowercase
    PRIMARY KEY (item_id, tag)
);
CREATE INDEX idx_taxa_tag ON item_taxa(tag);

CREATE TABLE models (
    model_id          TEXT PRIMARY KEY,      -- 'perch_v2', 'birdnet_v2.4'
    source            TEXT NOT NULL,         -- HF repo or URL
    filename          TEXT NOT NULL,
    revision          TEXT,
    sample_rate       INTEGER NOT NULL,
    window_duration_s REAL NOT NULL,
    window_overlap    REAL NOT NULL,
    embedding_dim     INTEGER NOT NULL,      -- detected from ONNX graph
    registered_at     TEXT NOT NULL
);

CREATE TABLE embeddings (
    item_id      INTEGER NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
    model_id     TEXT    NOT NULL REFERENCES models(model_id),
    window_index INTEGER NOT NULL,
    start_time   REAL    NOT NULL,           -- seconds within the SAMPLE
    end_time     REAL    NOT NULL,
    vector       BLOB    NOT NULL,           -- float32, embedding_dim wide
    PRIMARY KEY (item_id, model_id, window_index)
);
CREATE INDEX idx_emb_model ON embeddings(model_id, item_id);

CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
```

Two deliberate notes:

- **`start_time`/`end_time` are stored per row**, not derived from model metadata
  the way the Auricularia prototype does it. Per-row costs 8 bytes and stays
  correct if a model's window parameters ever change between ingests.
- **Three different time axes exist.** `items.sample_window_*` locates the sample
  inside the original full recording. `embeddings.start_time` is relative to the
  start of the sample itself. The extractor additionally works in a
  speed-shifted domain, but never persists those times — they are converted back
  to sample-relative before storage (D2b.2). Naming keeps the first two
  distinct; the third stays inside the extractor.

---

## 4. Layout

```
biotimbre-library/
├── pyproject.toml              # uv, Python 3.12
├── PLAN.md
├── README.md
├── data/                       # gitignored
│   ├── library.db
│   └── audio/<biotimbre_id>/<item_id>.wav
├── src/biotimbre/
│   ├── config.py               # TOML settings: paths, language, active models
│   ├── models_registry.py      # ModelSpec definitions + HF resolution
│   ├── domain/
│   │   ├── item.py             # LibraryItem, SearchResult, validation
│   │   └── errors.py
│   ├── db/
│   │   ├── schema.sql
│   │   ├── connection.py       # WAL, FK pragma, migrations
│   │   └── repository.py       # CRUD, biotimbre allocation, embedding rows
│   ├── embeddings/
│   │   ├── onnx_backend.py     # session lifecycle, dim detection
│   │   ├── audio.py            # load, mono, resample, window, pad
│   │   └── extractor.py        # audio -> (n_windows, dim) + window times
│   ├── search/
│   │   ├── vector_search.py    # cosine/dot, max-over-pairs, fusion
│   │   └── filters.py          # SQL pre-filter + haversine refine
│   ├── service.py              # LibraryService — the single API
│   │                           #   incl. get_audio() for playback (D8)
│   ├── io/sidecar.py           # parse the input_example/*.txt format
│   ├── i18n/{__init__.py,en.json,pt_BR.json}
│   ├── cli.py
│   ├── api/                    # phase 9, FastAPI read-only search
│   └── ui/                     # phase 6-7, NiceGUI pages
└── tests/
```

---

## 5. Search flow

1. Compute query embeddings for every registered model, each at its own sample
   rate / window / overlap, and at every distinct library speed plus `1.0`
   (D2c). Yields one window matrix per `(model, speed)` pair.
2. Build the candidate `item_id` set from metadata filters in SQL. Location uses
   a bounding-box pre-filter (index-friendly) refined by exact haversine in
   Python.
3. Per model, stream candidate windows batched from `embeddings`, cosine-score
   against the query window matrix **for the query speed matching that item's
   `speed`**, reduce max-over-pairs per item, retain the argmax window pair.
4. Fuse per-model item scores per D4.
5. Order, apply `top_k`, hydrate full metadata, return — including each hit's
   matched window times, per-model scores, and the speed used.

Implementation note: for large candidate sets, insert IDs into a temp table and
JOIN rather than building a giant `IN (...)` list — SQLite has a host-parameter
limit and long `IN` lists degrade.

---

## 6. Phases

Each phase ends in something verifiable. Phases 1–5 are the agreed scope for
this pass; review before phase 6.

**Phase 1 — Scaffold.** `uv` venv on Python 3.12, dependencies
(`numpy soundfile scipy onnxruntime huggingface_hub`, dev `pytest`), `config.py`,
`models_registry.py` with both model specs.
*Done when:* `uv run python -c "import biotimbre"` works.

**Phase 2 — Domain + DB.** `LibraryItem` with validation (date/time formats,
lat/lon ranges, `start < end`, hex ID, nullable filtered band, taxa
normalisation), errors, schema, connection with migrations, repository CRUD,
biotimbre ID allocation.
*Done when:* CRUD + allocation + validation tests pass, including the
open-ended `2000,-` band and missing-taxa cases from the real sidecars.

**Phase 3 — Embeddings.** ONNX wrapper with dimension detection, audio
load/mono/resample/window/pad, extractor producing `(n_windows, dim)` plus
window times, persistence to `embeddings`, `export-npy`/`import-npy`.
Speed shift (D2b) applied before resampling, window times converted back to the
sample time base.
*Done when:* both models run on both example files, dims are reported from the
graph, and window counts match the verified arithmetic below. Plus: the same
file at `speed = 0.5` yields 3 Perch windows, with window times still spanning
`[0, duration_s]` in the sample time base rather than the shifted one.

Window counts for a 5.0 s sample (measured, `models_registry.n_windows`):

| model | 0 % overlap | 50 % overlap |
|---|---|---|
| `perch_v2` (5.0 s window @ 32 kHz) | 1 | **1** |
| `birdnet_v2.4` (3.0 s window @ 48 kHz) | 2 | 3 |

Perch gives 1 window at *both* overlaps — a 5 s window over 5 s of audio leaves
no room for a second hop, and a trailing partial hop is zero-padded only when it
would otherwise produce no window at all. An earlier draft of this criterion
predicted 2, which would have failed a correct implementation.

**Phase 4 — Search.** Scoring, max-over-pairs reduction, fusion, SQL filters +
haversine.
Query-side speed sweep (D2c).
*Done when:* synthetic-vector tests with known-correct answers pass, including
a case where a filter changes the top-k, and a case where the two models
disagree. Plus an end-to-end regression for D2c: an item ingested at
`speed = 0.25` is found by a query of the same audio submitted at native speed —
the case that would silently fail without the sweep.

**Phase 5 — Service + CLI.** `LibraryService`, sidecar parser, CLI (`init`,
`add`, `import-dir`, `list`, `show`, `delete`, `search`, `models`, `backfill`,
`export-npy`).
*Done when:* both examples ingest from `input_example/` including their `.txt`
sidecars, and searching one against the library returns it as the top hit with
a sane score.

**Phase 6 — Admin GUI.** Upload, mandatory/optional field split, auto-embed on
add, edit, delete with cascade, variable-length waveform preview.

**Phase 7 — Search GUI.** Query upload, filter panel, ranked results with
per-model breakdown. Each result row has its own audio player (D8) with the
matched window marked; the query stays playable at the top of the page for A/B
comparison. The marker must **clamp `end_time` to the sample duration**: a
zero-padded final window legitimately reports a span past the end of the file
(BirdNET's third window ends at 6.0 s on a 5 s sample), and seeking there would
fail.
*Done when:* a query can be run and every returned hit played back, including
the query itself, without leaving the page.

**Phase 8 — i18n.** String extraction, EN + pt-BR catalogues, switcher.

**Phase 9 — Polish.** README, `backfill` for adding a third model to an existing
library, read-only public search endpoint behind a flag, auth stub.

---

## 7. Risks

- **ONNX input contracts unverified.** Whether either graph accepts a dynamic
  batch dimension is unknown; if not, inference is per-window in a loop. Phase 3
  establishes this before anything is built on top of it.
- **BirdNET preprocessing.** Assumed raw 48 kHz float32 audio in, matching the
  prototype's config. If the backbone expects mel spectrograms instead, phase 3
  surfaces it immediately via a shape mismatch.
- **Cross-model score comparability.** See D4 caveat.
- **Content above a model's Nyquist is lost at inference time** *unless* `speed`
  is set. Resampling to 32 kHz (Perch) or 48 kHz (BirdNET) anti-alias-filters
  away everything above 16 / 24 kHz, so a strongly ultrasonic sample left at
  `speed = 1.0` gets an embedding of whatever little remains. Stored audio and
  playback are unaffected (D1b, D8); only that item's searchability degrades.
  D2b is the fix, so the residual risk is just *forgetting to apply it*.
  Mitigation: at ingest, compare `filtered_band_high × speed` against each
  model's Nyquist and record a per-item advisory flag suggesting a speed value —
  turning a silent failure into a prompt. Phase 3 if cheap, otherwise phase 9.
- **Speed is a curator judgement call.** Nothing validates that a chosen speed is
  *good*, only that it is positive. Two curators could index similar ultrasonic
  material at 0.25 and 0.1 and get less-comparable embeddings. The D2c sweep
  makes both findable, but a convention documented in the README is worth more
  than any code here.
- **First run downloads 70 MB** from Hugging Face and needs network. Cache path
  is configurable for offline/air-gapped use.
- **Working tree is not a git repo.** Worth `git init` before phase 2 so the
  schema has history; not done without asking.
