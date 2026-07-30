-- Biotimbre library schema. See PLAN.md §3 for the rationale.
--
-- Metadata and embedding vectors share one file so that a metadata filter can
-- reduce the candidate set before any vector is scored.

CREATE TABLE IF NOT EXISTS biotimbres (
    biotimbre_id  TEXT PRIMARY KEY,          -- UUID; meant to be unique across libraries
    label         TEXT,                      -- optional human-readable name
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    item_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    biotimbre_id           TEXT    NOT NULL REFERENCES biotimbres(biotimbre_id),
    date                   TEXT    NOT NULL,   -- YYYY-MM-DD
    time                   TEXT    NOT NULL,   -- HH:MM:SS
    latitude               REAL    NOT NULL,
    longitude              REAL    NOT NULL,
    sample_window_start    REAL    NOT NULL,   -- seconds into the FULL recording
    sample_window_end      REAL    NOT NULL,
    device                 TEXT,
    filtered_band_low      REAL,               -- NULL = open (the '-' in sidecars)
    filtered_band_high     REAL,
    noise_reduction        INTEGER NOT NULL DEFAULT 0,
    speed                  REAL    NOT NULL DEFAULT 1.0,  -- <1 slower, >1 faster
    link_to_full_recording TEXT,
    notes                  TEXT,
    audio_path             TEXT    NOT NULL,   -- relative to the data directory
    audio_sha256           TEXT    NOT NULL,
    duration_s             REAL    NOT NULL,
    sample_rate            INTEGER NOT NULL,
    channels               INTEGER NOT NULL,
    created_at             TEXT    NOT NULL,
    updated_at             TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_biotimbre ON items(biotimbre_id);
CREATE INDEX IF NOT EXISTS idx_items_date      ON items(date);
CREATE INDEX IF NOT EXISTS idx_items_latlon    ON items(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_items_speed     ON items(speed);
CREATE INDEX IF NOT EXISTS idx_items_sha256    ON items(audio_sha256);

CREATE TABLE IF NOT EXISTS item_taxa (
    item_id INTEGER NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
    tag     TEXT    NOT NULL,                -- normalised lowercase
    PRIMARY KEY (item_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_taxa_tag ON item_taxa(tag);

CREATE TABLE IF NOT EXISTS models (
    model_id          TEXT PRIMARY KEY,      -- 'perch_v2', 'birdnet_v2.4'
    source            TEXT NOT NULL,         -- HF repo or URL
    filename          TEXT NOT NULL,
    revision          TEXT,
    sample_rate       INTEGER NOT NULL,
    window_duration_s REAL NOT NULL,
    window_overlap    REAL NOT NULL,
    embedding_dim     INTEGER NOT NULL,      -- detected from the ONNX graph
    registered_at     TEXT NOT NULL
);

-- One row per (item, model, window). start_time/end_time are stored rather than
-- derived from the model's window parameters, so they stay correct if a model is
-- re-registered with different parameters. Times are in the SAMPLE's time base,
-- already converted back from any speed shift (PLAN.md D2b.2).
CREATE TABLE IF NOT EXISTS embeddings (
    item_id      INTEGER NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
    model_id     TEXT    NOT NULL REFERENCES models(model_id),
    window_index INTEGER NOT NULL,
    start_time   REAL    NOT NULL,
    end_time     REAL    NOT NULL,
    vector       BLOB    NOT NULL,           -- float32, embedding_dim wide
    PRIMARY KEY (item_id, model_id, window_index)
);

CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model_id, item_id);

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
