# Metadata

*English · [Português (Brasil)](../pt-BR/metadata.md)*

## Fields

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
| `speed` | Playback speed used for embedding (see [Ultrasonic Recordings](audio-preparation.md#ultrasonic-recordings-and-the-speed-setting)). |
| `link_to_full_recording` | URL of the complete recording. |
| `notes` | Free text. |

## Identifiers

`item_id` is assigned automatically, never reused, and specific to one library;
it is a row identifier only.

A **biotimbre** is a sound-being: a sonotype whose species may be unknown.
`biotimbre_id` is therefore a UUID rather than a library-local counter, so that
the same sonotype is identified consistently when libraries are shared or
merged. Several items may share one `biotimbre_id` — the same species or
individual, with different sound content.

## One Species, One Biotimbre

A species is a complete identification, so the library assigns it a single
identity. An item tagged with a species already present in the library — for
example `Guyalna bonaerensis` — is filed under that species' existing
`biotimbre_id` instead of starting a second one for the same animal. Tags are
stored in lowercase, so capitalisation does not matter.

This rule applies to binomials only. A rank above species (`cicadidae`) does
not establish which being was recorded, and open nomenclature (`Guyalna sp.`,
`cf.`, `aff.`) explicitly states that the species is unknown; both start their
own biotimbre. An explicitly given `biotimbre_id` always takes precedence.

## Sidecar Files

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

The location may also be given as two lines, `latitude -20` and `longitude -46`,
the names a table uses. `filtered_band` accepts `-` for an open bound, e.g.
`2000,-`. Lines beginning
with `#` or `;` are comments. An unrecognised field name is reported as an
error rather than ignored, so a misspelling cannot silently discard a required
value.

A sidecar can also ask for the audio to be processed before it is stored, as
the add form's controls do. These lines are instructions, not description:

```
trim                5,10      # store only 5-10 s of the file
apply_highpass      2000      # zero-phase Butterworth highpass, in Hz
apply_lowpass       10000     # and lowpass
apply_filter_order  4         # optional; default 4
normalise           yes       # bring the stored copy's peak to -1 dBFS
```

The metadata, which describes the file as given, is corrected to describe what
is stored: a trim moves `sample_window` (to `5,10`, or by 5 s from where the
file already sat in a longer recording); a filter narrows `filtered_band` and
sets `filter_type` and `filter_order` if they are not given; normalisation adds
a line to `notes`. `biotimbre add` takes the same choices as `--trim`,
`--apply-highpass`, `--apply-lowpass`, `--apply-filter-order` and `--normalise`,
which replace the sidecar's line for the same choice. `import-dir` applies each
sidecar's own.

## Tables

`import-table` adds many items from one CSV table instead of a sidecar per
file. Each row is one item. Its columns are the names a sidecar takes, its
fields and processing lines (`latitude` and `longitude` or `location`), plus
the table's own `file`, `inat` and `inat_sound`:

```
file,inat,date,time,latitude,longitude,taxa,trim,normalise
GUYVIR.wav,,2020-01-12,09:40:21,-26.9,-48.7,cicadidae,"1,4",yes
,400139850,,,,,cicadidae,,
```

```bash
biotimbre import-table recordings/table.csv --dry-run   # check it, change nothing
biotimbre import-table recordings/table.csv
```

- **`file`**: a bare name is looked for in the table's folder, so the table is
  kept beside the recordings; a path relative to that folder also works, and an
  absolute path is used as given. A path written on another operating system
  (`C:\...` read on Linux, `/home/...` on Windows) is reported as such.
- **`inat`**: an iNaturalist observation ID or URL in place of `file`. Its
  recording is downloaded and its metadata fills whatever the row leaves empty,
  as `add --inat` does; `inat_sound` picks one of its recordings. Observations
  are requested 200 at a time and recordings one at a time, at most one request
  a second and under 5 GB of audio an hour, following iNaturalist's recommended
  practices, and a request refused as too many is retried after the wait it
  names. Long waits are announced.
- A table may be separated by commas, semicolons or tabs. In a
  semicolon-separated table a single number may use a decimal comma (`-26,9`);
  pairs such as `trim` always need decimal points.
- An unknown column is an error. An empty cell gives nothing. Every row is
  tried, and each is reported with its line number.
- A recording already in the library is **skipped**, so an interrupted import
  can be run again; `--keep-duplicates` stores it again instead.
