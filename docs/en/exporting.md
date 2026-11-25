# Exporting Data

*English · [Português (Brasil)](../pt-BR/exporting.md)*

## Audio and Embeddings

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

## Embeddings with Model Input Audio

The **embeddings + audio clips** entry of the Export menu, or
`biotimbre export-npy --with-audio`, also includes the audio of each window as
the model received it: speed-adjusted, resampled to the model's sample rate
and padded to the window length. The result is always a zip archive
(`item_12_birdnet_v2.4_with_audio.zip`) in which each `.npy` is accompanied by
a WAV file with the same name (`item_12_birdnet_v2.4_w00_0.00-3.00s.wav`). If
an item's stored windows no longer correspond to its audio, the export is
refused and the item must be re-embedded first.

## Metadata Tables (CSV)

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
  Each row is preceded by the rank, the ranking model's score, that model, the
  matched window, the distance from the filter's centre, the query speed, and
  the score and rank of each model. For a search without a recording, the score
  columns are left empty.
