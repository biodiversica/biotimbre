# Sharing Items Between Libraries

*English · [Português (Brasil)](../pt-BR/sharing.md)*

`biotimbre_id` is a UUID so that a sound-being has the same identity in every
library that holds it. Items are transferred as **bundles**: a single `.btz`
file containing an item's metadata, its stored audio and its embedding vectors.

## Creating and Importing Bundles

Bundles can be created from the Export menu in the interface or from the
command line:

```bash
biotimbre export-bundle 12 -o guyalna.btz          # one item, with its vectors
biotimbre export-bundle 12 13 14 -o batch.btz      # several items
biotimbre export-bundle 12 -o audio-only.btz --no-embeddings   # metadata and audio only
```

Search results can be shared the same way. `search --csv` writes the results
as the Search page exports them; delete the rows of any wrong matches in a
spreadsheet, and `--items-from` takes the items that remain, from that file or
any CSV with an `item_id` column:

```bash
biotimbre search query.wav --taxa cicadidae --csv hits.csv
biotimbre export-bundle --items-from hits.csv -o checked.btz
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

## Import Rules

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

## Importing From Another Library

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

## Security

A bundle from another person is treated as untrusted. Only the members named
in its manifest are read, paths that would escape the archive are refused,
sizes and compression ratios are limited, arrays are loaded with pickling
disabled, every audio digest is verified, and every field undergoes the same
validation as a value entered in the form.
