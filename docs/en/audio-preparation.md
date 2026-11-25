# Audio Preparation

*English · [Português (Brasil)](../pt-BR/audio-preparation.md)*

## Trimming, Filtering and Normalisation

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
version. Stored audio is also always WAV, as its name, its bundles and its
playback declare: a recording in another format (FLAC, OGG, MP3, AIFF) is
converted, a lossless one keeping every sample at its own bit depth and a lossy
one decoded, which costs no further quality. A mono WAV file that is neither
trimmed nor filtered is stored exactly as received.

## Ultrasonic Recordings and the Speed Setting

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
at which they matched. Setting a query speed instead (`search --query-speed`,
or *Query speed* on the Search page) embeds the query at that speed only, and
searches only the items stored at it.

If a declared `filtered_band` exceeds a model's limit, `add` suggests a speed
value. A suitable speed brings the highest frequency of interest below 16 kHz;
a factor of 1/4 or 1/8 suits most insect and bat recordings. Using the same
value across similar recordings matters more than the exact value chosen.
