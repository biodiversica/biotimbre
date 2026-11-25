# iNaturalist Integration

*English · [Português (Brasil)](../pt-BR/inaturalist.md)*

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

When an observation has several recordings, the one used is chosen from a
list, and the credit in `notes` is that recording's own licence.

From the command line, `--inat` takes the place of the audio file:

```bash
biotimbre add --inat 400139850                    # fetch it and fill the metadata
biotimbre add --inat 400139850 --trim 2 7 --taxa cicadidae   # options take precedence
biotimbre search --inat 400139850 --start-time 2 --end-time 7
biotimbre search --inat 400139850 --inat-filters  # also narrow by its taxon and place
```

`add --inat` fills the same fields as the Library page; any metadata option
given replaces what the observation says, and an obscured location has to be
given with `--latitude` and `--longitude`. `search --inat` uses the recording
as the query only: unlike the Search page, it applies the observation's taxon
and 25 km radius only with `--inat-filters`, since a filter set on the command
line would otherwise go unseen. `--inat-sound N` picks another of the
observation's recordings. The same caveats are printed as warnings.

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
