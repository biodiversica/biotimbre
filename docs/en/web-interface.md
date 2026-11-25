# Web Interface

*English · [Português (Brasil)](../pt-BR/web-interface.md)*

```bash
biotimbre gui        # http://127.0.0.1:8080
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

## Library Page

The Library page is divided into three collapsible sections.

- **Library items** — library totals, a map of every item and the item list.
  The totals are followed by the speeds present in the library and by its
  taxa, each with its number of items; when there are many taxa, the most
  frequent are shown first and the rest can be expanded. Selecting an item
  opens a dialog with its full metadata, its spectrogram or
  waveform, a player, and buttons to export, edit or delete it, or to **Find
  similar** items (see the Search page below). Editing warns
  before saving a change of `speed`, since that requires recomputing the
  embeddings. **Export CSV** downloads the metadata of every item as a table
  (see [Exporting Data](exporting.md)). Ticking items in the list (the box
  in the header ticks a whole page) shows **Export N as bundle**, which
  downloads them as one shareable `.btz`, with or without their embeddings.
- **Add an item** — uploads a recording and its metadata. The spectrogram and
  sample rate are shown before anything is stored, with each model's frequency
  limit indicated. The recording can be trimmed, filtered and normalised before
  it is stored (see [Audio Preparation](audio-preparation.md)). Validation errors
  are shown on the corresponding field.
- **Import items** — loads a `.btz` bundle from another library, or adds
  every item of another library on the same computer, with a preview of the
  outcome that can be adjusted before anything is added (see
  [Sharing Items Between Libraries](sharing.md)).

The recording location is entered in a single field as
`latitude, longitude` in decimal degrees. A comma, a semicolon or spaces are
accepted as separators, so coordinates copied from mapping services, GPS
applications or field notes can be pasted directly. **Pick on map** sets the
location by clicking on a map instead.

## Search Page

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
- **Options** set the maximum number of results, the similarity metric, the
  model to rank by (Perch v2 by default), and a minimum score on that model's
  scale. Each model's scores have a baseline and spread of their own, so a
  0.6 from one is not a 0.6 from another, and they are never combined. The
  query's **Query
  speed** starts empty, which searches every speed in the library (see
  [Ultrasonic Recordings](audio-preparation.md#ultrasonic-recordings-and-the-speed-setting)); a
  value searches only the items stored at it.

Each result shows the ranking model's score, the other models' scores with
the rank each gave the item (a high rank from both means the models agree),
its own player, and a spectrogram on which the window that produced the score
is marked. The query
remains playable at the top of the page for comparison by ear. Each result
also provides:

- **Details** — a read-only dialog with the item's full metadata, as on the
  Library page but without the edit and delete options.
- **Export** — the item's audio, its embeddings or a shareable bundle.
- **Find similar** — searches the library with that item's recording.
- A shortened biotimbre ID which, when clicked, copies the full UUID to the
  clipboard, for use in the biotimbre ID filter or in another library.

**Export CSV**, next to the result count, downloads the displayed results as a
table. Ticking results (or **Select all**) shows **Export N as bundle**, which
downloads the ticked ones as one shareable `.btz`; a new search starts with
nothing ticked. **Add to library…** opens the Library page's add form with the query
recording and all of its settings (and, for a recording fetched from
iNaturalist, the observation's metadata) so that a new recording can be added
to the library. Nothing is stored until the form is saved. This option is not
available in search-only mode.

**Find similar**, on a result or in an item's dialog, makes a library item the
query: its stored recording is staged as an upload would be, ready to be
trimmed or filtered, and the query area names the item. The item is left out
of its own results, where it would come first with a score of 1, unless
**Include the item itself** is ticked. The query speed is the item's own, so
an item stored slowed down is compared with the others stored at that speed.
From the command line, the same is `biotimbre search --item ITEM_ID`.

## Search-Only Mode

```bash
biotimbre --data-dir ./library gui --search-only
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
biotimbre --data-dir ./library gui --search-only --host 0.0.0.0
```

The equivalent configuration, for a machine that should only serve searches:

```toml
[ui]
search_only = true
```

## Selecting a Library at Startup

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

To change library later, the **⇄** button beside the library's name in the
header closes it, after asking, and returns to this page, with the library just
closed offered first to open again. Anything not saved on the pages — a
recording being added, a query — is dropped; the library itself is not
changed. Other browser tabs on the same server go to this page too. A
search-only run has no such button: it serves one library and no other.

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
