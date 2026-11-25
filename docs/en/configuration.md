# Configuration

*English · [Português (Brasil)](../pt-BR/configuration.md)*

The configuration file `biotimbre.toml` is optional, and every setting has a
default value.

**Windows paths.** In TOML, a backslash inside double quotes starts an escape
sequence, so `"C:\Users\me\biotimbre"` is rejected as invalid. Use single
quotes, which take the text literally (`'C:\Users\me\biotimbre'`), or forward
slashes (`"C:/Users/me/biotimbre"`).

```toml
[library]
data_dir = "data"
# db_path   = "/fast/library.db"      # may be stored separately from the audio
# audio_dir = "/bulk/audio"           # stored as an absolute path if outside data_dir
# model_cache_dir = "/path/to/cache"  # default: the Hugging Face cache

[models]
active = ["perch_v2", "birdnet_v2.4"]   # default: every known model
inference_batch_size = 1   # windows per model call; 1 is fastest here
inference_threads = 0      # 0 lets onnxruntime decide
# [models.custom.<id>] tables add models: see Adding a Model

[search]
top_k = 10
metric = "cosine"          # cosine | dot | euclidean
rank_by = "perch_v2"       # the model whose scores rank the results
batch_size = 8192
query_cache_entries = 16
max_query_duration_s = 15

[ui]
language = "en"            # en | pt_BR
search_only = false        # serve the Search page only, with a read-only library
dominant_frequency = true  # dashed, labelled line at the loudest frequency

[map]
# The location map on both pages. This is the only part of the interface that
# accesses the network merely by being displayed: Leaflet is served by this
# process, but the basemap tiles come from the URL below. An offline library
# should set enabled = false. Coordinates are still filtered on and displayed
# as text in either case.
enabled = true
# tile_url    = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
# attribution = "&copy; OpenStreetMap contributors"   # must credit the source
# max_zoom    = 19        # must match the tile server's deepest zoom level

[ui.theme]
# The light palette. Every key is optional; the values shown are the defaults.
# Colours are CSS values (hex, rgb(a) or named). primary..warning set Quasar's
# palette; the remaining keys style the header directly.
primary = "#20460a"
secondary = "#fcfee5"
accent = "#e9b44c"
dark = "#20460a"
positive = "#118b50"
negative = "#c62828"
info = "#2f6f9f"
warning = "#e9b44c"
button_text = "#fcfee5"        # label colour on filled primary buttons
page_bg = "#f4f6ef"            # page background below the header
text = "#1a2712"               # default label colour
header_bg = "#1b2516"
header_text = "#fcfee5"
header_inactive_text = "rgba(255, 255, 255, 0.65)"
header_active_bg = "rgba(255, 255, 255, 0.2)"

# The following keys style components whose colours Quasar and Tailwind
# otherwise fix regardless of ui.colors():
card = "#ffffff"               # card backgrounds (item dialog, forms)
link = "#118b50"               # hyperlinks (e.g. "open full recording")
muted = "#5b6b52"              # secondary text (hints, timestamps, IDs)
notice = "#92400e"             # inline warnings (speed shift, query limit, etc.)
surface = "#eef1e6"            # statistics tile backgrounds on the Library page
chip = "#dde4d1"               # small ID badge backgrounds
marker_border = "rgba(252, 254, 229, 0.8)"  # matched-window outline on a spectrogram
marker_fill = "rgba(252, 254, 229, 0.25)"

# Colours of the spectrogram images themselves. Evenly spaced #rrggbb stops
# from quiet (first) to loud (last); at least two are required. The ramp
# should increase monotonically in brightness so that intensity remains
# readable for colour-blind viewers. The default ramp runs from dark blue
# through magenta and orange to pale yellow.
spectrogram_colors = ["#0c0826", "#400c60", "#941a60", "#de5234", "#fca83c", "#fff4c8"]

[ui.dark_theme]
# The dark palette, with the same keys and meanings. Omitted keys fall back to
# the built-in dark defaults, never to the light values above, so overriding
# one colour does not bring the light palette into the dark theme.
# primary = "#6f9a7a"
# page_bg = "#0e100f"
# text = "#d6cdb8"
# card = "#161a18"
# spectrogram_colors is not read here: the images are rendered and cached on
# the server, where the viewer's theme is not known, so the [ui.theme] ramp
# is used for both.
```
