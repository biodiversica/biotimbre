# Adding a Model

*English · [Português (Brasil)](../pt-BR/adding-a-model.md)*

Besides the two built-in models (`perch_v2` and `birdnet_v2.4`), any ONNX
embedding model can be added in the configuration file, one table per model.
It is downloaded either from a Hugging Face repository or from an `http(s)`
URL; local files are not accepted, so that a library can always fetch the
models it was built with on another computer.

```toml
[models.custom.my_model]                 # the ID: lowercase, digits, _ . -
repo_id = "owner/repository"
filename = "model.onnx"                  # path of the file in the repository
revision = "3f2a9c1"                     # optional: pin a commit
sample_rate = 32000                      # Hz the model expects
window_duration_s = 5.0                  # seconds per window
window_overlap = 0.5                     # optional, 0 to 0.95; default 0.5
description = "My model"                 # optional

[models.custom.other_model]
url = "https://example.org/models/encoder.onnx"
sha256 = "9b74c9897bac770ffc029102a200c5de..."  # optional: pin the file
sample_rate = 16000
window_duration_s = 1.0
```

The model must take raw mono audio, as a `(windows, samples)` array at its
sample rate, and return one embedding per window. biotimbre does not compute spectrograms, so any
front end must be part of the graph.

Unless `[models] active` lists the models explicitly, every model defined this
way is used alongside the built-in ones. To compute its embeddings for the
items already in the library:

```bash
biotimbre backfill my_model
```

Existing items are read again from their stored audio; no item needs to be
imported again.

**The library remembers its models.** On first use, the model's definition is
recorded in `library.db`, together with the SHA-256 of a URL model's file. A
library opened later without that configuration file — by someone else, or
with `--search-only` on another computer — still downloads and uses the model.
A URL whose file has since changed is refused rather than mixed with the
stored vectors.

**A definition cannot change once vectors exist.** Vectors are comparable only
when they come from the same file with the same sample rate, window and
overlap. If any of these changes under an ID that already has vectors, the
model is refused with the differing field named. Define the changed model
under a new ID and backfill it.
