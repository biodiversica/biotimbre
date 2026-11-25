# Libraries

*English · [Português (Brasil)](../pt-BR/libraries.md)*

## Library Location

By default, the library is stored in `./data`, relative to the current
directory. The location can be set in three ways:

```bash
# 1. For a single invocation
biotimbre --data-dir /media/drive/biotimbre list

# 2. With a configuration file in the working directory (or --config /path/to/it)
echo '[library]
data_dir = "/media/drive/biotimbre"' > biotimbre.toml

# 3. With an environment variable pointing to a configuration file
export BIOTIMBRE_CONFIG=/media/drive/biotimbre/biotimbre.toml
```

`--data-dir` takes precedence over the configuration file. Relative paths in a
configuration file are resolved against the directory containing that file,
not the working directory, so a configuration file and its library can be
moved together.

**Opening an existing library.** No separate step is required: every command
opens the database and creates anything that is missing.

```bash
biotimbre --data-dir /media/drive/biotimbre list
```

`init` may be run safely on a populated library; it only creates what is absent
and downloads the models. Stored audio paths are relative to the data
directory, so **the whole folder can be moved or renamed without breaking any
reference.**

## Data Layout

```
data/
├── library.db                        # metadata and embedding vectors
└── audio/<biotimbre_id>/<item_id>.wav
```

A library is backed up by copying its folder. Vectors are stored in the
database so that metadata filters can narrow the candidate set before any
vector is scored; `export-npy` provides the vectors as files when needed for
other tools.
