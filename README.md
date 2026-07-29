# Biotimbre Library

A local library of animal vocalization samples with embedding-based similarity
search, using Perch v2 and BirdNET 2.4 ONNX backbones via `onnxruntime`.

See [PLAN.md](PLAN.md) for the design and its rationale.

## Status

Under construction. Phases 1–5 (library core + CLI, no GUI) are in progress.

## Setup

```bash
uv venv --python 3.12
uv sync --extra dev
```

## Usage

```bash
uv run biotimbre --help
```

## Development

```bash
uv run pytest
```
