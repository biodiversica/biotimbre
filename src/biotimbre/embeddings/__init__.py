"""Audio preprocessing and ONNX embedding extraction."""

from biotimbre.embeddings.audio import (
    MAX_QUERY_DURATION_S,
    AudioProperties,
    file_sha256,
    load_mono,
    prepare_windows,
    read_properties,
    resolve_query_segment,
)
from biotimbre.embeddings.extractor import Extractor, ExtractorPool, WindowEmbeddings
from biotimbre.embeddings.onnx_backend import OnnxEmbedder

__all__ = [
    "MAX_QUERY_DURATION_S",
    "AudioProperties",
    "Extractor",
    "ExtractorPool",
    "OnnxEmbedder",
    "WindowEmbeddings",
    "file_sha256",
    "load_mono",
    "prepare_windows",
    "read_properties",
    "resolve_query_segment",
]
