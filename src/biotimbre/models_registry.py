"""Registry of embedding models.

A "model" here is an *(ONNX graph, preprocessing)* pair: two entries may share a
graph but differ in sample rate, window length or overlap, and they are then
distinct models as far as the library is concerned, because their embeddings are
not comparable. ``model_id`` is therefore the identity of a preprocessing recipe,
not just of a file.

``embedding_dim`` is deliberately absent from :class:`ModelSpec`: it is read from
the ONNX graph at registration time and persisted in the ``models`` table. See
PLAN.md §1.

Two models are built in (:data:`MODEL_SPECS`). Others are defined by the user
under ``[models.custom.<id>]`` in the config file, and come either from a
Hugging Face repository or from a plain ``http(s)`` URL — never a local path,
since a library has to be able to fetch every model it was built with on
whichever machine it is opened. Once registered, a library remembers such a
model in its ``models`` table (:func:`spec_from_row`), so it keeps working
without the config file that first defined it.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: Fractional window overlap used at ingest. A vocalization straddling a window
#: boundary is split and scores poorly in both halves at 0.0; 0.5 costs little
#: (a 90 s Perch sample is ~220 KB of vectors) and materially improves recall.
DEFAULT_WINDOW_OVERLAP = 0.5

#: Highest window overlap accepted. At 1.0 the hop would be zero.
MAX_WINDOW_OVERLAP = 0.95

#: Prefix of a Hugging Face source, as ``models.source`` stores it.
HF_PREFIX = "hf://"

#: How a URL model's file is pinned: its SHA-256, in ``models.revision``.
SHA256_PREFIX = "sha256:"

#: Model IDs name files (exports, bundle members) and appear on the command
#: line, so they are kept to characters that are safe in all of those.
MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

#: Bytes read per chunk when downloading or hashing a model file.
_CHUNK = 1 << 20


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    """Definition of an embedding model and its required preprocessing.

    Exactly one of ``repo_id`` and ``url`` says where the ONNX file comes from.

    Attributes:
        model_id: Stable identifier, used as the ``models`` table primary key
            and in ``embeddings.model_id``.
        repo_id: Hugging Face repository holding the ONNX file.
        url: ``http(s)`` address of the ONNX file itself.
        filename: For a repository, the ONNX file's path within it. For a URL,
            the name the file is cached under; defaults to the URL's last path
            segment.
        sample_rate: Rate (Hz) the graph expects its input audio at.
        window_duration_s: Length of one inference window, in seconds.
        window_overlap: Fractional overlap between consecutive windows,
            [0, :data:`MAX_WINDOW_OVERLAP`].
        revision: For a repository, the git revision to pin; ``None`` tracks the
            default branch. For a URL, ``"sha256:<hex>"`` of the file; ``None``
            until the first download records it.
        description: Human-readable label for UI and CLI output.

    Raises:
        ValueError: If the definition is incomplete or inconsistent.
    """

    model_id: str
    repo_id: str | None = None
    url: str | None = None
    filename: str = ""
    sample_rate: int
    window_duration_s: float
    window_overlap: float = DEFAULT_WINDOW_OVERLAP
    revision: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        name = f"model {self.model_id!r}"
        if not MODEL_ID_PATTERN.match(self.model_id or ""):
            raise ValueError(
                f"Invalid model id {self.model_id!r}: use 1-64 lowercase "
                f"letters, digits, '_', '.' or '-', starting with a letter or "
                f"digit."
            )
        if bool(self.repo_id) == bool(self.url):
            raise ValueError(f"{name} needs exactly one of repo_id or url.")
        if self.url:
            parsed = urllib.parse.urlparse(self.url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(
                    f"{name}: url must be an http:// or https:// address, "
                    f"got {self.url!r}."
                )
            if not self.filename:
                object.__setattr__(
                    self,
                    "filename",
                    urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]),
                )
            if not self.filename or "/" in self.filename or "\\" in self.filename:
                raise ValueError(
                    f"{name}: cannot name the file from the url; give filename."
                )
            if self.revision is not None and not _is_sha256_revision(self.revision):
                raise ValueError(
                    f"{name}: a url model is pinned by sha256, not revision."
                )
        if not self.filename:
            raise ValueError(f"{name} needs filename: the ONNX file in the repository.")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise ValueError(f"{name}: sample_rate must be a whole number of Hz.")
        if self.sample_rate <= 0:
            raise ValueError(f"{name}: sample_rate must be positive.")
        if not self.window_duration_s > 0:
            raise ValueError(f"{name}: window_duration_s must be positive.")
        if not 0.0 <= self.window_overlap <= MAX_WINDOW_OVERLAP:
            raise ValueError(
                f"{name}: window_overlap must be between 0 and "
                f"{MAX_WINDOW_OVERLAP}."
            )

    @property
    def source(self) -> str:
        """Provenance string persisted in ``models.source``."""
        return self.url if self.url else f"{HF_PREFIX}{self.repo_id}"

    @property
    def recipe(self) -> tuple:
        """Every field that decides what vectors this model produces.

        Two specs with the same recipe produce comparable vectors; two with
        different ones must not share a ``model_id`` in a library.
        """
        return (
            self.source,
            self.filename,
            self.revision,
            int(self.sample_rate),
            float(self.window_duration_s),
            float(self.window_overlap),
        )

    @property
    def window_samples(self) -> int:
        """Window length in samples at :attr:`sample_rate`."""
        return int(self.window_duration_s * self.sample_rate)

    @property
    def stride_samples(self) -> int:
        """Hop between consecutive window starts, in samples."""
        return max(1, int(self.window_samples * (1.0 - self.window_overlap)))

    @property
    def nyquist_hz(self) -> float:
        """Highest frequency surviving the resample to :attr:`sample_rate`.

        Content above this is removed by the anti-alias filter during
        resampling, which is what ``items.speed`` exists to work around
        (PLAN.md D2b).
        """
        return self.sample_rate / 2.0

    def n_windows(self, n_samples: int) -> int:
        """Number of windows produced for an input of ``n_samples``.

        Always at least one window: a signal shorter than the window is
        zero-padded rather than dropped. A trailing partial hop is not padded
        into an extra window, so a 5 s input with a 5 s window yields exactly
        one window regardless of overlap.

        Args:
            n_samples: Input length in samples, at :attr:`sample_rate`.

        Returns:
            Window count, >= 1.
        """
        excess = max(0, n_samples - self.window_samples)
        return 1 + -(-excess // self.stride_samples)


def _is_sha256_revision(value: str) -> bool:
    return value.startswith(SHA256_PREFIX) and bool(
        _SHA256_HEX.match(value[len(SHA256_PREFIX):])
    )


def sha256_revision(digest: str) -> str:
    """The ``revision`` that pins a URL model to a file digest.

    Args:
        digest: The file's SHA-256, as 64 hex characters, any case.

    Returns:
        ``"sha256:<lowercase hex>"``.

    Raises:
        ValueError: If ``digest`` is not 64 hex characters.
    """
    text = str(digest).strip().lower()
    if not _SHA256_HEX.match(text):
        raise ValueError(f"sha256 must be 64 hexadecimal characters, got {digest!r}.")
    return SHA256_PREFIX + text


def spec_from_row(row: Mapping) -> ModelSpec:
    """Rebuild a model's spec from its ``models`` table row.

    This is what lets a library carry a user-defined model: everything needed
    to fetch the file again and cut audio for it is in the row.

    Args:
        row: A ``models`` row, as :meth:`Repository.get_model` returns it.

    Returns:
        The spec the row was registered from.

    Raises:
        ValueError: If the row's source is neither a Hugging Face repository
            nor a URL.
    """
    source = str(row["source"])
    location: dict[str, str] = {}
    if source.startswith(HF_PREFIX):
        location["repo_id"] = source[len(HF_PREFIX):]
    elif source.startswith(("http://", "https://")):
        location["url"] = source
    else:
        raise ValueError(
            f"Model {row['model_id']!r} is registered with source {source!r}, "
            f"which is neither a Hugging Face repository nor a URL."
        )
    return ModelSpec(
        model_id=str(row["model_id"]),
        **location,
        filename=str(row["filename"]),
        revision=row["revision"],
        sample_rate=int(row["sample_rate"]),
        window_duration_s=float(row["window_duration_s"]),
        window_overlap=float(row["window_overlap"]),
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "perch_v2": ModelSpec(
        model_id="perch_v2",
        repo_id="biodiversica/Perch-onnx-backbone",
        filename="perch_v2_backbone.onnx",
        sample_rate=32_000,
        window_duration_s=5.0,
        description="Perch v2 backbone (embeddings only)",
    ),
    "birdnet_v2.4": ModelSpec(
        model_id="birdnet_v2.4",
        repo_id="biodiversica/BirdNET-onnx-backbone",
        filename="model_backbone.onnx",
        sample_rate=48_000,
        window_duration_s=3.0,
        description="BirdNET 2.4 backbone (embeddings only)",
    ),
}


def get_spec(
    model_id: str, extra: Mapping[str, ModelSpec] | None = None
) -> ModelSpec:
    """Look up a model spec by ID.

    Args:
        model_id: Key of :data:`MODEL_SPECS` or of ``extra``.
        extra: User-defined models to look in besides the built-in ones,
            typically :attr:`Settings.custom_models`. They take precedence.

    Returns:
        The matching :class:`ModelSpec`.

    Raises:
        KeyError: If no such model is known, listing the valid IDs.
    """
    if extra and model_id in extra:
        return extra[model_id]
    try:
        return MODEL_SPECS[model_id]
    except KeyError:
        known = ", ".join(sorted({*MODEL_SPECS, *(extra or {})})) or "(none)"
        raise KeyError(
            f"Unknown model id {model_id!r}. Known models: {known}. Define "
            f"others under [models.custom.<id>] in the config file."
        ) from None


def resolve_model_file(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    """Return the local path to the ONNX file, downloading it only if absent.

    A URL model is handled by :func:`resolve_url_file`; what follows describes
    a Hugging Face one, and the URL path follows the same rules.

    The cache is consulted **offline first**. Only a genuine cache miss reaches
    the network. That matters for three reasons:

    *Correctness.* Re-resolving against the Hub could pick up a new revision of a
    model whose ``model_id`` is unchanged, and embeddings from a different model
    build are not comparable with the ones already stored. Pinning to whatever was
    first downloaded keeps the library internally consistent; use a new
    ``model_id`` (or an explicit ``revision``) to adopt a new build deliberately.

    *Speed.* Revalidating a cached file cost 0.18–0.49 s per model per call, paid
    on every single search.

    *Offline use.* This is a local library; it must work with no network, which
    cache-first gives for free rather than needing ``HF_HUB_OFFLINE``.

    A side effect is that huggingface_hub's "unauthenticated requests" warning no
    longer appears on every search — it can only surface on a real download.

    Args:
        spec: The model to resolve.
        cache_dir: Override the Hugging Face cache location. ``None`` uses the
            HF default.

    Returns:
        Path to the cached ONNX file.

    Raises:
        RuntimeError: If the file is neither cached nor downloadable.
    """
    if spec.url:
        return resolve_url_file(spec, cache_dir)[0]

    from huggingface_hub import hf_hub_download

    arguments = {
        "repo_id": spec.repo_id,
        "filename": spec.filename,
        "revision": spec.revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }

    try:
        return Path(hf_hub_download(**arguments, local_files_only=True))
    except Exception:
        # Not in the cache. Fall through to a download; any failure there is what
        # gets reported, since it carries the useful diagnostic.
        pass

    try:
        return Path(hf_hub_download(**arguments))
    except Exception as exc:  # network, auth, missing file — unrecoverable here
        raise RuntimeError(
            f"Could not obtain model {spec.model_id!r} "
            f"({spec.filename} from {spec.repo_id}): {exc}"
        ) from exc


# ── URL models ────────────────────────────────────────────────────────────────


def default_url_cache() -> Path:
    """Where URL models are cached when no ``model_cache_dir`` is configured.

    ``$XDG_CACHE_HOME/biotimbre/models``, which is ``~/.cache/biotimbre/models``
    unless that variable says otherwise — beside Hugging Face's own default.
    """
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "biotimbre" / "models"


def url_cache_path(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    """Where a URL model's file is cached.

    One directory per URL, named by a digest of it, so two models whose files
    share a name cannot overwrite each other.

    Args:
        spec: A URL model.
        cache_dir: The configured ``model_cache_dir``, or ``None``.
    """
    root = (
        Path(cache_dir) / "biotimbre-urls" if cache_dir is not None
        else default_url_cache()
    )
    key = hashlib.sha256(str(spec.url).encode("utf-8")).hexdigest()[:16]
    return root / key / spec.filename


def resolve_url_file(
    spec: ModelSpec, cache_dir: Path | None = None
) -> tuple[Path, str]:
    """Return a URL model's cached file and its digest, downloading if needed.

    As with the Hub, the cache is used first and the network only on a miss. The
    file's SHA-256 is computed as it downloads and kept beside it, so the
    library can record which file its vectors came from. A spec pinned to a
    digest accepts only that file: a cached copy that differs is fetched again,
    and a download that differs is discarded with an error, since the vectors
    already stored were made by a different file.

    Args:
        spec: A URL model.
        cache_dir: The configured ``model_cache_dir``, or ``None``.

    Returns:
        The local path and ``"sha256:<hex>"`` of its contents.

    Raises:
        RuntimeError: If the file is neither cached nor downloadable, or does
            not match its pinned digest.
    """
    path = url_cache_path(spec, cache_dir)
    digest_file = path.with_name(path.name + ".sha256")
    wanted = spec.revision

    if path.is_file() and digest_file.is_file():
        cached = digest_file.read_text(encoding="utf-8").strip()
        if wanted is None or cached == wanted:
            return path, cached

    try:
        digest = _download(str(spec.url), path)
    except Exception as exc:  # network, HTTP status, disk
        raise RuntimeError(
            f"Could not obtain model {spec.model_id!r} from {spec.url}: {exc}"
        ) from exc
    if wanted is not None and digest != wanted:
        path.unlink(missing_ok=True)
        digest_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"Model {spec.model_id!r} downloaded from {spec.url} has "
            f"{digest}, but this library's vectors were made by a file with "
            f"{wanted}. The file at that address has changed: define the new "
            f"file under a new model id."
        )
    digest_file.write_text(digest + "\n", encoding="utf-8")
    return path, digest


def _download(url: str, destination: Path) -> str:
    """Stream ``url`` to ``destination``, returning ``"sha256:<hex>"``.

    Written beside the destination first and moved into place only when
    complete, so an interrupted download never looks like a cached model.
    """
    import urllib.request

    from biotimbre import __version__

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": f"biotimbre/{__version__}"}
    )
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open(
            "wb"
        ) as out:
            while chunk := response.read(_CHUNK):
                hasher.update(chunk)
                out.write(chunk)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return SHA256_PREFIX + hasher.hexdigest()
