"""Database connection setup and schema initialisation.

Two pragmas matter and are set on every connection:

``foreign_keys=ON``
    SQLite defaults this *off*, per-connection. Without it the
    ``ON DELETE CASCADE`` on ``item_taxa`` and ``embeddings`` silently does
    nothing and deleting an item orphans its vectors.

``journal_mode=WAL``
    Lets a reader (a search) proceed while a writer (an ingest) commits.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from biotimbre.domain.errors import BiotimbreError, ValidationError

#: Bumped whenever schema.sql changes in a way that needs a migration.
#: 2: biotimbre_id switched from a per-library 4-hex-char code to a UUID
#: (so it means the same thing across libraries). No migration is provided —
#: a version-1 database fails to open with a clear message instead.
#: 3: items.noise_reduction removed; items.microphone, recorded_by, filter_type,
#: filter_order, temperature, humidity added. No migration is provided.
SCHEMA_VERSION = 3

_SCHEMA_KEY = "schema_version"

#: schema_meta keys for the library's identity. The UUID is generated once, on
#: first initialize(), and never changes — it is what lets two copies of a
#: shared library (e.g. after a directory copy) be recognised as the same
#: library. The name has no default: it is unset until a caller explicitly
#: sets one, since "must be set" is only meaningful if nothing invents one.
_UUID_KEY = "library_uuid"
_NAME_KEY = "library_name"


def _schema_sql() -> str:
    return (
        resources.files("biotimbre.db").joinpath("schema.sql").read_text("utf-8")
    )


def connect(
    db_path: str | Path,
    *,
    read_only: bool = False,
    allow_threads: bool = False,
) -> sqlite3.Connection:
    """Open a connection with the library's required pragmas applied.

    Args:
        db_path: Path to the SQLite file. Its parent directory must exist.
        read_only: Open via a ``file:`` URI in read-only mode. Useful for a
            public search deployment that must not be able to write.
        allow_threads: Permit use from threads other than the one that opened it.
            Required by the web UI, where a request handler and a background
            inference task do not share a thread; sqlite3 otherwise raises
            "SQLite objects created in a thread can only be used in that same
            thread". **The caller must serialise access** — one connection used
            concurrently is still unsafe, so the UI holds a lock around service
            calls.

    Returns:
        A connection with :class:`sqlite3.Row` row factory.

    Raises:
        BiotimbreError: If the database cannot be opened.
    """
    path = Path(db_path)
    try:
        if read_only:
            conn = sqlite3.connect(
                f"file:{path}?mode=ro",
                uri=True,
                timeout=30.0,
                check_same_thread=not allow_threads,
            )
        else:
            conn = sqlite3.connect(
                str(path), timeout=30.0, check_same_thread=not allow_threads
            )
    except sqlite3.Error as exc:
        raise BiotimbreError(f"Could not open database {path}: {exc}") from exc

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def initialize(
    db_path: str | Path, *, allow_threads: bool = False
) -> sqlite3.Connection:
    """Create the schema if absent, then return an open connection.

    Idempotent: safe to call on an existing library.

    Args:
        db_path: Path to the SQLite file. Parent directories are created.
        allow_threads: See :func:`connect`.

    Returns:
        An initialised connection.

    Raises:
        BiotimbreError: If the file holds a schema from a newer version of this
            software, which this code cannot safely read, or an older version
            this code no longer supports (no migration path is provided).
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(path, allow_threads=allow_threads)
    try:
        conn.executescript(_schema_sql())
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = ?", (_SCHEMA_KEY,)
        ).fetchone()

        if row is None:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
                (_SCHEMA_KEY, str(SCHEMA_VERSION)),
            )
        else:
            found = int(row["value"])
            if found > SCHEMA_VERSION:
                raise BiotimbreError(
                    f"Database {path} has schema version {found}, but this "
                    f"software understands at most {SCHEMA_VERSION}. "
                    f"Upgrade biotimbre-library."
                )
            elif found < SCHEMA_VERSION:
                raise BiotimbreError(
                    f"Database {path} has schema version {found}, but this "
                    f"software requires version {SCHEMA_VERSION}. No "
                    f"automatic migration is provided (see SCHEMA_VERSION's "
                    f"docstring for what changed between versions). Move "
                    f"this data_dir aside and run `biotimbre init` to start "
                    f"a fresh library, or reinstall an older biotimbre-library "
                    f"version to keep using this one."
                )

        has_uuid = conn.execute(
            "SELECT 1 FROM schema_meta WHERE key = ?", (_UUID_KEY,)
        ).fetchone()
        if has_uuid is None:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
                (_UUID_KEY, str(uuid.uuid4())),
            )
        conn.commit()
    except BiotimbreError:
        conn.close()
        raise
    except sqlite3.Error as exc:
        conn.close()
        raise BiotimbreError(f"Could not initialise schema in {path}: {exc}") from exc
    return conn


@dataclass(frozen=True)
class LibraryIdentity:
    """A library's stable identity, distinct from where it happens to sit.

    Attributes:
        uuid: Generated once by :func:`initialize` and never changed —
            stable across renames, moves, and directory copies, so a shared
            library keeps one identity no matter how many copies exist.
        name: Human-readable name. ``None`` until :func:`set_library_name`
            is called — there is no invented default (e.g. the folder name),
            since a name that nobody chose is not a useful identity.
    """

    uuid: str
    name: str | None


def get_library_identity(conn: sqlite3.Connection) -> LibraryIdentity:
    """Read the library's UUID and name.

    Args:
        conn: An initialised connection (see :func:`initialize`).

    Returns:
        The current :class:`LibraryIdentity`.
    """
    rows = conn.execute(
        "SELECT key, value FROM schema_meta WHERE key IN (?, ?)",
        (_UUID_KEY, _NAME_KEY),
    ).fetchall()
    values = {row["key"]: row["value"] for row in rows}
    return LibraryIdentity(uuid=values[_UUID_KEY], name=values.get(_NAME_KEY))


def set_library_name(conn: sqlite3.Connection, name: str) -> None:
    """Set (or rename) the library's name.

    Args:
        conn: An initialised connection.
        name: The new name. Must be non-empty.

    Raises:
        ValidationError: If ``name`` is empty or all whitespace.
    """
    name = name.strip()
    if not name:
        raise ValidationError("Library name must not be empty.", "name")
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (_NAME_KEY, name),
    )
    conn.commit()
