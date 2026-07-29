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
from importlib import resources
from pathlib import Path

from biotimbre.domain.errors import BiotimbreError

#: Bumped whenever schema.sql changes in a way that needs a migration.
SCHEMA_VERSION = 1

_SCHEMA_KEY = "schema_version"


def _schema_sql() -> str:
    return (
        resources.files("biotimbre.db").joinpath("schema.sql").read_text("utf-8")
    )


def connect(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with the library's required pragmas applied.

    Args:
        db_path: Path to the SQLite file. Its parent directory must exist.
        read_only: Open via a ``file:`` URI in read-only mode. Useful for a
            public search deployment that must not be able to write.

    Returns:
        A connection with :class:`sqlite3.Row` row factory.

    Raises:
        BiotimbreError: If the database cannot be opened.
    """
    path = Path(db_path)
    try:
        if read_only:
            conn = sqlite3.connect(
                f"file:{path}?mode=ro", uri=True, timeout=30.0
            )
        else:
            conn = sqlite3.connect(str(path), timeout=30.0)
    except sqlite3.Error as exc:
        raise BiotimbreError(f"Could not open database {path}: {exc}") from exc

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def initialize(db_path: str | Path) -> sqlite3.Connection:
    """Create the schema if absent, then return an open connection.

    Idempotent: safe to call on an existing library.

    Args:
        db_path: Path to the SQLite file. Parent directories are created.

    Returns:
        An initialised connection.

    Raises:
        BiotimbreError: If the file holds a schema from a newer version of this
            software, which this code cannot safely read.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(path)
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
            # found < SCHEMA_VERSION would run migrations here. There are none
            # to run at version 1; the branch exists so the check above is not
            # mistaken for complete migration support.
        conn.commit()
    except BiotimbreError:
        conn.close()
        raise
    except sqlite3.Error as exc:
        conn.close()
        raise BiotimbreError(f"Could not initialise schema in {path}: {exc}") from exc
    return conn
