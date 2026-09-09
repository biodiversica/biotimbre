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

import re
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
#: 4: items.sample_window_start/_end are nullable — the window locates a sample
#: inside a full recording, which not every sample has. Migrated in place by
#: :func:`_migrate_3_to_4`; nothing is lost, since only a constraint relaxes.
SCHEMA_VERSION = 4

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


def _items_table_ddl(name: str) -> str:
    """The current ``items`` definition from schema.sql, under another name.

    A migration that rebuilds the table must not restate its columns: two
    definitions of the same table drift apart. This lifts the one in
    schema.sql and only changes the name it is created under.

    Args:
        name: Name to create the table as.

    Returns:
        A ``CREATE TABLE`` statement.

    Raises:
        BiotimbreError: If schema.sql no longer holds a recognisable
            ``items`` definition.
    """
    match = re.search(
        r"CREATE TABLE IF NOT EXISTS items\s*\(.*?\n\);", _schema_sql(), re.S
    )
    if match is None:
        raise BiotimbreError(
            "schema.sql no longer contains an 'items' table definition; "
            "the 3 → 4 migration cannot rebuild it."
        )
    return re.sub(
        r"^CREATE TABLE IF NOT EXISTS items\s*\(",
        f"CREATE TABLE {name} (",
        match.group(0),
        count=1,
    )


def _migrate_3_to_4(conn: sqlite3.Connection) -> None:
    """Drop the NOT NULL on items.sample_window_start/_end, keeping every row.

    SQLite cannot relax a column constraint in place, so this is the standard
    rebuild: copy into a table with the new definition, drop the old one, take
    its name. Every existing row already has a window, so nothing changes but
    what is allowed from here on.

    Args:
        conn: An open connection to the version-3 database.
    """
    # Foreign keys off for the swap: item_taxa and embeddings point at
    # items, and dropping it with them enforced would cascade the rows away.
    # It has to be off *outside* a transaction to take effect.
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN")
        conn.execute(_items_table_ddl("items_migrating"))
        columns = ", ".join(
            row["name"]
            for row in conn.execute("PRAGMA table_info(items_migrating)")
        )
        conn.execute(
            f"INSERT INTO items_migrating ({columns}) SELECT {columns} FROM items"
        )
        conn.execute("DROP TABLE items")
        conn.execute("ALTER TABLE items_migrating RENAME TO items")
        # The old table's indexes went with it; schema.sql recreates them.
        conn.executescript(_schema_sql())
        broken = conn.execute("PRAGMA foreign_key_check").fetchall()
        if broken:
            raise BiotimbreError(
                f"Migration to schema version 4 would leave {len(broken)} "
                f"orphaned row(s) behind; the database was left unchanged."
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


#: How to get from one schema version to the next, keyed by the version being
#: left behind. A version with no entry here has no migration and fails to open.
_MIGRATIONS = {3: _migrate_3_to_4}


def _migrate(conn: sqlite3.Connection, found: int, path: Path) -> None:
    """Bring a database up to :data:`SCHEMA_VERSION`, one version at a time.

    Args:
        conn: An open connection to the database.
        found: The version the database is at.
        path: Where the database lives, for error messages.

    Raises:
        BiotimbreError: If some step on the way has no migration.
    """
    while found < SCHEMA_VERSION:
        step = _MIGRATIONS.get(found)
        if step is None:
            raise BiotimbreError(
                f"Database {path} has schema version {found}, but this "
                f"software requires version {SCHEMA_VERSION}. No "
                f"automatic migration is provided from version {found} "
                f"(see SCHEMA_VERSION's docstring for what changed between "
                f"versions). Move this data_dir aside and run `biotimbre "
                f"init` to start a fresh library, or reinstall an older "
                f"biotimbre-library version to keep using this one."
            )
        step(conn)
        found += 1
        conn.execute(
            "UPDATE schema_meta SET value = ? WHERE key = ?",
            (str(found), _SCHEMA_KEY),
        )
        conn.commit()


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
            with no migration path to the current one.
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
                _migrate(conn, found, path)

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
