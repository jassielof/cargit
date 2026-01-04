"""Database storage for cargit metadata.

Uses SQLite for persistent storage of installed binary information,
build statistics, and cache state.

Performance optimizations:
- Thread-local connection pooling
- WAL mode for better concurrency
- Prepared statements via parameterized queries
- Batch operations where possible
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Generator

from .core import CONFIG_DIR, OLD_CARGIT_DIR

DATABASE_FILE = CONFIG_DIR / "data.sqlite3"
OLD_DATABASE_FILE = OLD_CARGIT_DIR / "cli.db"

# Thread-local storage for database connections
_local = threading.local()
_initialized = False
_init_lock = threading.Lock()


def _init_database() -> None:
    """Initialize database (run migrations, etc.) - only once per process."""
    global _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        # Handle migration from old database location
        if OLD_DATABASE_FILE.exists() and not DATABASE_FILE.exists():
            import shutil
            from rich import print as rprint

            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            rprint(f"[blue]Migrating database to {DATABASE_FILE}...[/blue]")
            shutil.copy2(OLD_DATABASE_FILE, DATABASE_FILE)
            rprint("[green]Database migration completed![/green]")

        # Ensure directory exists
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Create tables and run migrations with a temporary connection
        conn = sqlite3.connect(DATABASE_FILE)
        try:
            conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster, still safe
            _ensure_table(conn)
            _run_migrations(conn)
        finally:
            conn.close()

        _initialized = True


def _get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    _init_database()

    conn = getattr(_local, 'connection', None)
    if conn is None:
        conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.connection = conn

    return conn


@contextmanager
def _transaction() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database transactions with automatic commit/rollback."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Ensure the installed binaries table exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS installed_binaries (
            alias TEXT PRIMARY KEY,
            repo_url TEXT NOT NULL,
            branch TEXT NOT NULL,
            commit_hash TEXT NOT NULL,
            install_dir TEXT NOT NULL,
            bin_path TEXT NOT NULL,
            crate TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    # Create index for repo_url lookups
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_repo_url ON installed_binaries(repo_url)"
    )
    conn.commit()


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Run database schema migrations."""
    cursor = conn.cursor()

    # Check if new columns exist
    cursor.execute("PRAGMA table_info(installed_binaries)")
    columns = {row[1] for row in cursor.fetchall()}

    migrations = [
        ("artifacts_cleaned", "ALTER TABLE installed_binaries ADD COLUMN artifacts_cleaned BOOLEAN DEFAULT 0"),
        ("repo_deleted", "ALTER TABLE installed_binaries ADD COLUMN repo_deleted BOOLEAN DEFAULT 0"),
        ("binary_type", "ALTER TABLE installed_binaries ADD COLUMN binary_type TEXT DEFAULT 'symlink'"),
        ("binary_copy_path", "ALTER TABLE installed_binaries ADD COLUMN binary_copy_path TEXT"),
        ("last_build_duration", "ALTER TABLE installed_binaries ADD COLUMN last_build_duration REAL"),
        ("total_build_count", "ALTER TABLE installed_binaries ADD COLUMN total_build_count INTEGER DEFAULT 0"),
        ("avg_build_duration", "ALTER TABLE installed_binaries ADD COLUMN avg_build_duration REAL"),
    ]

    for column_name, migration_sql in migrations:
        if column_name not in columns:
            cursor.execute(migration_sql)

    conn.commit()


def load_metadata() -> dict[str, Any]:
    """Load metadata from SQLite database."""
    conn = _get_connection()
    cursor = conn.execute("SELECT alias, repo_url, branch, commit_hash, install_dir, bin_path, crate FROM installed_binaries")

    installed = {
        row["alias"]: {
            "repo_url": row["repo_url"],
            "branch": row["branch"],
            "commit": row["commit_hash"],
            "install_dir": row["install_dir"],
            "bin_path": row["bin_path"],
            "alias": row["alias"],
            "crate": row["crate"],
        }
        for row in cursor.fetchall()
    }

    return {"installed": installed}


def save_binary_metadata(
    alias: str,
    repo_url: str,
    branch: str,
    commit: str,
    install_dir: str,
    bin_path: str,
    crate: str | None = None,
    binary_type: str = "copy",
    binary_copy_path: str | None = None,
) -> None:
    """Save or update a single binary's metadata."""
    with _transaction() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO installed_binaries
            (alias, repo_url, branch, commit_hash, install_dir, bin_path, crate, binary_type, binary_copy_path, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                alias,
                repo_url,
                branch,
                commit,
                install_dir,
                bin_path,
                crate,
                binary_type,
                binary_copy_path,
            ),
        )


def remove_binary_metadata(alias: str) -> None:
    """Remove a binary's metadata from database."""
    with _transaction() as conn:
        conn.execute("DELETE FROM installed_binaries WHERE alias = ?", (alias,))


def get_binary_metadata(alias: str) -> dict[str, Any] | None:
    """Get metadata for a specific binary."""
    conn = _get_connection()
    cursor = conn.execute(
        "SELECT * FROM installed_binaries WHERE alias = ?", (alias,)
    )
    row = cursor.fetchone()

    if row is None:
        return None

    return _row_to_dict(row)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a database row to a dictionary with proper type handling."""
    keys = row.keys()
    return {
        "repo_url": row["repo_url"],
        "branch": row["branch"],
        "commit": row["commit_hash"],
        "install_dir": row["install_dir"],
        "bin_path": row["bin_path"],
        "alias": row["alias"],
        "crate": row["crate"],
        "artifacts_cleaned": bool(row["artifacts_cleaned"]) if "artifacts_cleaned" in keys else False,
        "repo_deleted": bool(row["repo_deleted"]) if "repo_deleted" in keys else False,
        "binary_type": row["binary_type"] if "binary_type" in keys else "symlink",
        "binary_copy_path": row["binary_copy_path"] if "binary_copy_path" in keys else None,
        "last_build_duration": row["last_build_duration"] if "last_build_duration" in keys else None,
        "total_build_count": row["total_build_count"] if "total_build_count" in keys else 0,
        "avg_build_duration": row["avg_build_duration"] if "avg_build_duration" in keys else None,
    }


def mark_artifacts_cleaned(alias: str, binary_copy_path: str) -> None:
    """Mark that artifacts were cleaned for a binary."""
    with _transaction() as conn:
        conn.execute(
            """
            UPDATE installed_binaries
            SET artifacts_cleaned = 1, binary_type = 'copy', binary_copy_path = ?, updated_at = CURRENT_TIMESTAMP
            WHERE alias = ?
            """,
            (binary_copy_path, alias),
        )


def mark_repo_deleted(alias: str) -> None:
    """Mark that repo was deleted for a binary."""
    with _transaction() as conn:
        conn.execute(
            """
            UPDATE installed_binaries
            SET repo_deleted = 1, updated_at = CURRENT_TIMESTAMP
            WHERE alias = ?
            """,
            (alias,),
        )


def reset_cache_flags(alias: str) -> None:
    """Reset cache flags after rebuilding."""
    with _transaction() as conn:
        conn.execute(
            """
            UPDATE installed_binaries
            SET artifacts_cleaned = 0, repo_deleted = 0, binary_type = 'symlink',
                binary_copy_path = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE alias = ?
            """,
            (alias,),
        )


def get_all_repo_urls() -> list[str]:
    """Get all tracked repository URLs."""
    conn = _get_connection()
    cursor = conn.execute("SELECT DISTINCT repo_url FROM installed_binaries")
    return [row["repo_url"] for row in cursor.fetchall()]


def update_build_time(alias: str, duration_seconds: float) -> None:
    """Update build time statistics for a binary using running average."""
    with _transaction() as conn:
        # Use a single UPDATE with COALESCE to handle NULLs
        conn.execute(
            """
            UPDATE installed_binaries
            SET last_build_duration = ?,
                total_build_count = COALESCE(total_build_count, 0) + 1,
                avg_build_duration = COALESCE(avg_build_duration, 0) +
                    (? - COALESCE(avg_build_duration, 0)) / (COALESCE(total_build_count, 0) + 1),
                updated_at = CURRENT_TIMESTAMP
            WHERE alias = ?
            """,
            (duration_seconds, duration_seconds, alias),
        )


def get_build_stats(alias: str) -> dict[str, Any] | None:
    """Get build statistics for a binary."""
    conn = _get_connection()
    cursor = conn.execute(
        """
        SELECT last_build_duration, total_build_count, avg_build_duration
        FROM installed_binaries WHERE alias = ?
        """,
        (alias,)
    )
    row = cursor.fetchone()

    if row is None:
        return None

    return {
        "last_build_duration": row["last_build_duration"],
        "total_build_count": row["total_build_count"] or 0,
        "avg_build_duration": row["avg_build_duration"],
    }


def get_all_binaries_full() -> list[dict[str, Any]]:
    """Get all binaries with full metadata including build stats."""
    conn = _get_connection()
    cursor = conn.execute("SELECT * FROM installed_binaries ORDER BY alias")

    return [
        {
            "alias": row["alias"],
            "repo_url": row["repo_url"],
            "branch": row["branch"],
            "commit": row["commit_hash"],
            "install_dir": row["install_dir"],
            "bin_path": row["bin_path"],
            "crate": row["crate"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            **{k: row[k] if k in row.keys() else default
               for k, default in [
                   ("artifacts_cleaned", False),
                   ("repo_deleted", False),
                   ("binary_type", "symlink"),
                   ("last_build_duration", None),
                   ("total_build_count", 0),
                   ("avg_build_duration", None),
               ]}
        }
        for row in cursor.fetchall()
    ]


def get_binaries_by_repo(repo_url: str) -> list[dict[str, Any]]:
    """Get all binaries installed from a specific repository."""
    conn = _get_connection()
    cursor = conn.execute(
        "SELECT alias, branch, commit_hash, crate FROM installed_binaries WHERE repo_url = ?",
        (repo_url,)
    )

    return [
        {
            "alias": row["alias"],
            "branch": row["branch"],
            "commit": row["commit_hash"],
            "crate": row["crate"],
        }
        for row in cursor.fetchall()
    ]


def count_binaries() -> int:
    """Get total number of installed binaries."""
    conn = _get_connection()
    cursor = conn.execute("SELECT COUNT(*) as count FROM installed_binaries")
    return cursor.fetchone()["count"]


def get_status_summary() -> dict[str, Any]:
    """Get summary statistics for status command (single query)."""
    conn = _get_connection()
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN artifacts_cleaned = 1 THEN 1 ELSE 0 END) as artifacts_cleaned,
            SUM(CASE WHEN repo_deleted = 1 THEN 1 ELSE 0 END) as repos_deleted,
            COUNT(DISTINCT repo_url) as unique_repos,
            SUM(total_build_count) as total_builds,
            AVG(avg_build_duration) as overall_avg_build_time
        FROM installed_binaries
    """)
    row = cursor.fetchone()

    return {
        "total_binaries": row["total"] or 0,
        "artifacts_cleaned": row["artifacts_cleaned"] or 0,
        "repos_deleted": row["repos_deleted"] or 0,
        "unique_repos": row["unique_repos"] or 0,
        "total_builds": row["total_builds"] or 0,
        "overall_avg_build_time": row["overall_avg_build_time"],
    }


# Batch operations for better performance

def save_multiple_binaries(binaries: list[dict[str, Any]]) -> None:
    """Save multiple binaries in a single transaction."""
    with _transaction() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO installed_binaries
            (alias, repo_url, branch, commit_hash, install_dir, bin_path, crate, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                (
                    b["alias"],
                    b["repo_url"],
                    b["branch"],
                    b["commit"],
                    b["install_dir"],
                    b["bin_path"],
                    b.get("crate"),
                )
                for b in binaries
            ],
        )


def cleanup_connection() -> None:
    """Clean up thread-local connection (call at end of program)."""
    conn = getattr(_local, 'connection', None)
    if conn is not None:
        conn.close()
        _local.connection = None
