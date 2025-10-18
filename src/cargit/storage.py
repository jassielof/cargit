import sqlite3
from pathlib import Path
from typing import Any

from .core import CARGIT_DIR

DATABASE_FILE = CARGIT_DIR / "cargit.db"


def _get_connection() -> sqlite3.Connection:
    """Get database connection and ensure table exists"""
    conn = sqlite3.Connection(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    _ensure_table(conn)
    return conn


def _ensure_table(conn: sqlite3.Connection):
    """Ensure the installed binaries table exists"""
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
    conn.commit()


def load_metadata() -> dict[str, Any]:
    """Load metadata from SQLite database"""
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT * FROM installed_binaries")
        rows = cursor.fetchall()

        installed = {}
        for row in rows:
            installed[row["alias"]] = {
                "repo_url": row["repo_url"],
                "branch": row["branch"],
                "commit": row["commit_hash"],
                "install_dir": row["install_dir"],
                "bin_path": row["bin_path"],
                "alias": row["alias"],
                "crate": row["crate"],
            }

        return {"installed": installed}
    finally:
        conn.close()


def save_metadata(data: dict[str, Any]):
    """Save metadata to SQLite database"""
    conn = _get_connection()
    try:
        # Clear existing data and insert new data
        # Note: In practice, you might want to use UPSERT operations instead
        for alias, info in data.get("installed", {}).items():
            conn.execute(
                """
                INSERT OR REPLACE INTO installed_binaries
                (alias, repo_url, branch, commit_hash, install_dir, bin_path, crate, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    alias,
                    info["repo_url"],
                    info["branch"],
                    info["commit"],
                    info["install_dir"],
                    info["bin_path"],
                    info.get("crate"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def save_binary_metadata(
    alias: str,
    repo_url: str,
    branch: str,
    commit: str,
    install_dir: str,
    bin_path: str,
    crate: str | None = None,
):
    """Save or update a single binary's metadata"""
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO installed_binaries
            (alias, repo_url, branch, commit_hash, install_dir, bin_path, crate, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (alias, repo_url, branch, commit, install_dir, bin_path, crate),
        )
        conn.commit()
    finally:
        conn.close()


def remove_binary_metadata(alias: str):
    """Remove a binary's metadata from database"""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM installed_binaries WHERE alias = ?", (alias,))
        conn.commit()
    finally:
        conn.close()


def get_binary_metadata(alias: str) -> dict[str, Any] | None:
    """Get metadata for a specific binary"""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM installed_binaries WHERE alias = ?", (alias,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "repo_url": row["repo_url"],
            "branch": row["branch"],
            "commit": row["commit_hash"],
            "install_dir": row["install_dir"],
            "bin_path": row["bin_path"],
            "alias": row["alias"],
            "crate": row["crate"],
        }
    finally:
        conn.close()
