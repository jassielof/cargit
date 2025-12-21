"""Cargit - Git-based cargo binary installer.

Fast incremental updates through cached repositories and build artifacts.
"""

from __future__ import annotations

import atexit


def main() -> None:
    """Entry point for the cargit CLI."""
    # Lazy import for faster startup
    from cli.cli import app
    from cli.storage import cleanup_connection

    # Register cleanup handler
    atexit.register(cleanup_connection)

    app()


__all__ = ["main"]
