"""Remove command for cargit."""

import os
import shutil
import sys
from pathlib import Path

import typer
from rich import print as rprint

from cli.core import CargitError, ensure_dirs, get_repo_path
from cli.storage import get_binary_metadata, remove_binary_metadata

app = typer.Typer()


@app.command()
def remove(alias: str = typer.Argument(..., help="Alias of binary to remove")):
    """Remove installed binary

    Examples:
        cargit remove typst
        cargit remove my-tool
    """
    try:
        ensure_dirs()
        info = get_binary_metadata(alias)

        if info is None:
            rprint(f"[red]Error: Binary '{alias}' not found[/red]")
            sys.exit(1)

        # Remove installed binary copy
        installed_path = Path(info["install_dir"]) / alias

        # Try with platform extension if not found (e.g., .exe on Windows)
        if not installed_path.exists():
            if info.get("binary_copy_path"):
                installed_path = Path(info["binary_copy_path"])
            elif os.name == "nt" and not alias.endswith(".exe"):
                installed_path = Path(info["install_dir"]) / f"{alias}.exe"

        if installed_path.exists() or installed_path.is_symlink():
            installed_path.unlink()
            rprint(f"[blue]Removed binary: {installed_path}[/blue]")

        # Remove cached repo
        repo_path = get_repo_path(info["repo_url"])
        if repo_path.exists():
            try:
                shutil.rmtree(repo_path)
                rprint(f"[blue]Removed cached repo: {repo_path}[/blue]")
            except (PermissionError, OSError) as e:
                rprint(
                    f"[yellow]Warning: Could not fully remove cached repo (some files may be locked): {e}[/yellow]"
                )
                # Try to remove with ignore_errors as fallback
                shutil.rmtree(repo_path, ignore_errors=True)

        # Remove from metadata
        remove_binary_metadata(alias)

        rprint(f"[green]Successfully removed {alias}![/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)
