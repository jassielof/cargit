"""Remove command for cargit."""

import shutil
import sys
from pathlib import Path

import typer
from rich import print as rprint

from cli.core import CargitError, ensure_dirs, get_repo_path
from cli.storage import get_binary_metadata, remove_binary_metadata


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

        # Remove symlink
        symlink_path = Path(info["install_dir"]) / alias
        if symlink_path.exists():
            symlink_path.unlink()
            rprint(f"[blue]Removed binary: {symlink_path}[/blue]")

        # Remove cached repo
        repo_path = get_repo_path(info["repo_url"])
        if repo_path.exists():
            shutil.rmtree(repo_path)
            rprint(f"[blue]Removed cached repo: {repo_path}[/blue]")

        # Remove from metadata
        remove_binary_metadata(alias)

        rprint(f"[green]Successfully removed {alias}![/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)
