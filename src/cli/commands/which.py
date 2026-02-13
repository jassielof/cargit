"""Which command for cargit."""

import os
import sys
from pathlib import Path

import typer
from rich import print as rprint

from cli.storage import get_binary_metadata

app = typer.Typer()


@app.command()
def which(alias: str = typer.Argument(..., help="Alias of binary to locate")):
    """Show absolute path to installed binary

    Examples:
        cargit which typst
        cargit which my-tool
    """
    info = get_binary_metadata(alias)

    if info is None:
        rprint(f"[red]Error: Binary '{alias}' not found[/red]")
        sys.exit(1)

    binary_path = Path(info["install_dir"]) / alias

    # Try with platform extension if not found (e.g., .exe on Windows)
    if not binary_path.exists():
        if info.get("binary_copy_path"):
            binary_path = Path(info["binary_copy_path"])
        elif os.name == "nt" and not alias.endswith(".exe"):
            binary_path = Path(info["install_dir"]) / f"{alias}.exe"

    if binary_path.exists():
        rprint(str(binary_path.resolve()))
    else:
        rprint(f"[red]Error: Binary path does not exist: {binary_path}[/red]")
        sys.exit(1)
