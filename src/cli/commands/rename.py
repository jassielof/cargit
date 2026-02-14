"""Rename command for cargit."""

import sys
from pathlib import Path

import typer
from rich import print as rprint

from cli.core import CargitError, ensure_dirs
from cli.storage import (
    get_binary_metadata,
    remove_binary_metadata,
    save_binary_metadata,
)

app = typer.Typer()


@app.command()
def rename(
    current_alias: str = typer.Argument(..., help="Current alias of the binary"),
    new_alias: str = typer.Option(..., "--to", help="New alias for the binary"),
):
    """
    Rename an installed binary's alias
    """
    try:
        ensure_dirs()

        # Check if old alias exists
        old_info = get_binary_metadata(current_alias)
        if old_info is None:
            rprint(f"[red]Error: Binary '{current_alias}' not found[/red]")
            sys.exit(1)

        # Check if new alias already exists
        new_info = get_binary_metadata(new_alias)
        if new_info is not None:
            rprint(f"[red]Error: Binary '{new_alias}' already exists[/red]")
            sys.exit(1)

        import os
        import shutil

        install_dir = Path(old_info["install_dir"])
        old_path = install_dir / current_alias

        # Try with platform extension if not found (e.g., .exe on Windows)
        if not old_path.exists():
            if old_info.get("binary_copy_path"):
                old_path = Path(old_info["binary_copy_path"])
            elif os.name == "nt" and not current_alias.endswith(".exe"):
                old_path = install_dir / f"{current_alias}.exe"

        # Preserve extension from old path
        if old_path.suffix and not new_alias.endswith(old_path.suffix):
            new_path = install_dir / f"{new_alias}{old_path.suffix}"
        else:
            new_path = install_dir / new_alias

        if old_path.exists():
            if new_path.exists():
                new_path.unlink()
            shutil.move(str(old_path), str(new_path))
            rprint(f"[green]Renamed binary: {current_alias} -> {new_alias}[/green]")
        else:
            rprint(
                f"[yellow]Warning: Binary not found at {old_path}, attempting to copy from build output[/yellow]"
            )
            source_fallback = Path(old_info["bin_path"])
            if not source_fallback.exists():
                rprint("[red]Error: No source binary found to rename[/red]")
                sys.exit(1)
            shutil.copy2(source_fallback, new_path)
            new_path.chmod(0o755)

        # Update metadata: save with new alias, remove old
        save_binary_metadata(
            alias=new_alias,
            repo_url=old_info["repo_url"],
            branch=old_info["branch"],
            commit=old_info["commit"],
            install_dir=old_info["install_dir"],
            bin_path=old_info["bin_path"],
            crate=old_info.get("crate"),
            binary_type=old_info.get("binary_type", "copy"),
            binary_copy_path=str(new_path),
        )

        remove_binary_metadata(current_alias)

        rprint(f"[green]Successfully renamed {current_alias} to {new_alias}![/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)
