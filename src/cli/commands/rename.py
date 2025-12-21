"""Rename command for cargit."""

import sys
from pathlib import Path

import typer
from rich import print as rprint

from cli.core import CargitError, ensure_dirs
from cli.storage import get_binary_metadata, save_binary_metadata, remove_binary_metadata


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

        install_dir = Path(old_info["install_dir"])
        old_symlink = install_dir / current_alias
        new_symlink = install_dir / new_alias

        # Create new symlink pointing to the same binary
        if old_symlink.exists() or old_symlink.is_symlink():
            target = old_symlink.resolve()
            new_symlink.symlink_to(target)
            rprint(f"[green]Created new symlink: {new_alias} -> {target}[/green]")

            # Remove old symlink
            old_symlink.unlink()
            rprint(f"[blue]Removed old symlink: {current_alias}[/blue]")
        else:
            rprint(f"[yellow]Warning: Old symlink not found at {old_symlink}[/yellow]")
            # Still update metadata even if symlink is missing
            new_symlink.symlink_to(old_info["bin_path"])

        # Update metadata: save with new alias, remove old
        save_binary_metadata(
            alias=new_alias,
            repo_url=old_info["repo_url"],
            branch=old_info["branch"],
            commit=old_info["commit"],
            install_dir=old_info["install_dir"],
            bin_path=old_info["bin_path"],
            crate=old_info.get("crate"),
        )

        remove_binary_metadata(current_alias)

        rprint(f"[green]Successfully renamed {current_alias} to {new_alias}![/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)
