"""Install command for cargit."""

import sys
from pathlib import Path

import typer
from rich import print as rprint

from cli.core import (
    BIN_DIR,
    CargitError,
    build_binary,
    clone_repository,
    ensure_dirs,
    get_binary_name_from_cargo,
    get_current_commit,
    install_binary,
)
from cli.storage import get_binary_metadata, get_binaries_by_repo, save_binary_metadata

def install(
    git_url: str = typer.Argument(..., help="Git repository URL"),
    crate: str | None = typer.Argument(
        None, help="Crate name (for workspaces with multiple crates)"
    ),
    branch: str | None = typer.Option(None, "--branch", help="Branch to install from"),
    alias: str | None = typer.Option(
        None, "--alias", help="Alias for installed binary"
    ),
    dir_path: str | None = typer.Option(None, "--dir", help="Install directory"),
):
    """Install a Rust binary from git repository.

    For workspace repositories with multiple crates, specify the crate name
    as the second argument.

    Examples:
        cargit install https://github.com/typst/typst typst-cli
        cargit install https://github.com/sharkdp/fd
        cargit install https://github.com/BurntSushi/ripgrep --alias rg
    """
    try:
        ensure_dirs()

        # Check if we already have binaries from this repo
        existing = get_binaries_by_repo(git_url)
        if existing:
            existing_aliases = [b["alias"] for b in existing]
            rprint(
                f"[yellow]Note: Repository already has installed binaries: {', '.join(existing_aliases)}[/yellow]"
            )
            rprint(
                "[yellow]They share the same repository and will be at the same commit.[/yellow]"
            )

        # Clone or update repository
        repo_path, actual_branch = clone_repository(git_url, branch)

        # Determine binary name/alias
        if alias is None:
            alias = get_binary_name_from_cargo(repo_path, crate)

        # Check if alias already exists
        existing_binary = get_binary_metadata(alias)
        if existing_binary is not None:
            rprint(f"[red]Error: Binary '{alias}' already exists[/red]")
            rprint(
                f"[dim]Use --alias to specify a different name, or 'cargit remove {alias}' first[/dim]"
            )
            sys.exit(1)

        # Determine install directory
        install_dir = Path(dir_path) if dir_path else BIN_DIR

        # Build binary
        binary_path, build_duration = build_binary(repo_path, crate, alias)

        # Install binary
        install_binary(binary_path, alias, install_dir)

        # Update metadata
        save_binary_metadata(
            alias=alias,
            repo_url=git_url,
            branch=actual_branch,
            commit=get_current_commit(repo_path),
            install_dir=str(install_dir),
            bin_path=str(binary_path),
            crate=crate,
        )

        rprint(f"[green]Successfully installed {alias}![/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)
