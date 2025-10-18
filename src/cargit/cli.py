import sys
from pathlib import Path

import typer
from rich import print as rprint

from cargit.core import (
    CargitError,
    build_binary,
    clone_repository,
    ensure_dirs,
    get_current_commit,
    get_repo_path,
    install_binary,
)
from cargit.storage import (
    load_metadata,
    save_binary_metadata,
    remove_binary_metadata,
    get_binary_metadata,
)
from cargit.utils import display_installed_table
from cargit.core import BIN_DIR

app = typer.Typer(help="Git-based cargo binary installer with incremental updates")


@app.command()
def install(
    git_url: str = typer.Argument(..., help="Git repository URL"),
    crate: str | None = typer.Argument(
        None, help="Crate name (for workspaces with multiple crates)"
    ),
    branch: str | None = typer.Option(None, "--branch", help="Branch to track"),
    alias: str | None = typer.Option(None, "--alias", help="Alias for installed binary"),
    dir_path: str | None = typer.Option(None, "--dir", help="Install directory"),
):
    """Install a Rust binary from git repository"""
    try:
        ensure_dirs()

        # Clone or update repository
        repo_path, actual_branch = clone_repository(git_url, branch)

        # Determine binary name/alias
        if alias is None:
            from .core import get_binary_name_from_cargo

            alias = get_binary_name_from_cargo(repo_path, crate)

        # Determine install directory
        install_dir = Path(dir_path) if dir_path else BIN_DIR

        # Build binary
        binary_path = build_binary(repo_path, crate)

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


@app.command()
def update(
    all: bool = typer.Option(False, "--all", help="Update all installed binaries"),
    alias: str | None = typer.Option(None, "--alias", help="Update specific binary"),
):
    """Update installed binaries"""
    from .core import get_remote_commit, run_command

    try:
        ensure_dirs()
        metadata = load_metadata()

        if not metadata["installed"]:
            rprint("[yellow]No binaries installed[/yellow]")
            return

        if not all and not alias:
            rprint("[red]Error: Must specify --all or --alias <binary>[/red]")
            sys.exit(1)

        targets = []
        if all:
            targets = list(metadata["installed"].keys())
        elif alias:
            if alias not in metadata["installed"]:
                rprint(f"[red]Error: Binary '{alias}' not found[/red]")
                sys.exit(1)
            targets = [alias]

        for binary_alias in targets:
            rprint(f"[blue]Checking {binary_alias}...[/blue]")

            info = metadata["installed"][binary_alias]
            repo_path = get_repo_path(info["repo_url"])

            if not repo_path.exists():
                rprint(
                    f"[yellow]Repository missing for {binary_alias}, reinstalling...[/yellow]"
                )
                # Reinstall
                repo_path, _ = clone_repository(info["repo_url"], info.get("branch"))
                binary_path = build_binary(repo_path, info.get("crate"))
                install_binary(binary_path, binary_alias, Path(info["install_dir"]))

                # Update metadata
                save_binary_metadata(
                    alias=binary_alias,
                    repo_url=info["repo_url"],
                    branch=info["branch"],
                    commit=get_current_commit(repo_path),
                    install_dir=info["install_dir"],
                    bin_path=str(binary_path),
                    crate=info.get("crate"),
                )

            else:
                # Check for updates
                run_command(["git", "fetch", "origin"], cwd=repo_path)

                current_commit = get_current_commit(repo_path)
                remote_commit = get_remote_commit(repo_path, info["branch"])

                if current_commit == remote_commit:
                    rprint(f"[green]{binary_alias} is up to date[/green]")
                    continue

                rprint(f"[blue]Updating {binary_alias}...[/blue]")

                # Update repository
                run_command(
                    ["git", "reset", "--hard", f"origin/{info['branch']}"],
                    cwd=repo_path,
                )

                # Rebuild and reinstall
                binary_path = build_binary(repo_path, info.get("crate"))
                install_binary(binary_path, binary_alias, Path(info["install_dir"]))

                # Update metadata
                save_binary_metadata(
                    alias=binary_alias,
                    repo_url=info["repo_url"],
                    branch=info["branch"],
                    commit=get_current_commit(repo_path),
                    install_dir=info["install_dir"],
                    bin_path=str(binary_path),
                    crate=info.get("crate"),
                )

                rprint(f"[green]Updated {binary_alias}![/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command("list")
def list_binaries():
    """List installed binaries"""
    metadata = load_metadata()

    if not metadata["installed"]:
        rprint("[yellow]No binaries installed[/yellow]")
        return

    display_installed_table(metadata["installed"])


@app.command()
def remove(alias: str = typer.Option(..., "--alias", help="Alias of binary to remove")):
    """Remove installed binary"""
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
            import shutil

            shutil.rmtree(repo_path)
            rprint(f"[blue]Removed cached repo: {repo_path}[/blue]")

        # Remove from metadata
        remove_binary_metadata(alias)

        rprint(f"[green]Successfully removed {alias}![/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def which(alias: str = typer.Option(..., "--alias", help="Alias of binary to locate")):
    """Show absolute path to installed binary"""
    info = get_binary_metadata(alias)

    if info is None:
        rprint(f"[red]Error: Binary '{alias}' not found[/red]")
        sys.exit(1)

    symlink_path = Path(info["install_dir"]) / alias

    if symlink_path.exists():
        rprint(str(symlink_path.resolve()))
    else:
        rprint(f"[red]Error: Binary path does not exist: {symlink_path}[/red]")
        sys.exit(1)
