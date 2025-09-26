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
from cargit.storage import load_metadata, save_metadata
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
    name: str | None = typer.Option(None, "--name", help="Alias for installed binary"),
    dir_path: str | None = typer.Option(None, "--dir", help="Install directory"),
):
    """Install a Rust binary from git repository"""
    try:
        ensure_dirs()

        # Clone or update repository
        repo_path, actual_branch = clone_repository(git_url, branch)

        # Determine binary name/alias
        if name is None:
            from .core import get_binary_name_from_cargo

            name = get_binary_name_from_cargo(repo_path, crate)

        # Determine install directory
        install_dir = Path(dir_path) if dir_path else BIN_DIR

        # Build binary
        binary_path = build_binary(repo_path, crate)

        # Install binary
        install_binary(binary_path, name, install_dir)

        # Update metadata
        metadata = load_metadata()
        metadata["installed"][name] = {
            "repo_url": git_url,
            "branch": actual_branch,
            "commit": get_current_commit(repo_path),
            "install_dir": str(install_dir),
            "bin_path": str(binary_path),
            "alias": name,
            "crate": crate,
        }
        save_metadata(metadata)

        rprint(f"[green]Successfully installed {name}![/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def update(
    all: bool = typer.Option(False, "--all", help="Update all installed binaries"),
    name: str | None = typer.Option(None, "--name", help="Update specific binary"),
):
    """Update installed binaries"""
    from .core import get_remote_commit, run_command

    try:
        ensure_dirs()
        metadata = load_metadata()

        if not metadata["installed"]:
            rprint("[yellow]No binaries installed[/yellow]")
            return

        if not all and not name:
            rprint("[red]Error: Must specify --all or --name <binary>[/red]")
            sys.exit(1)

        targets = []
        if all:
            # Use the built-in list() function explicitly
            targets = __builtins__["list"](metadata["installed"].keys())
        elif name:
            if name not in metadata["installed"]:
                rprint(f"[red]Error: Binary '{name}' not found[/red]")
                sys.exit(1)
            targets = [name]

        for binary_name in targets:
            rprint(f"[blue]Checking {binary_name}...[/blue]")

            info = metadata["installed"][binary_name]
            repo_path = get_repo_path(info["repo_url"])

            if not repo_path.exists():
                rprint(
                    f"[yellow]Repository missing for {binary_name}, reinstalling...[/yellow]"
                )
                # Reinstall
                repo_path, _ = clone_repository(info["repo_url"], info.get("branch"))
                binary_path = build_binary(repo_path, info.get("crate"))
                install_binary(binary_path, binary_name, Path(info["install_dir"]))

                # Update metadata
                info["commit"] = get_current_commit(repo_path)
                info["bin_path"] = str(binary_path)

            else:
                # Check for updates
                run_command(["git", "fetch", "origin"], cwd=repo_path)

                current_commit = get_current_commit(repo_path)
                remote_commit = get_remote_commit(repo_path, info["branch"])

                if current_commit == remote_commit:
                    rprint(f"[green]{binary_name} is up to date[/green]")
                    continue

                rprint(f"[blue]Updating {binary_name}...[/blue]")

                # Update repository
                run_command(
                    ["git", "reset", "--hard", f"origin/{info['branch']}"],
                    cwd=repo_path,
                )

                # Rebuild and reinstall
                binary_path = build_binary(repo_path, info.get("crate"))
                install_binary(binary_path, binary_name, Path(info["install_dir"]))

                # Update metadata
                info["commit"] = get_current_commit(repo_path)
                info["bin_path"] = str(binary_path)

                rprint(f"[green]Updated {binary_name}![/green]")

        save_metadata(metadata)

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
def remove(name: str = typer.Option(..., "--name", help="Name of binary to remove")):
    """Remove installed binary"""
    try:
        ensure_dirs()
        metadata = load_metadata()

        if name not in metadata["installed"]:
            rprint(f"[red]Error: Binary '{name}' not found[/red]")
            sys.exit(1)

        info = metadata["installed"][name]

        # Remove symlink
        symlink_path = Path(info["install_dir"]) / name
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
        del metadata["installed"][name]
        save_metadata(metadata)

        rprint(f"[green]Successfully removed {name}![/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def which(name: str = typer.Option(..., "--name", help="Name of binary to locate")):
    """Show absolute path to installed binary"""
    metadata = load_metadata()

    if name not in metadata["installed"]:
        rprint(f"[red]Error: Binary '{name}' not found[/red]")
        sys.exit(1)

    info = metadata["installed"][name]
    symlink_path = Path(info["install_dir"]) / name

    if symlink_path.exists():
        rprint(str(symlink_path.resolve()))
    else:
        rprint(f"[red]Error: Binary path does not exist: {symlink_path}[/red]")
        sys.exit(1)
