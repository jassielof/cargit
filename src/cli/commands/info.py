"""Info command for cargit."""

import shutil
import sys
import tempfile
from pathlib import Path

import typer
from rich import print as rprint

from cli.core import (
    CargitError,
    ensure_dirs,
    get_repo_path,
    run_command,
    _expand_workspace_members,
    _get_available_crates_from_members,
)
from cli.storage import get_binaries_by_repo


def info(
    git_url: str = typer.Argument(..., help="Git repository URL to inspect"),
):
    """Show information about a repository.

    Displays available crates, binary names, and workspace structure
    for a git repository without installing anything.

    Examples:
        cargit info https://github.com/typst/typst
        cargit info https://github.com/sharkdp/fd
    """
    try:
        ensure_dirs()

        repo_path, temp_clone = _prepare_repo_for_info(git_url)

        try:
            cargo_data = _load_cargo_data(repo_path)
            rprint(f"\n[bold cyan]Repository: {git_url}[/bold cyan]")
            rprint(f"{'─' * 60}\n")

            if "workspace" in cargo_data:
                _render_workspace_info(repo_path, cargo_data, git_url)
            else:
                _render_single_crate_info(cargo_data, git_url)

            _render_existing_installations(git_url)

        finally:
            if temp_clone:
                shutil.rmtree(repo_path, ignore_errors=True)

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _prepare_repo_for_info(git_url: str) -> tuple[Path, bool]:
    repo_path = get_repo_path(git_url)
    if repo_path.exists():
        return repo_path, False

    rprint("[blue]Fetching repository information...[/blue]")
    temp_dir = tempfile.mkdtemp(prefix="cargit_info_")
    repo_path = Path(temp_dir)

    run_command(
        ["git", "clone", "--depth=1", "--single-branch", git_url, str(repo_path)]
    )
    return repo_path, True


def _load_cargo_data(repo_path: Path) -> dict:
    cargo_toml = repo_path / "Cargo.toml"
    if not cargo_toml.exists():
        rprint("[red]Error: No Cargo.toml found in repository[/red]")
        sys.exit(1)

    try:
        import tomllib

        with open(cargo_toml, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        import tomlkit

        with open(cargo_toml, "r", encoding="utf-8") as f:
            return tomlkit.load(f)


def _render_workspace_info(repo_path: Path, cargo_data: dict, git_url: str):
    rprint("[bold]Type:[/bold] Cargo Workspace")

    default_members = cargo_data["workspace"].get("default-members", [])
    if default_members:
        rprint(f"[bold]Default member:[/bold] {default_members[0]}")

    members = cargo_data["workspace"].get("members", [])
    expanded = _expand_workspace_members(repo_path, members)
    crates = _get_available_crates_from_members(repo_path, expanded)

    rprint(f"\n[bold]Available crates ({len(crates)}):[/bold]")
    for crate in sorted(crates):
        is_default = any(crate in m for m in default_members)
        marker = " [green](default)[/green]" if is_default else ""
        rprint(f"  • {crate}{marker}")

    rprint(f"\n[dim]Install with: cargit install {git_url} <crate_name>[/dim]")


def _render_single_crate_info(cargo_data: dict, git_url: str):
    rprint("[bold]Type:[/bold] Single Crate")

    if "package" in cargo_data:
        pkg = cargo_data["package"]
        name = pkg.get("name", "unknown")
        version = pkg.get("version", "unknown")
        description = pkg.get("description", "")

        rprint(f"[bold]Name:[/bold] {name}")
        rprint(f"[bold]Version:[/bold] {version}")
        if description:
            rprint(f"[bold]Description:[/bold] {description}")

    rprint(f"\n[dim]Install with: cargit install {git_url}[/dim]")


def _render_existing_installations(git_url: str):
    existing = get_binaries_by_repo(git_url)
    if not existing:
        return

    rprint("\n[bold yellow]Already installed from this repo:[/bold yellow]")
    for binary in existing:
        crate_info = f" (crate: {binary['crate']})" if binary.get("crate") else ""
        rprint(f"  • {binary['alias']}{crate_info} @ {binary['commit'][:8]}")
