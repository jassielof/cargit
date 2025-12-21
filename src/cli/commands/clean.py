"""Clean command for cargit."""

import shutil
import sys
from pathlib import Path

import typer
from rich import print as rprint

from cli.core import (
    CACHE_DIR,
    CargitError,
    ensure_dirs,
    get_repo_path,
    get_cache_size,
    format_size,
    clean_all_cache,
    clean_binary_artifacts,
    clean_binary_repo,
    copy_binary_to_safe_location,
    find_orphaned_repos,
    install_binary,
)
from cli.storage import get_binary_metadata, mark_artifacts_cleaned, mark_repo_deleted


def clean(
    alias: str | None = typer.Argument(None, help="Alias of binary to clean"),
    all: bool = typer.Option(False, "--all", help="Remove all cached repos and artifacts"),
    artifacts: bool = typer.Option(False, "--artifacts", help="Clean build artifacts"),
    repos: bool = typer.Option(False, "--repos", help="Delete repository"),
    orphaned: bool = typer.Option(False, "--orphaned", help="Remove orphaned repos"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without deleting"),
):
    """Clean cache to free disk space"""
    try:
        ensure_dirs()
        _validate_clean_args(alias, all, artifacts, repos, orphaned)

        if all:
            _clean_all_cache(dry_run)
        elif artifacts:
            _clean_binary_artifacts(alias, dry_run)
        elif repos:
            _clean_binary_repo(alias, dry_run)
        elif orphaned:
            _clean_orphaned_repos(dry_run)

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _validate_clean_args(alias, all, artifacts, repos, orphaned):
    mode_count = sum([all, artifacts, repos, orphaned])
    if mode_count == 0:
        rprint("[red]Error: Must specify one of --all, --artifacts, --repos, or --orphaned[/red]")
        sys.exit(1)
    if mode_count > 1:
        rprint("[red]Error: Can only specify one mode at a time[/red]")
        sys.exit(1)
    if (artifacts or repos) and not alias:
        rprint("[red]Error: Must specify <alias> when using --artifacts or --repos[/red]")
        sys.exit(1)
    if (all or orphaned) and alias:
        rprint("[red]Error: Cannot specify <alias> with --all or --orphaned[/red]")
        sys.exit(1)


def _clean_all_cache(dry_run: bool):
    cache_size = get_cache_size(CACHE_DIR) if CACHE_DIR.exists() else 0

    if dry_run:
        rprint(f"[yellow]Would remove all cache ({format_size(cache_size)})[/yellow]")
        return

    if cache_size == 0:
        rprint("[yellow]Cache is already empty[/yellow]")
        return

    rprint(f"[yellow]This will remove all cached repos and artifacts ({format_size(cache_size)})[/yellow]")
    if not typer.confirm("Are you sure?"):
        rprint("[blue]Aborted[/blue]")
        return

    if clean_all_cache():
        rprint(f"[green]Successfully cleaned cache (freed {format_size(cache_size)})[/green]")
    else:
        rprint("[red]Failed to clean cache[/red]")
        sys.exit(1)


def _clean_binary_artifacts(alias: str, dry_run: bool):
    info = get_binary_metadata(alias)
    if info is None:
        rprint(f"[red]Error: Binary '{alias}' not found[/red]")
        sys.exit(1)

    repo_path = get_repo_path(info["repo_url"])
    target_dir = repo_path / "target"
    if not repo_path.exists():
        rprint(f"[yellow]Repository not found at {repo_path}")
        return

    artifacts_size = get_cache_size(target_dir) if target_dir.exists() else 0

    if dry_run:
        rprint(f"[yellow]Would clean build artifacts for {alias} ({format_size(artifacts_size)})[/yellow]")
        return

    if artifacts_size == 0:
        rprint(f"[yellow]No build artifacts to clean for {alias}")
        return

    binary_path = Path(info["bin_path"])
    if not binary_path.exists():
        rprint(f"[red]Error: Binary not found at {binary_path}")
        sys.exit(1)

    rprint("[blue]Copying binary to safe location...[/blue]")
    safe_path = copy_binary_to_safe_location(binary_path, alias)

    rprint(f"[blue]Cleaning build artifacts ({format_size(artifacts_size)})...[/blue]")
    if clean_binary_artifacts(repo_path):
        install_binary(safe_path, alias, Path(info["install_dir"]), binary_type="copy")
        mark_artifacts_cleaned(alias, str(safe_path))
        rprint(f"[green]Cleaned artifacts for {alias} (freed {format_size(artifacts_size)})[/green]")
    else:
        rprint(f"[red]Failed to clean artifacts for {alias}")
        sys.exit(1)


def _clean_binary_repo(alias: str, dry_run: bool):
    info = get_binary_metadata(alias)
    if info is None:
        rprint(f"[red]Error: Binary '{alias}' not found[/red]")
        sys.exit(1)

    repo_path = get_repo_path(info["repo_url"])
    if not repo_path.exists():
        rprint(f"[yellow]Repository not found at {repo_path}")
        return

    repo_size = get_cache_size(repo_path)

    if dry_run:
        rprint(f"[yellow]Would delete repository for {alias} ({format_size(repo_size)})[/yellow]")
        return

    binary_path = Path(info["bin_path"])
    if not binary_path.exists():
        rprint(f"[red]Error: Binary not found at {binary_path}")
        sys.exit(1)

    rprint("[blue]Copying binary to safe location...[/blue]")
    safe_path = copy_binary_to_safe_location(binary_path, alias)

    rprint(f"[blue]Deleting repository ({format_size(repo_size)})...[/blue]")
    if clean_binary_repo(repo_path):
        install_binary(safe_path, alias, Path(info["install_dir"]), binary_type="copy")
        mark_repo_deleted(alias)
        mark_artifacts_cleaned(alias, str(safe_path))
        rprint(f"[green]Deleted repository for {alias} (freed {format_size(repo_size)})[/green]")
    else:
        rprint(f"[red]Failed to delete repository for {alias}")
        sys.exit(1)


def _clean_orphaned_repos(dry_run: bool):
    orphaned_repos = find_orphaned_repos()
    if not orphaned_repos:
        rprint("[green]No orphaned repositories found[/green]")
        return

    total_size = sum(size for _, size in orphaned_repos)
    rprint(f"[yellow]Found {len(orphaned_repos)} orphaned repos ({format_size(total_size)}):[/yellow]")
    for repo_path, size in orphaned_repos:
        rprint(f"  - {repo_path} ({format_size(size)})")

    if dry_run:
        rprint(f"[yellow]Would remove {len(orphaned_repos)} orphaned repos[/yellow]")
        return

    if not typer.confirm(f"Remove {len(orphaned_repos)} orphaned repositories?"):
        rprint("[blue]Aborted[/blue]")
        return

    removed = 0
    for repo_path, _ in orphaned_repos:
        try:
            shutil.rmtree(repo_path)
            removed += 1
        except Exception as e:
            rprint(f"[yellow]Warning: Could not delete {repo_path}: {e}")

    rprint(f"[green]Removed {removed}/{len(orphaned_repos)} orphaned repos[/green]")
