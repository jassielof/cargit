"""Sync command for cargit."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from cli.core import (
    CargitError,
    build_binary,
    ensure_dirs,
    get_current_commit,
    get_repo_path,
    install_binary,
    fetch_repo_silent,
    check_update_available,
    reset_to_remote,
    get_default_branch,
)
from cli.storage import (
    load_metadata,
    save_binary_metadata,
    get_binary_metadata,
    reset_cache_flags,
)


def sync(
    jobs: int = typer.Option(8, "--jobs", "-j", help="Number of parallel git operations"),
    fetch_only: bool = typer.Option(False, "--fetch-only", help="Only fetch, don't reset or build"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
):
    """Sync all binaries: parallel fetch & reset, then sequential builds."""
    try:
        ensure_dirs()
        metadata = load_metadata()

        if not metadata["installed"]:
            rprint("[yellow]No binaries installed[/yellow]")
            return

        sync_items = _collect_sync_items(metadata)
        if not sync_items:
            rprint("[yellow]No binaries to sync[/yellow]")
            return

        rprint(f"[blue]Syncing {len(sync_items)} binaries...[/blue]\n")

        fetch_results = _sync_phase_fetch(sync_items, jobs)
        updates_needed, up_to_date = _evaluate_updates(sync_items, fetch_results)

        if not updates_needed:
            rprint("\n[green]All binaries are up to date![/green]")
            return

        if _maybe_exit_after_check(updates_needed, dry_run, fetch_only):
            return

        updates_after_reset = _sync_phase_reset(updates_needed, jobs)
        if not updates_after_reset:
            rprint("\n[yellow]No binaries to build after reset failures[/yellow]")
            return

        _sync_phase_build(updates_after_reset, up_to_date)

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _collect_sync_items(metadata: dict) -> list[dict]:
    sync_items: list[dict] = []
    for binary_alias in metadata["installed"]:
        info = get_binary_metadata(binary_alias)
        if info is None:
            continue

        stored_branch = info.get("branch", "")
        if stored_branch.startswith(("commit:", "tag:")):
            rprint(f"[dim]Skipping {binary_alias}: pinned to {stored_branch}[/dim]")
            continue

        repo_path = get_repo_path(info["repo_url"])
        if not repo_path.exists():
            if info.get("repo_deleted", False):
                rprint(f"[yellow]{binary_alias}: needs reclone (repo was cleaned)[/yellow]")
            else:
                rprint(f"[yellow]{binary_alias}: repo missing[/yellow]")
            continue

        if info.get("artifacts_cleaned", False):
            rprint(f"[yellow]{binary_alias}: needs rebuild (artifacts cleaned)[/yellow]")

        sync_items.append({
            "alias": binary_alias,
            "info": info,
            "repo_path": repo_path,
            "branch": stored_branch or None,
        })

    return sync_items


def _sync_phase_fetch(sync_items: list[dict], jobs: int) -> dict[str, bool]:
    rprint("[bold cyan]Phase 1/3: Fetching repositories[/bold cyan]")
    fetch_results: dict[str, bool] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Fetching...", total=len(sync_items))

        def fetch_one(item: dict) -> tuple[str, bool, str | None]:
            success, error = fetch_repo_silent(item["repo_path"], item["branch"])
            return item["alias"], success, error

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(fetch_one, item): item["alias"] for item in sync_items}
            for future in as_completed(futures):
                alias, success, error = future.result()
                fetch_results[alias] = success
                if not success:
                    progress.console.print(f"  [yellow]✗ {alias}: {error}[/yellow]")
                progress.advance(task)

    return fetch_results


def _evaluate_updates(sync_items: list[dict], fetch_results: dict[str, bool]):
    updates_needed: list[dict] = []
    up_to_date: list[str] = []

    for item in sync_items:
        alias = item["alias"]
        if not fetch_results.get(alias, False):
            continue

        has_update, current, remote = check_update_available(item["repo_path"], item["branch"])
        needs_rebuild = item["info"].get("artifacts_cleaned", False)

        if has_update:
            item.update({"current_commit": current, "remote_commit": remote})
            updates_needed.append(item)
        elif needs_rebuild:
            item.update({"current_commit": current, "remote_commit": current, "rebuild_only": True})
            updates_needed.append(item)
        else:
            up_to_date.append(alias)

    rprint(f"  [green]✓ Fetched {sum(fetch_results.values())}/{len(sync_items)} repos[/green]")
    if up_to_date:
        rprint(f"  [dim]Already up to date: {', '.join(up_to_date)}[/dim]")
    if updates_needed:
        rprint(f"  [yellow]Updates available: {len(updates_needed)}[/yellow]")

    return updates_needed, up_to_date


def _maybe_exit_after_check(updates_needed: list[dict], dry_run: bool, fetch_only: bool) -> bool:
    if dry_run:
        rprint("\n[yellow]Dry run - would update:[/yellow]")
        for item in updates_needed:
            if item.get("rebuild_only"):
                rprint(f"  [cyan]{item['alias']}[/cyan]: rebuild (artifacts cleaned)")
            else:
                rprint(f"  [cyan]{item['alias']}[/cyan]: {item['current_commit'][:8]} → {item['remote_commit'][:8]}")
        return True

    if fetch_only:
        rprint("\n[blue]Fetch-only mode - skipping reset and build[/blue]")
        return True

    return False


def _sync_phase_reset(updates_needed: list[dict], jobs: int) -> list[dict]:
    rprint("\n[bold cyan]Phase 2/3: Resetting repositories[/bold cyan]")

    reset_items = [item for item in updates_needed if not item.get("rebuild_only")]
    if not reset_items:
        rprint("  [dim]No git resets needed (rebuild only)[/dim]")
        return updates_needed

    reset_results: dict[str, bool] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Resetting...", total=len(reset_items))

        def reset_one(item: dict) -> tuple[str, bool, str | None]:
            branch = item["branch"] or _safe_default_branch(item)
            success, error = reset_to_remote(item["repo_path"], branch)
            return item["alias"], success, error

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(reset_one, item): item["alias"] for item in reset_items}
            for future in as_completed(futures):
                alias, success, error = future.result()
                reset_results[alias] = success
                if not success:
                    progress.console.print(f"  [yellow]✗ {alias}: {error}[/yellow]")
                progress.advance(task)

    rprint(f"  [green]✓ Reset {sum(reset_results.values())}/{len(reset_items)} repos[/green]")

    return [
        item for item in updates_needed
        if item.get("rebuild_only") or reset_results.get(item["alias"], False)
    ]


def _safe_default_branch(item: dict) -> str:
    try:
        return get_default_branch(item["repo_path"])
    except Exception:
        raise CargitError("Could not determine branch")


def _sync_phase_build(updates_needed: list[dict], up_to_date: list[str]):
    rprint(f"\n[bold cyan]Phase 3/3: Building {len(updates_needed)} binaries[/bold cyan]")

    built = 0
    failed = 0

    for idx, item in enumerate(updates_needed, 1):
        alias = item["alias"]
        info = item["info"]

        rprint(f"\n[blue][{idx}/{len(updates_needed)}] Building {alias}...[/blue]")

        try:
            binary_path, _ = build_binary(item["repo_path"], info.get("crate"), alias)
            install_binary(binary_path, alias, Path(info["install_dir"]))

            branch = item["branch"] or get_default_branch(item["repo_path"])
            if info.get("artifacts_cleaned", False) or info.get("repo_deleted", False):
                reset_cache_flags(alias)

            save_binary_metadata(
                alias=alias,
                repo_url=info["repo_url"],
                branch=branch,
                commit=get_current_commit(item["repo_path"]),
                install_dir=info["install_dir"],
                bin_path=str(binary_path),
                crate=info.get("crate"),
            )

            rprint(f"  [green]✓ {alias} built successfully[/green]")
            built += 1

        except CargitError as e:
            rprint(f"  [red]✗ {alias} failed: {e}[/red]")
            failed += 1

    rprint(f"\n[bold]{'═' * 50}[/bold]")
    rprint("[bold green]Sync complete![/bold green]")
    rprint(f"  Built: {built}, Failed: {failed}, Skipped: {len(up_to_date)}")
