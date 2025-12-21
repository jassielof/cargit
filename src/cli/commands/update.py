"""Update command for cargit."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from cli.core import (
    CargitError,
    build_binary,
    clone_repository,
    ensure_dirs,
    get_current_commit,
    get_repo_path,
    install_binary,
    update_repository,
    fetch_repo_silent,
    check_update_available,
    reset_to_remote,
)
from cli.storage import (
    load_metadata,
    save_binary_metadata,
    get_binary_metadata,
    reset_cache_flags,
)


def update(
    alias: str | None = typer.Argument(
        None, help="Alias of binary to update (omit for --all)"
    ),
    all: bool = typer.Option(
        False,
        "--all",
        help="Update all installed binaries to its latest branch commit (HEAD), skipping pinned commits/tags",
    ),
    branch: str | None = typer.Option(
        None,
        "--branch",
        help="Switch to given branch and update to its latest commit (HEAD)",
    ),
    commit: str | None = typer.Option(
        None,
        "--commit",
        help="Update and pin to specific commit hash from current branch unless --branch is also specified",
    ),
    tag: str | None = typer.Option(
        None, "--tag", help="Update and pin to specific tag"
    ),
    check: bool = typer.Option(
        False, "--check", help="Check (fetch) for updates without applying them"
    ),
    jobs: int = typer.Option(
        8, "--jobs", "-j", help="Number of parallel git fetch operations (default: 8)"
    ),
):
    """
    Update installed binaries
    """
    try:
        ensure_dirs()
        metadata = load_metadata()

        if not metadata["installed"]:
            rprint("[yellow]No binaries installed[/yellow]")
            return

        _validate_update_args(alias, all, branch, commit, tag, check)
        targets = _collect_update_targets(metadata, alias, all)

        if all and check and len(targets) > 1:
            _parallel_check_updates(targets, jobs)
            return

        if all and len(targets) > 1:
            rprint(f"[blue]Updating {len(targets)} binaries...[/blue]")

        for idx, binary_alias in enumerate(targets, 1):
            _process_single_update(
                binary_alias,
                branch,
                commit,
                tag,
                check,
                all,
                idx,
                len(targets),
            )

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _validate_update_args(
    alias: str | None, all: bool, branch: str | None, commit: str | None, tag: str | None, check: bool
):
    if all and alias:
        rprint("[red]Error: Cannot specify both <alias> and --all[/red]")
        sys.exit(1)

    if not all and not alias:
        rprint("[red]Error: Must specify <alias> or use --all[/red]")
        sys.exit(1)

    if all and (branch or commit or tag) and not check:
        rprint("[red]Error: Cannot specify --branch, --commit, or --tag with --all[/red]")
        sys.exit(1)

    if sum([bool(commit), bool(tag)]) > 1:
        rprint("[red]Error: Can only specify one of --commit or --tag[/red]")
        sys.exit(1)


def _collect_update_targets(metadata: dict, alias: str | None, all_mode: bool) -> list[str]:
    if all_mode:
        return list(metadata["installed"].keys())

    if alias not in metadata["installed"]:
        rprint(f"[red]Error: Binary '{alias}' not found[/red]")
        sys.exit(1)
    return [alias]


def _process_single_update(
    binary_alias: str,
    branch: str | None,
    commit: str | None,
    tag: str | None,
    check: bool,
    all_mode: bool,
    idx: int,
    total: int,
):
    info = get_binary_metadata(binary_alias)
    if info is None:
        rprint(f"[red]Error: Binary '{binary_alias}' not found in database[/red]")
        return

    stored_branch = info.get("branch", "")
    _print_update_header(all_mode, binary_alias, idx, total)

    repo_path = get_repo_path(info["repo_url"])

    if _handle_recovery_paths(binary_alias, info, repo_path, stored_branch, check):
        return

    target_branch, target_commit, target_tag = _resolve_update_target(
        stored_branch, branch, commit, tag, check, binary_alias, all_mode
    )

    if target_branch is None and target_commit is None and target_tag is None:
        return

    if check:
        _perform_update_check(
            binary_alias, repo_path, target_branch, target_commit, target_tag, info
        )
        return

    _perform_update_apply(
        binary_alias,
        repo_path,
        target_branch,
        target_commit,
        target_tag,
        info,
    )


def _print_update_header(all_mode: bool, binary_alias: str, idx: int, total: int):
    if all_mode and total > 1:
        rprint(f"[blue][{idx}/{total}] Checking {binary_alias}...[/blue]")
    else:
        rprint(f"[blue]Checking {binary_alias}...[/blue]")


def _handle_recovery_paths(
    binary_alias: str,
    info: dict,
    repo_path: Path,
    stored_branch: str,
    check: bool,
) -> bool:
    if info.get("repo_deleted", False) and not repo_path.exists():
        if check:
            rprint(
                f"[yellow]{binary_alias}: Repository was deleted. Run update without --check to reclone.[/yellow]"
            )
            return True
        rprint(f"[yellow]{binary_alias}: Recloning deleted repository...[/yellow]")

    if info.get("artifacts_cleaned", False) and repo_path.exists():
        if check:
            rprint(
                f"[yellow]{binary_alias}: Build artifacts were cleaned. Run update without --check to rebuild.[/yellow]"
            )
            return True

        rprint(f"[yellow]{binary_alias}: Rebuilding cleaned artifacts...[/yellow]")
        try:
            binary_path, _ = build_binary(repo_path, info.get("crate"), binary_alias)
            install_binary(binary_path, binary_alias, Path(info["install_dir"]))
            reset_cache_flags(binary_alias)
            save_binary_metadata(
                alias=binary_alias,
                repo_url=info["repo_url"],
                branch=stored_branch,
                commit=get_current_commit(repo_path),
                install_dir=info["install_dir"],
                bin_path=str(binary_path),
                crate=info.get("crate"),
            )
            rprint(f"[green]{binary_alias}: Rebuilt successfully[/green]")
        except Exception as e:
            rprint(f"[red]{binary_alias}: Failed to rebuild: {e}[/red]")
        return True

    if repo_path.exists():
        return False

    if check:
        rprint(
            f"[red]Repository missing for {binary_alias}. Run update without --check to reinstall.[/red]"
        )
        return True

    rprint(f"[yellow]Repository missing for {binary_alias}, reinstalling...[/yellow]")
    reinstall_branch = stored_branch if not stored_branch.startswith(("commit:", "tag:")) else None
    repo_path, new_branch = clone_repository(info["repo_url"], reinstall_branch)
    binary_path, _ = build_binary(repo_path, info.get("crate"), binary_alias)
    install_binary(binary_path, binary_alias, Path(info["install_dir"]))
    save_binary_metadata(
        alias=binary_alias,
        repo_url=info["repo_url"],
        branch=new_branch,
        commit=get_current_commit(repo_path),
        install_dir=info["install_dir"],
        bin_path=str(binary_path),
        crate=info.get("crate"),
    )
    return True


def _resolve_update_target(
    stored_branch: str,
    branch: str | None,
    commit: str | None,
    tag: str | None,
    check: bool,
    binary_alias: str,
    all_mode: bool,
) -> tuple[str | None, str | None, str | None]:
    explicit_target = branch or commit or tag
    if explicit_target:
        return branch, commit, tag

    if stored_branch.startswith("commit:"):
        if check:
            rprint(
                f"[yellow]{binary_alias} is pinned to a specific commit (skipping check)[/yellow]"
            )
        else:
            rprint(
                f"[yellow]{binary_alias} is pinned to a specific commit[/yellow]\n[yellow]Use explicit --branch, --commit, or --tag to change[/yellow]"
            )
        return None, None, None

    if stored_branch.startswith("tag:"):
        if check:
            rprint(
                f"[yellow]{binary_alias} is pinned to a specific tag (skipping check)[/yellow]"
            )
        else:
            rprint(
                f"[yellow]{binary_alias} is pinned to a specific tag[/yellow]\n[yellow]Use explicit --branch, --commit, or --tag to change[/yellow]"
            )
        return None, None, None

    if all_mode and not check and stored_branch.startswith(("commit:", "tag:")):
        rprint(
            f"[yellow]Skipping {binary_alias} (pinned to specific commit/tag)[/yellow]"
        )
        return None, None, None

    return stored_branch, None, None


def _perform_update_check(
    binary_alias: str,
    repo_path: Path,
    target_branch: str | None,
    target_commit: str | None,
    target_tag: str | None,
    info: dict,
):
    from cli.core import check_for_updates

    has_update, remote_commit = check_for_updates(
        repo_path, target_branch, target_commit, target_tag
    )

    if has_update:
        current_commit = info.get("commit", "unknown")[:8]
        rprint(
            f"[yellow]Update available for {binary_alias}:[/yellow] {current_commit} -> {remote_commit[:8]}"
        )
    else:
        rprint(f"[green]{binary_alias} is up to date[/green]")


def _perform_update_apply(
    binary_alias: str,
    repo_path: Path,
    target_branch: str | None,
    target_commit: str | None,
    target_tag: str | None,
    info: dict,
):
    new_branch, updated = update_repository(
        repo_path,
        target_branch,
        target_commit,
        target_tag,
    )

    if not updated:
        rprint(f"[green]{binary_alias} is up to date[/green]")
        return

    rprint(f"[blue]Rebuilding {binary_alias}...[/blue]")
    binary_path, _ = build_binary(repo_path, info.get("crate"), binary_alias)
    install_binary(binary_path, binary_alias, Path(info["install_dir"]))

    save_binary_metadata(
        alias=binary_alias,
        repo_url=info["repo_url"],
        branch=new_branch,
        commit=get_current_commit(repo_path),
        install_dir=info["install_dir"],
        bin_path=str(binary_path),
        crate=info.get("crate"),
    )

    rprint(f"[green]Updated {binary_alias}![/green]")


def _parallel_check_updates(targets: list[str], jobs: int):
    """Check for updates in parallel (fetch + compare commits)."""

    check_items = _collect_check_items(targets)
    if not check_items:
        rprint("[yellow]No binaries to check[/yellow]")
        return

    rprint(f"[blue]Checking {len(check_items)} binaries for updates...[/blue]")

    fetch_results = _fetch_check_items(check_items, jobs)
    updates_available, up_to_date = _compare_commits(check_items, fetch_results)
    _print_parallel_check_results(updates_available, up_to_date)


def _collect_check_items(targets: list[str]) -> list[tuple[str, Path, str, str]]:
    check_items: list[tuple[str, Path, str, str]] = []
    for binary_alias in targets:
        info = get_binary_metadata(binary_alias)
        if info is None:
            rprint(f"[yellow]Skipping {binary_alias}: not found in database[/yellow]")
            continue

        stored_branch = info.get("branch", "")
        if stored_branch.startswith(("commit:", "tag:")):
            rprint(f"[yellow]Skipping {binary_alias}: pinned to {stored_branch}[/yellow]")
            continue

        repo_path = get_repo_path(info["repo_url"])
        if not repo_path.exists():
            rprint(
                f"[yellow]{binary_alias}: repository missing, run 'cargit update {binary_alias}' to reinstall[/yellow]"
            )
            continue

        check_items.append(
            (binary_alias, repo_path, stored_branch, info.get("commit", ""))
        )

    return check_items


def _fetch_check_items(check_items: list[tuple[str, Path, str, str]], jobs: int) -> dict[str, bool]:
    fetch_results: dict[str, bool] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Fetching repositories...", total=len(check_items))

        def fetch_one(item: tuple[str, Path, str, str]) -> tuple[str, bool, str | None]:
            alias, repo_path, branch, _ = item
            success, error = fetch_repo_silent(repo_path, branch or None)
            return alias, success, error

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(fetch_one, item): item[0] for item in check_items}

            for future in as_completed(futures):
                alias, success, error = future.result()
                fetch_results[alias] = success
                if not success:
                    progress.console.print(f"[yellow]{alias}: fetch failed - {error}[/yellow]")
                progress.advance(task)

    return fetch_results


def _compare_commits(
    check_items: list[tuple[str, Path, str, str]], fetch_results: dict[str, bool]
) -> tuple[list[tuple[str, str, str]], list[str]]:
    updates_available: list[tuple[str, str, str]] = []
    up_to_date: list[str] = []

    for alias, repo_path, branch, current_commit in check_items:
        if not fetch_results.get(alias, False):
            continue
        try:
            has_update, _, remote = check_update_available(repo_path, branch or None)
            if has_update:
                updates_available.append((alias, current_commit[:8], remote[:8]))
            else:
                up_to_date.append(alias)
        except Exception as e:
            rprint(f"[yellow]{alias}: could not check - {e}[/yellow]")

    return updates_available, up_to_date


def _print_parallel_check_results(
    updates_available: list[tuple[str, str, str]], up_to_date: list[str]
):
    if updates_available:
        rprint(f"\n[yellow]Updates available ({len(updates_available)}):[/yellow]")
        for alias, old, new in updates_available:
            rprint(f"  [cyan]{alias}[/cyan]: {old} → {new}")

    if up_to_date:
        rprint(f"\n[green]Up to date ({len(up_to_date)}):[/green] {', '.join(up_to_date)}")

    if updates_available:
        rprint("\n[blue]Run 'cargit sync' to fetch, reset, and rebuild all.[/blue]")
