"""CLI commands for cargit.

This module provides the command-line interface for cargit,
including install, update, sync, clean, status, and config commands.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from cargit.core import (
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
    BIN_DIR,
)
from cargit.storage import (
    load_metadata,
    save_binary_metadata,
    remove_binary_metadata,
    get_binary_metadata,
)
from cargit.utils import display_installed_table

app = typer.Typer(
    help="Git-based cargo binary installer with cached repositories for faster updates",
    no_args_is_help=True,
)


@app.command(no_args_is_help=True)
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
    from cargit.storage import get_binaries_by_repo

    try:
        ensure_dirs()

        # Check if we already have binaries from this repo
        existing = get_binaries_by_repo(git_url)
        if existing:
            existing_aliases = [b["alias"] for b in existing]
            rprint(f"[yellow]Note: Repository already has installed binaries: {', '.join(existing_aliases)}[/yellow]")
            rprint("[yellow]They share the same repository and will be at the same commit.[/yellow]")

        # Clone or update repository
        repo_path, actual_branch = clone_repository(git_url, branch)

        # Determine binary name/alias
        if alias is None:
            from .core import get_binary_name_from_cargo

            alias = get_binary_name_from_cargo(repo_path, crate)

        # Check if alias already exists
        existing_binary = get_binary_metadata(alias)
        if existing_binary is not None:
            rprint(f"[red]Error: Binary '{alias}' already exists[/red]")
            rprint(f"[dim]Use --alias to specify a different name, or 'cargit remove {alias}' first[/dim]")
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


@app.command(no_args_is_help=True)
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

        # Validate arguments
        if all and alias:
            rprint("[red]Error: Cannot specify both <alias> and --all[/red]")
            sys.exit(1)

        if not all and not alias:
            rprint("[red]Error: Must specify <alias> or use --all[/red]")
            sys.exit(1)

        # Cannot specify branch/commit/tag with --all (unless checking)
        if all and (branch or commit or tag) and not check:
            rprint(
                "[red]Error: Cannot specify --branch, --commit, or --tag with --all[/red]"
            )
            sys.exit(1)

        # Cannot specify multiple targets
        target_count = sum([bool(commit), bool(tag)])
        if target_count > 1:
            rprint("[red]Error: Can only specify one of --commit or --tag[/red]")
            sys.exit(1)

        targets = []
        if all:
            targets = list(metadata["installed"].keys())
        elif alias:
            if alias not in metadata["installed"]:
                rprint(f"[red]Error: Binary '{alias}' not found[/red]")
                sys.exit(1)
            targets = [alias]

        # Parallel check mode for --all --check
        if all and check and len(targets) > 1:
            _parallel_check_updates(targets, jobs)
            return

        # Show progress for batch updates
        if all and len(targets) > 1:
            rprint(f"[blue]Updating {len(targets)} binaries...[/blue]")

        for idx, binary_alias in enumerate(targets, 1):
            info = get_binary_metadata(binary_alias)
            if info is None:
                rprint(f"[red]Error: Binary '{binary_alias}' not found in database[/red]")
                continue

            stored_branch = info.get("branch", "")

            # Skip pinned commits/tags when using --all
            if all and not check:
                if stored_branch.startswith("commit:") or stored_branch.startswith(
                    "tag:"
                ):
                    rprint(
                        f"[yellow]Skipping {binary_alias} (pinned to specific commit/tag)[/yellow]"
                    )
                    continue

            # Show progress for batch updates
            if all and len(targets) > 1:
                rprint(f"[blue][{idx}/{len(targets)}] Checking {binary_alias}...[/blue]")
            else:
                rprint(f"[blue]Checking {binary_alias}...[/blue]")

            repo_path = get_repo_path(info["repo_url"])

            # Check for recovery scenarios
            from cargit.storage import reset_cache_flags

            # Scenario 1: Repo was deleted
            if info.get("repo_deleted", False) and not repo_path.exists():
                if check:
                    rprint(f"[yellow]{binary_alias}: Repository was deleted. Run update without --check to reclone.[/yellow]")
                    continue

                rprint(f"[yellow]{binary_alias}: Recloning deleted repository...[/yellow]")
                # Will be handled by the "not repo_path.exists()" block below

            # Scenario 2: Artifacts were cleaned
            elif info.get("artifacts_cleaned", False) and repo_path.exists():
                if check:
                    rprint(f"[yellow]{binary_alias}: Build artifacts were cleaned. Run update without --check to rebuild.[/yellow]")
                    continue

                rprint(f"[yellow]{binary_alias}: Rebuilding cleaned artifacts...[/yellow]")
                try:
                    binary_path, _ = build_binary(repo_path, info.get("crate"), binary_alias)
                    install_binary(binary_path, binary_alias, Path(info["install_dir"]))

                    # Reset cache flags and update metadata
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
                    continue
                except Exception as e:
                    rprint(f"[red]{binary_alias}: Failed to rebuild: {e}[/red]")
                    continue

            if not repo_path.exists():
                if check:
                    rprint(
                        f"[red]Repository missing for {binary_alias}. Run update without --check to reinstall.[/red]"
                    )
                    continue

                rprint(
                    f"[yellow]Repository missing for {binary_alias}, reinstalling...[/yellow]"
                )
                # Extract branch from stored info if it's a regular branch
                reinstall_branch = None

                if not stored_branch.startswith(
                    "commit:"
                ) and not stored_branch.startswith("tag:"):
                    reinstall_branch = stored_branch

                # Reinstall
                repo_path, new_branch = clone_repository(
                    info["repo_url"], reinstall_branch
                )
                binary_path, _ = build_binary(repo_path, info.get("crate"), binary_alias)
                install_binary(binary_path, binary_alias, Path(info["install_dir"]))

                # Update metadata
                save_binary_metadata(
                    alias=binary_alias,
                    repo_url=info["repo_url"],
                    branch=new_branch,
                    commit=get_current_commit(repo_path),
                    install_dir=info["install_dir"],
                    bin_path=str(binary_path),
                    crate=info.get("crate"),
                )
            else:
                # Determine target (branch, commit, or tag)
                if tag or commit or branch:
                    # User explicitly specified what to update to
                    target_branch = branch
                    target_commit = commit
                    target_tag = tag
                else:
                    # No explicit target, use stored info
                    # Parse stored branch info
                    if stored_branch.startswith("commit:"):
                        # Pinned to commit - don't update unless explicit
                        if not check:
                            rprint(
                                f"[yellow]{binary_alias} is pinned to a specific commit[/yellow]"
                            )
                            rprint(
                                "[yellow]Use explicit --branch, --commit, or --tag to change[/yellow]"
                            )
                            continue
                        else:
                            # For check mode, skip pinned commits
                            rprint(
                                f"[yellow]{binary_alias} is pinned to a specific commit (skipping check)[/yellow]"
                            )
                            continue
                    elif stored_branch.startswith("tag:"):
                        # Pinned to tag - don't update unless explicit
                        if not check:
                            rprint(
                                f"[yellow]{binary_alias} is pinned to a specific tag[/yellow]"
                            )
                            rprint(
                                "[yellow]Use explicit --branch, --commit, or --tag to change[/yellow]"
                            )
                            continue
                        else:
                            # For check mode, skip pinned tags
                            rprint(
                                f"[yellow]{binary_alias} is pinned to a specific tag (skipping check)[/yellow]"
                            )
                            continue
                    else:
                        # Regular branch - update to latest
                        target_branch = stored_branch
                        target_commit = None
                        target_tag = None

                if check:
                    # Check mode: fetch and compare commits
                    from cargit.core import check_for_updates

                    has_update, remote_commit = check_for_updates(
                        repo_path, target_branch, target_commit, target_tag
                    )

                    if has_update:
                        current_commit = info.get("commit", "unknown")[:8]
                        rprint(
                            f"[yellow]Update available for {binary_alias}:[/yellow] "
                            f"{current_commit} -> {remote_commit[:8]}"
                        )
                    else:
                        rprint(f"[green]{binary_alias} is up to date[/green]")
                else:
                    # Update repository
                    new_branch, updated = update_repository(
                        repo_path,
                        target_branch,
                        target_commit,
                        target_tag,
                    )

                    if updated:
                        rprint(f"[blue]Rebuilding {binary_alias}...[/blue]")

                        # Rebuild and reinstall
                        binary_path, _ = build_binary(repo_path, info.get("crate"), binary_alias)
                        install_binary(
                            binary_path, binary_alias, Path(info["install_dir"])
                        )

                        # Update metadata
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
                    else:
                        rprint(f"[green]{binary_alias} is up to date[/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _parallel_check_updates(targets: list[str], jobs: int):
    """Check for updates in parallel (fetch + compare commits)."""

    # Collect info for all targets
    check_items: list[tuple[str, Path, str]] = []  # (alias, repo_path, branch)

    for binary_alias in targets:
        info = get_binary_metadata(binary_alias)
        if info is None:
            rprint(f"[yellow]Skipping {binary_alias}: not found in database[/yellow]")
            continue

        stored_branch = info.get("branch", "")

        # Skip pinned commits/tags
        if stored_branch.startswith("commit:") or stored_branch.startswith("tag:"):
            rprint(f"[yellow]Skipping {binary_alias}: pinned to {stored_branch}[/yellow]")
            continue

        repo_path = get_repo_path(info["repo_url"])
        if not repo_path.exists():
            rprint(f"[yellow]{binary_alias}: repository missing, run 'cargit update {binary_alias}' to reinstall[/yellow]")
            continue

        check_items.append((binary_alias, repo_path, stored_branch, info.get("commit", "")))

    if not check_items:
        rprint("[yellow]No binaries to check[/yellow]")
        return

    rprint(f"[blue]Checking {len(check_items)} binaries for updates...[/blue]")

    # Phase 1: Parallel fetch
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

    # Phase 2: Check for updates (sequential, fast)
    updates_available = []
    up_to_date = []

    for alias, repo_path, branch, current_commit in check_items:
        if not fetch_results.get(alias, False):
            continue

        try:
            has_update, local, remote = check_update_available(repo_path, branch or None)
            if has_update:
                updates_available.append((alias, current_commit[:8], remote[:8]))
            else:
                up_to_date.append(alias)
        except Exception as e:
            rprint(f"[yellow]{alias}: could not check - {e}[/yellow]")

    # Print results
    if updates_available:
        rprint(f"\n[yellow]Updates available ({len(updates_available)}):[/yellow]")
        for alias, old, new in updates_available:
            rprint(f"  [cyan]{alias}[/cyan]: {old} → {new}")

    if up_to_date:
        rprint(f"\n[green]Up to date ({len(up_to_date)}):[/green] {', '.join(up_to_date)}")

    if updates_available:
        rprint("\n[blue]Run 'cargit sync' to fetch, reset, and rebuild all.[/blue]")


@app.command("list")
def list_binaries():
    """List installed binaries"""
    metadata = load_metadata()

    if not metadata["installed"]:
        rprint("[yellow]No binaries installed[/yellow]")
        return

    display_installed_table(metadata["installed"])


@app.command()
def status():
    """Show cache status and statistics.

    Displays:
    - Number of installed binaries
    - Cache size (repos + build artifacts)
    - Build statistics
    - Binaries needing attention (missing repos, cleaned artifacts)

    Examples:
        cargit status
    """
    from cargit.core import (
        CACHE_DIR,
        DATA_DIR,
        get_cache_size,
        format_size,
        get_repo_path,
    )
    from cargit.storage import get_status_summary, get_all_binaries_full
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console

    console = Console()

    ensure_dirs()

    # Get summary stats
    summary = get_status_summary()
    binaries = get_all_binaries_full()

    # Calculate cache sizes
    total_cache_size = get_cache_size(CACHE_DIR) if CACHE_DIR.exists() else 0
    bin_dir_size = get_cache_size(DATA_DIR) if DATA_DIR.exists() else 0

    # Calculate per-repo sizes
    repo_sizes: dict[str, int] = {}
    for binary in binaries:
        repo_path = get_repo_path(binary["repo_url"])
        if repo_path.exists() and binary["repo_url"] not in repo_sizes:
            repo_sizes[binary["repo_url"]] = get_cache_size(repo_path)

    # Header
    rprint("\n[bold cyan]Cargit Status[/bold cyan]")
    rprint(f"{'─' * 50}\n")

    # Summary panel
    summary_text = f"""[cyan]Installed binaries:[/cyan] {summary['total_binaries']}
[cyan]Unique repositories:[/cyan] {summary['unique_repos']}
[cyan]Total cache size:[/cyan] {format_size(total_cache_size)}
[cyan]Binary links size:[/cyan] {format_size(bin_dir_size)}"""

    if summary['total_builds'] > 0:
        avg_time = summary['overall_avg_build_time']
        if avg_time:
            if avg_time >= 60:
                avg_str = f"{int(avg_time // 60)}m {int(avg_time % 60)}s"
            else:
                avg_str = f"{avg_time:.1f}s"
            summary_text += f"\n[cyan]Total builds:[/cyan] {summary['total_builds']}"
            summary_text += f"\n[cyan]Avg build time:[/cyan] {avg_str}"

    console.print(Panel(summary_text, title="Summary", border_style="blue"))

    # Attention needed section
    needs_attention = []
    if summary['repos_deleted'] > 0:
        needs_attention.append(f"[yellow]{summary['repos_deleted']} binary(ies) with deleted repos[/yellow]")
    if summary['artifacts_cleaned'] > 0:
        needs_attention.append(f"[yellow]{summary['artifacts_cleaned']} binary(ies) with cleaned artifacts[/yellow]")

    if needs_attention:
        rprint("\n[bold yellow]⚠ Attention Needed[/bold yellow]")
        for item in needs_attention:
            rprint(f"  • {item}")
        rprint("[dim]Run 'cargit sync' to rebuild affected binaries[/dim]")

    # Binaries table with sizes
    if binaries:
        rprint("\n[bold]Installed Binaries[/bold]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Alias", style="cyan")
        table.add_column("Branch/Ref", style="green")
        table.add_column("Repo Size", justify="right")
        table.add_column("Last Build", justify="right")
        table.add_column("Status")

        for binary in binaries:
            # Branch display
            branch = binary["branch"]
            if branch.startswith("commit:"):
                parts = branch.split(":")
                if len(parts) >= 3:
                    branch = f"📌 {parts[1]}@{parts[2]}"
            elif branch.startswith("tag:"):
                branch = f"🏷️  {branch.split(':', 1)[1]}"

            # Size
            repo_size = repo_sizes.get(binary["repo_url"], 0)
            size_str = format_size(repo_size) if repo_size > 0 else "[dim]N/A[/dim]"

            # Build time
            if binary.get("last_build_duration"):
                duration = binary["last_build_duration"]
                if duration >= 60:
                    build_str = f"{int(duration // 60)}m {int(duration % 60)}s"
                else:
                    build_str = f"{duration:.1f}s"
            else:
                build_str = "[dim]—[/dim]"

            # Status
            status_parts = []
            if binary.get("repo_deleted"):
                status_parts.append("[red]repo deleted[/red]")
            elif binary.get("artifacts_cleaned"):
                status_parts.append("[yellow]needs rebuild[/yellow]")
            else:
                status_parts.append("[green]✓[/green]")

            status_str = ", ".join(status_parts) if status_parts else "[green]✓[/green]"

            table.add_row(
                binary["alias"],
                branch,
                size_str,
                build_str,
                status_str,
            )

        console.print(table)

    # Paths info
    rprint(f"\n[dim]Cache: {CACHE_DIR}[/dim]")
    rprint(f"[dim]Binaries: {DATA_DIR}[/dim]")


@app.command()
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
    from cargit.core import (
        get_repo_path,
        _expand_workspace_members,
        _get_available_crates_from_members,
    )
    from cargit.storage import get_binaries_by_repo
    import tempfile
    import shutil

    try:
        ensure_dirs()

        # Check if we already have this repo
        repo_path = get_repo_path(git_url)
        temp_clone = False

        if not repo_path.exists():
            # Clone to temp directory for inspection
            rprint("[blue]Fetching repository information...[/blue]")
            temp_dir = tempfile.mkdtemp(prefix="cargit_info_")
            repo_path = Path(temp_dir)
            temp_clone = True

            from cargit.core import run_command
            run_command([
                "git", "clone", "--depth=1", "--single-branch",
                git_url, str(repo_path)
            ])

        try:
            cargo_toml = repo_path / "Cargo.toml"
            if not cargo_toml.exists():
                rprint("[red]Error: No Cargo.toml found in repository[/red]")
                sys.exit(1)

            # Parse Cargo.toml
            try:
                import tomllib
                with open(cargo_toml, "rb") as f:
                    cargo_data = tomllib.load(f)
            except ImportError:
                import tomlkit
                with open(cargo_toml, "r", encoding="utf-8") as f:
                    cargo_data = tomlkit.load(f)

            rprint(f"\n[bold cyan]Repository: {git_url}[/bold cyan]")
            rprint(f"{'─' * 60}\n")

            # Check if this is a workspace
            if "workspace" in cargo_data:
                rprint("[bold]Type:[/bold] Cargo Workspace")

                # Get default members
                default_members = cargo_data["workspace"].get("default-members", [])
                if default_members:
                    rprint(f"[bold]Default member:[/bold] {default_members[0]}")

                # Expand and list all members
                members = cargo_data["workspace"].get("members", [])
                expanded = _expand_workspace_members(repo_path, members)
                crates = _get_available_crates_from_members(repo_path, expanded)

                rprint(f"\n[bold]Available crates ({len(crates)}):[/bold]")
                for crate in sorted(crates):
                    is_default = any(crate in m for m in default_members)
                    marker = " [green](default)[/green]" if is_default else ""
                    rprint(f"  • {crate}{marker}")

                rprint(f"\n[dim]Install with: cargit install {git_url} <crate_name>[/dim]")

            else:
                rprint("[bold]Type:[/bold] Single Crate")

                # Get package info
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

            # Check for existing installations
            existing = get_binaries_by_repo(git_url)
            if existing:
                rprint("\n[bold yellow]Already installed from this repo:[/bold yellow]")
                for binary in existing:
                    crate_info = f" (crate: {binary['crate']})" if binary.get("crate") else ""
                    rprint(f"  • {binary['alias']}{crate_info} @ {binary['commit'][:8]}")

        finally:
            if temp_clone:
                shutil.rmtree(repo_path, ignore_errors=True)

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    init: bool = typer.Option(False, "--init", help="Initialize config file with defaults"),
    edit: bool = typer.Option(False, "--edit", help="Open config file in editor"),
):
    """Manage cargit configuration.

    Examples:
        cargit config --show     # Show current configuration
        cargit config --init     # Create config file with defaults
        cargit config --edit     # Open config in $EDITOR
    """
    from cargit.config import load_config, init_config, CONFIG_FILE
    import os

    if not any([show, init, edit]):
        rprint("[red]Error: Must specify one of --show, --init, or --edit[/red]")
        sys.exit(1)

    if show:
        config_data = load_config()
        rprint("[bold cyan]Current Configuration[/bold cyan]")
        rprint(f"[dim]Config file: {CONFIG_FILE}[/dim]")
        rprint(f"[dim]{'─' * 50}[/dim]\n")

        for section, values in config_data.items():
            rprint(f"[bold yellow]\\[{section}][/bold yellow]")
            if isinstance(values, dict):
                for key, value in values.items():
                    rprint(f"  [cyan]{key}[/cyan] = {value}")
            else:
                rprint(f"  {values}")
            rprint()

    elif init:
        config_path = init_config()
        rprint(f"[green]Configuration file created at {config_path}[/green]")

    elif edit:
        if not CONFIG_FILE.exists():
            init_config()
            rprint(f"[blue]Created default config at {CONFIG_FILE}[/blue]")

        editor = os.environ.get("EDITOR", "nano")
        os.system(f"{editor} {CONFIG_FILE}")


@app.command(no_args_is_help=True)
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


@app.command()
def remove(alias: str = typer.Argument(..., help="Alias of binary to remove")):
    """Remove installed binary

    Examples:
        cargit remove typst
        cargit remove my-tool
    """
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

    symlink_path = Path(info["install_dir"]) / alias

    if symlink_path.exists():
        rprint(str(symlink_path.resolve()))
    else:
        rprint(f"[red]Error: Binary path does not exist: {symlink_path}[/red]")
        sys.exit(1)


@app.command()
def clean(
    alias: str | None = typer.Argument(None, help="Alias of binary to clean (omit for --all or --orphaned)"),
    all: bool = typer.Option(False, "--all", help="Remove all cached repositories and build artifacts"),
    artifacts: bool = typer.Option(False, "--artifacts", help="Clean build artifacts for specific binary"),
    repos: bool = typer.Option(False, "--repos", help="Delete repository for specific binary"),
    orphaned: bool = typer.Option(False, "--orphaned", help="Remove orphaned repositories not tracked in database"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
):
    """Clean cache to free disk space

    Examples:
        cargit clean --all               # Remove all caches (with confirmation)
        cargit clean --all --dry-run     # Preview what would be deleted
        cargit clean fd --artifacts      # Clean build artifacts for fd
        cargit clean fd --repos          # Delete repo for fd
        cargit clean --orphaned          # Remove untracked repositories
    """
    from cargit.core import (
        CACHE_DIR,
        get_repo_path,
        get_cache_size,
        format_size,
        clean_all_cache,
        clean_binary_artifacts,
        clean_binary_repo,
        copy_binary_to_safe_location,
        find_orphaned_repos,
    )
    from cargit.storage import mark_artifacts_cleaned, mark_repo_deleted

    try:
        ensure_dirs()

        # Validate arguments
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

        # Mode 1: Clean all cache
        if all:
            cache_size = get_cache_size(CACHE_DIR) if CACHE_DIR.exists() else 0

            if dry_run:
                rprint(f"[yellow]Would remove all cache ({format_size(cache_size)})[/yellow]")
                rprint(f"[yellow]Cache directory: {CACHE_DIR}[/yellow]")
                return

            if cache_size == 0:
                rprint("[yellow]Cache is already empty[/yellow]")
                return

            # Ask for confirmation
            rprint(f"[yellow]This will remove all cached repositories and build artifacts ({format_size(cache_size)})[/yellow]")
            rprint(f"[yellow]Cache directory: {CACHE_DIR}[/yellow]")
            confirm = typer.confirm("Are you sure?")

            if not confirm:
                rprint("[blue]Aborted[/blue]")
                return

            if clean_all_cache():
                rprint(f"[green]Successfully cleaned cache (freed {format_size(cache_size)})[/green]")
            else:
                rprint("[red]Failed to clean cache[/red]")
                sys.exit(1)

        # Mode 2: Clean specific binary artifacts
        elif artifacts:
            info = get_binary_metadata(alias)
            if info is None:
                rprint(f"[red]Error: Binary '{alias}' not found[/red]")
                sys.exit(1)

            repo_path = get_repo_path(info["repo_url"])
            target_dir = repo_path / "target"

            if not repo_path.exists():
                rprint(f"[yellow]Repository not found at {repo_path}[/yellow]")
                return

            artifacts_size = get_cache_size(target_dir) if target_dir.exists() else 0

            if dry_run:
                rprint(f"[yellow]Would clean build artifacts for {alias} ({format_size(artifacts_size)})[/yellow]")
                rprint("[yellow]Would copy binary to safe location[/yellow]")
                return

            if artifacts_size == 0:
                rprint(f"[yellow]No build artifacts to clean for {alias}[/yellow]")
                return

            # Copy binary to safe location before cleaning
            binary_path = Path(info["bin_path"])
            if not binary_path.exists():
                rprint(f"[red]Error: Binary not found at {binary_path}[/red]")
                sys.exit(1)

            rprint("[blue]Copying binary to safe location...[/blue]")
            safe_path = copy_binary_to_safe_location(binary_path, alias)

            # Clean artifacts
            rprint(f"[blue]Cleaning build artifacts ({format_size(artifacts_size)})...[/blue]")
            if clean_binary_artifacts(repo_path):
                # Update symlink to point to copied binary
                from cargit.core import install_binary
                install_binary(safe_path, alias, Path(info["install_dir"]), binary_type="copy")

                # Mark in database
                mark_artifacts_cleaned(alias, str(safe_path))

                rprint(f"[green]Successfully cleaned artifacts for {alias} (freed {format_size(artifacts_size)})[/green]")
                rprint(f"[yellow]Binary is now using copied version. Run 'cargit update {alias}' to rebuild and restore symlink.[/yellow]")
            else:
                rprint(f"[red]Failed to clean artifacts for {alias}[/red]")
                sys.exit(1)

        # Mode 3: Clean specific binary repo
        elif repos:
            info = get_binary_metadata(alias)
            if info is None:
                rprint(f"[red]Error: Binary '{alias}' not found[/red]")
                sys.exit(1)

            repo_path = get_repo_path(info["repo_url"])

            if not repo_path.exists():
                rprint(f"[yellow]Repository not found at {repo_path}[/yellow]")
                return

            repo_size = get_cache_size(repo_path)

            if dry_run:
                rprint(f"[yellow]Would delete repository for {alias} ({format_size(repo_size)})[/yellow]")
                rprint("[yellow]Would copy binary to safe location[/yellow]")
                return

            # Copy binary to safe location before deleting repo
            binary_path = Path(info["bin_path"])
            if not binary_path.exists():
                rprint(f"[red]Error: Binary not found at {binary_path}[/red]")
                sys.exit(1)

            rprint("[blue]Copying binary to safe location...[/blue]")
            safe_path = copy_binary_to_safe_location(binary_path, alias)

            # Delete repo
            rprint(f"[blue]Deleting repository ({format_size(repo_size)})...[/blue]")
            if clean_binary_repo(repo_path):
                # Update symlink to point to copied binary
                from cargit.core import install_binary
                install_binary(safe_path, alias, Path(info["install_dir"]), binary_type="copy")

                # Mark in database
                mark_repo_deleted(alias)
                mark_artifacts_cleaned(alias, str(safe_path))  # Also mark artifacts as cleaned

                rprint(f"[green]Successfully deleted repository for {alias} (freed {format_size(repo_size)})[/green]")
                rprint(f"[yellow]Binary is now using copied version. Run 'cargit update {alias}' to reclone and rebuild.[/yellow]")
            else:
                rprint(f"[red]Failed to delete repository for {alias}[/red]")
                sys.exit(1)

        # Mode 4: Clean orphaned repos
        elif orphaned:
            orphaned_repos = find_orphaned_repos()

            if not orphaned_repos:
                rprint("[green]No orphaned repositories found[/green]")
                return

            total_size = sum(size for _, size in orphaned_repos)

            rprint(f"[yellow]Found {len(orphaned_repos)} orphaned repositories ({format_size(total_size)}):[/yellow]")
            for repo_path, size in orphaned_repos:
                rprint(f"  - {repo_path} ({format_size(size)})")

            if dry_run:
                rprint(f"[yellow]Would remove {len(orphaned_repos)} orphaned repositories (total: {format_size(total_size)})[/yellow]")
                return

            # Ask for confirmation
            confirm = typer.confirm(f"Remove {len(orphaned_repos)} orphaned repositories?")

            if not confirm:
                rprint("[blue]Aborted[/blue]")
                return

            # Delete orphaned repos
            import shutil
            removed = 0
            for repo_path, _ in orphaned_repos:
                try:
                    shutil.rmtree(repo_path)
                    removed += 1
                except Exception as e:
                    rprint(f"[yellow]Warning: Could not delete {repo_path}: {e}[/yellow]")

            rprint(f"[green]Successfully removed {removed}/{len(orphaned_repos)} orphaned repositories (freed {format_size(total_size)})[/green]")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def sync(
    jobs: int = typer.Option(
        8, "--jobs", "-j", help="Number of parallel git operations (default: 8)"
    ),
    fetch_only: bool = typer.Option(
        False, "--fetch-only", help="Only fetch repositories, don't reset or build"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without making changes"
    ),
):
    """
    Sync all binaries: parallel fetch & reset, then sequential builds.

    This is the recommended way to update all binaries efficiently:
    - Phase 1: Parallel git fetch (network-bound)
    - Phase 2: Parallel git reset (disk-bound, only repos with updates)
    - Phase 3: Sequential cargo builds (CPU-bound, one at a time)

    Examples:
        cargit sync                  # Full sync: fetch, reset, build
        cargit sync --fetch-only     # Only fetch, don't reset or build
        cargit sync --dry-run        # Preview what would be updated
        cargit sync -j 4             # Use 4 parallel git operations
    """
    from cargit.core import get_default_branch

    try:
        ensure_dirs()
        metadata = load_metadata()

        if not metadata["installed"]:
            rprint("[yellow]No binaries installed[/yellow]")
            return

        # Collect sync candidates (skip pinned)
        sync_items: list[dict] = []

        for binary_alias in metadata["installed"]:
            info = get_binary_metadata(binary_alias)
            if info is None:
                continue

            stored_branch = info.get("branch", "")

            # Skip pinned commits/tags
            if stored_branch.startswith("commit:") or stored_branch.startswith("tag:"):
                rprint(f"[dim]Skipping {binary_alias}: pinned to {stored_branch}[/dim]")
                continue

            repo_path = get_repo_path(info["repo_url"])

            # Handle missing repos
            if not repo_path.exists():
                if info.get("repo_deleted", False):
                    rprint(f"[yellow]{binary_alias}: needs reclone (repo was cleaned)[/yellow]")
                else:
                    rprint(f"[yellow]{binary_alias}: repo missing, run 'cargit update {binary_alias}'[/yellow]")
                continue

            # Handle cleaned artifacts
            if info.get("artifacts_cleaned", False):
                rprint(f"[yellow]{binary_alias}: needs rebuild (artifacts cleaned)[/yellow]")

            sync_items.append({
                "alias": binary_alias,
                "info": info,
                "repo_path": repo_path,
                "branch": stored_branch or None,
            })

        if not sync_items:
            rprint("[yellow]No binaries to sync[/yellow]")
            return

        rprint(f"[blue]Syncing {len(sync_items)} binaries...[/blue]\n")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Parallel fetch
        # ═══════════════════════════════════════════════════════════════════
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

        # Check which repos have updates
        updates_needed: list[dict] = []
        up_to_date: list[str] = []

        for item in sync_items:
            alias = item["alias"]
            if not fetch_results.get(alias, False):
                continue

            has_update, current, remote = check_update_available(item["repo_path"], item["branch"])

            # Also rebuild if artifacts were cleaned
            needs_rebuild = item["info"].get("artifacts_cleaned", False)

            if has_update:
                item["current_commit"] = current
                item["remote_commit"] = remote
                updates_needed.append(item)
            elif needs_rebuild:
                item["current_commit"] = current
                item["remote_commit"] = current
                item["rebuild_only"] = True
                updates_needed.append(item)
            else:
                up_to_date.append(alias)

        rprint(f"  [green]✓ Fetched {sum(fetch_results.values())}/{len(sync_items)} repositories[/green]")

        if up_to_date:
            rprint(f"  [dim]Already up to date: {', '.join(up_to_date)}[/dim]")

        if not updates_needed:
            rprint("\n[green]All binaries are up to date![/green]")
            return

        rprint(f"  [yellow]Updates available: {len(updates_needed)}[/yellow]")

        if dry_run:
            rprint("\n[yellow]Dry run - would update:[/yellow]")
            for item in updates_needed:
                if item.get("rebuild_only"):
                    rprint(f"  [cyan]{item['alias']}[/cyan]: rebuild (artifacts cleaned)")
                else:
                    rprint(f"  [cyan]{item['alias']}[/cyan]: {item['current_commit'][:8]} → {item['remote_commit'][:8]}")
            return

        if fetch_only:
            rprint("\n[blue]Fetch-only mode - skipping reset and build[/blue]")
            rprint("[yellow]Updates available for:[/yellow]")
            for item in updates_needed:
                if item.get("rebuild_only"):
                    rprint(f"  [cyan]{item['alias']}[/cyan]: needs rebuild")
                else:
                    rprint(f"  [cyan]{item['alias']}[/cyan]: {item['current_commit'][:8]} → {item['remote_commit'][:8]}")
            return

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Parallel reset
        # ═══════════════════════════════════════════════════════════════════
        rprint("\n[bold cyan]Phase 2/3: Resetting repositories[/bold cyan]")

        # Only reset repos that have actual git updates (not rebuild-only)
        reset_items = [item for item in updates_needed if not item.get("rebuild_only")]

        if reset_items:
            reset_results: dict[str, bool] = {}

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task("[cyan]Resetting...", total=len(reset_items))

                def reset_one(item: dict) -> tuple[str, bool, str | None]:
                    # Get the branch to reset to
                    branch = item["branch"]
                    if not branch:
                        try:
                            branch = get_default_branch(item["repo_path"])
                        except Exception:
                            return item["alias"], False, "Could not determine branch"

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

            # Filter out failed resets from build list
            updates_needed = [
                item for item in updates_needed
                if item.get("rebuild_only") or reset_results.get(item["alias"], False)
            ]

            rprint(f"  [green]✓ Reset {sum(reset_results.values())}/{len(reset_items)} repositories[/green]")
        else:
            rprint("  [dim]No git resets needed (rebuild only)[/dim]")

        if not updates_needed:
            rprint("\n[yellow]No binaries to build after reset failures[/yellow]")
            return

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Sequential builds
        # ═══════════════════════════════════════════════════════════════════
        rprint(f"\n[bold cyan]Phase 3/3: Building {len(updates_needed)} binaries[/bold cyan]")

        from cargit.storage import reset_cache_flags

        built = 0
        failed = 0

        for idx, item in enumerate(updates_needed, 1):
            alias = item["alias"]
            info = item["info"]

            rprint(f"\n[blue][{idx}/{len(updates_needed)}] Building {alias}...[/blue]")

            try:
                binary_path, _ = build_binary(item["repo_path"], info.get("crate"), alias)
                install_binary(binary_path, alias, Path(info["install_dir"]))

                # Get branch for metadata
                branch = item["branch"]
                if not branch:
                    branch = get_default_branch(item["repo_path"])

                # Reset cache flags if this was a rebuild after cleaning
                if info.get("artifacts_cleaned", False) or info.get("repo_deleted", False):
                    reset_cache_flags(alias)

                # Update metadata
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

        # Summary
        rprint(f"\n[bold]{'═' * 50}[/bold]")
        rprint("[bold green]Sync complete![/bold green]")
        rprint(f"  Built: {built}, Failed: {failed}, Skipped: {len(up_to_date)}")

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)
