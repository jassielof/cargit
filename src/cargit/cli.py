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
    update_repository,
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
    alias: str | None = typer.Option(
        None, "--alias", help="Alias for installed binary"
    ),
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
    alias: str | None = typer.Argument(
        None, help="Alias of binary to update (omit for --all)"
    ),
    all: bool = typer.Option(False, "--all", help="Update all installed binaries"),
    branch: str | None = typer.Option(
        None, "--branch", help="Switch to a different branch"
    ),
    commit: str | None = typer.Option(
        None, "--commit", help="Update to specific commit hash"
    ),
    tag: str | None = typer.Option(None, "--tag", help="Update to specific tag"),
):
    """Update installed binaries

    Update behavior:\n
    - Branch only: Update to latest HEAD of that branch\n
    - Commit only: Pin to that commit (on current branch)\n
    - Branch + Commit: Pin to that commit (tracked on that branch)\n
    - Tag only: Pin to that tag\n
    - --all: Updates only branch-tracked binaries (skips pinned commits/tags)

    Examples:\n
    - cargit update --all                    # Update all branch-tracked binaries\n
    - cargit update typst                    # Update typst to latest on its branch\n
    - cargit update typst --branch dev       # Switch to dev branch (latest)\n
    - cargit update typst --commit abc123    # Pin to commit abc123\n
    - cargit update typst --branch dev --commit abc123  # Pin to commit on dev branch\n
    - cargit update typst --tag v0.11.0      # Pin to tag v0.11.0
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

        # Cannot specify branch/commit/tag with --all
        if all and (branch or commit or tag):
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

        # TODO: Improve update for all packages to be parallelized (both git fetch and cargo build)
        for binary_alias in targets:
            info = metadata["installed"][binary_alias]
            stored_branch = info.get("branch", "")

            # Skip pinned commits/tags when using --all
            if all:
                if stored_branch.startswith("commit:") or stored_branch.startswith(
                    "tag:"
                ):
                    rprint(
                        f"[yellow]Skipping {binary_alias} (pinned to specific commit/tag)[/yellow]"
                    )
                    continue

            rprint(f"[blue]Checking {binary_alias}...[/blue]")
            repo_path = get_repo_path(info["repo_url"])

            if not repo_path.exists():
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
                binary_path = build_binary(repo_path, info.get("crate"))
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
                        # This should have been caught by the skip logic above for --all
                        # For individual updates, stay on the pinned commit
                        rprint(
                            f"[yellow]{binary_alias} is pinned to a specific commit[/yellow]"
                        )
                        rprint(
                            "[yellow]Use explicit --branch, --commit, or --tag to change[/yellow]"
                        )
                        continue
                    elif stored_branch.startswith("tag:"):
                        # Pinned to tag - don't update unless explicit
                        rprint(
                            "[yellow]{binary_alias} is pinned to a specific tag[/yellow]"
                        )
                        rprint(
                            "[yellow]Use explicit --branch, --commit, or --tag to change[/yellow]"
                        )
                        continue
                    else:
                        # Regular branch - update to latest
                        target_branch = stored_branch
                        target_commit = None
                        target_tag = None

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
                    binary_path = build_binary(repo_path, info.get("crate"))
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

                    rprint(f"[green]Updated {binary_alias}![/green]")
                else:
                    rprint(f"[green]{binary_alias} is up to date[/green]")

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
def rename(
    current_alias: str = typer.Argument(..., help="Current alias of the binary"),
    new_alias: str = typer.Option(..., "--to", help="New alias for the binary"),
):
    """Rename an installed binary's alias

    Examples:
        cargit rename typst-dev --to typst
        cargit rename old-name --to new-name
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
