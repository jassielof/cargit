"""Info command for cargit."""

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

from cli.core import (
    CargitError,
    _expand_workspace_members,
    ensure_dirs,
    get_current_commit,
    get_default_branch,
    run_command,
)
from cli.storage import get_binaries_by_repo

app = typer.Typer()


@app.command()
def info(
    git_url: str = typer.Argument(..., help="Git repository URL to inspect"),
    branches: bool = typer.Option(
        False, "--branches", help="List remote branches (slower)."
    ),
    tags: bool = typer.Option(False, "--tags", help="List remote tags (slower)."),
):
    """Show repository binaries, features, and workspace layout without installing.

    - Verifies the remote exists (git ls-remote) before cloning.
    - Uses a fast, shallow, temporary clone (no persistence).
    - Handles single crates and workspaces, surfacing binaries and required features.
    """
    try:
        ensure_dirs()

        _assert_remote_exists(git_url)
        repo_path, temp_clone = _prepare_repo_for_info(git_url, tags)

        try:
            default_branch = get_default_branch(repo_path)
            commit = get_current_commit(repo_path)
            cargo_data = _load_cargo_data(repo_path)

            rprint(f"\n[bold cyan]Repository: {git_url}[/bold cyan]")
            rprint(f"{'─' * 60}\n")
            rprint(f"[bold]Default branch:[/bold] {default_branch}")
            rprint(f"[bold]Current commit:[/bold] {commit[:8]}\n")

            if branches:
                _render_remote_refs(git_url, ref_type="heads")
            if tags:
                _render_remote_refs(git_url, ref_type="tags")
            if branches or tags:
                rprint("")

            if "workspace" in cargo_data:
                _render_workspace_info(repo_path, cargo_data, git_url)
            else:
                _render_single_crate_info(repo_path, cargo_data, git_url)

            _render_existing_installations(git_url)

        finally:
            if temp_clone:
                shutil.rmtree(repo_path, ignore_errors=True)

    except CargitError as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _assert_remote_exists(git_url: str) -> None:
    try:
        run_command(
            ["git", "ls-remote", "--exit-code", git_url, "HEAD"], capture_output=True
        )
    except CargitError as e:
        raise CargitError(f"Could not reach repository: {e}")


def _prepare_repo_for_info(git_url: str, include_tags: bool) -> tuple[Path, bool]:
    rprint("[blue]Fetching repository information...[/blue]")
    temp_dir = tempfile.mkdtemp(prefix="cargit_info_")
    repo_path = Path(temp_dir)

    clone_cmd = [
        "git",
        "clone",
        "--depth=1",
        "--single-branch",
        "--filter=blob:limit=200k",
    ]

    if not include_tags:
        clone_cmd.append("--no-tags")

    clone_cmd.extend([git_url, str(repo_path)])
    run_command(clone_cmd)
    return repo_path, True


def _load_cargo_data(repo_path: Path) -> dict:
    cargo_toml = repo_path / "Cargo.toml"
    if not cargo_toml.exists():
        raise CargitError("No Cargo.toml found in repository")

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

    workspace = cargo_data["workspace"]
    workspace_pkg = workspace.get("package", {})
    workspace_version = (
        workspace_pkg.get("version") if isinstance(workspace_pkg, dict) else None
    )

    default_members = workspace.get("default-members", [])
    default_members_norm = [m.replace("\\", "/") for m in default_members]
    members = workspace.get("members", [])
    expanded_members = _expand_workspace_members(repo_path, members)

    default_display = ", ".join(default_members) if default_members else "(none)"
    rprint(f"[bold]Default members:[/bold] {default_display}")
    if default_members:
        rprint(
            "[dim]All default members are installed when no crate is specified.[/dim]"
        )

    rendered = 0
    rprint("\n[bold]Crates:[/bold]")
    for member_path in expanded_members:
        member_norm = member_path.replace("\\", "/")
        crate_path = repo_path / member_path
        crate_data = _load_member_toml(crate_path)
        crate_info = _extract_crate_info(crate_data, crate_path, workspace_version)

        if not crate_info["binaries"]:
            continue  # Skip pure library crates

        rendered += 1
        is_default = (
            member_norm in default_members_norm
            or crate_info["package"] in default_members_norm
            or crate_info["package"] in default_members
        )
        default_marker = " [green](default)[/green]" if is_default else ""
        rprint(f"  • {crate_info['package']}{default_marker}")
        rprint(f"    Version: {crate_info['version']}")
        rprint(f"    Path: {member_path}")

        rprint("    Binaries:")
        for bin_entry in crate_info["binaries"]:
            features = bin_entry.get("required_features", [])
            feature_str = (
                f" [dim](features: {', '.join(features)})[/dim]" if features else ""
            )
            rprint(f"      - {bin_entry['name']}{feature_str}")

        feature_union = _collect_feature_union(crate_info["binaries"])
        install_cmd = _build_install_command(
            git_url, crate_info["package"], feature_union, annotate_features=True
        )
        rprint(f"    Install: {install_cmd}\n")

    if rendered == 0:
        rprint(
            "  [yellow]No crates with binaries were found in workspace members.[/yellow]"
        )


def _render_existing_installations(git_url: str):
    existing = get_binaries_by_repo(git_url)
    if not existing:
        return

    rprint("\n[bold yellow]Already installed from this repo:[/bold yellow]")
    for binary in existing:
        crate_info = f" (crate: {binary['crate']})" if binary.get("crate") else ""
        rprint(f"  • {binary['alias']}{crate_info} @ {binary['commit'][:8]}")


def _render_single_crate_info(repo_path: Path, cargo_data: dict, git_url: str):
    rprint("[bold]Type:[/bold] Single Crate")

    crate_info = _extract_crate_info(cargo_data, repo_path)
    rprint(f"[bold]Name:[/bold] {crate_info['package']}")
    rprint(f"[bold]Version:[/bold] {crate_info['version']}")
    if crate_info.get("description"):
        rprint(f"[bold]Description:[/bold] {crate_info['description']}")

    if not crate_info["binaries"]:
        raise CargitError(
            "This crate does not define any binaries (bin section missing)."
        )

    rprint("\n[bold]Binaries:[/bold]")
    for bin_entry in crate_info["binaries"]:
        features = bin_entry.get("required_features", [])
        feature_str = (
            f" [dim](features: {', '.join(features)})[/dim]" if features else ""
        )
        rprint(f"  • {bin_entry['name']}{feature_str}")

    feature_union = _collect_feature_union(crate_info["binaries"])
    install_cmd = _build_install_command(
        git_url, None, feature_union, annotate_features=True
    )
    rprint(f"\n[dim]Install with: {install_cmd}[/dim]")


def _extract_crate_info(
    cargo_data: dict, crate_path: Path, workspace_version: str | None = None
) -> dict[str, Any]:
    if "package" not in cargo_data:
        raise CargitError(f"Cargo.toml at {crate_path} is missing [package] section")

    package = cargo_data["package"]
    bins = cargo_data.get("bin", []) or []

    binaries: list[dict[str, Any]] = []
    for bin_entry in bins:
        name = bin_entry.get("name")
        if not name:
            continue
        binaries.append(
            {
                "name": name,
                "required_features": bin_entry.get("required-features", []),
            }
        )

    version = package.get("version", "unknown")
    if isinstance(version, dict) and version.get("workspace") and workspace_version:
        version = workspace_version

    return {
        "package": package.get("name", "unknown"),
        "version": version,
        "description": package.get("description", ""),
        "binaries": binaries,
    }


def _collect_feature_union(binaries: list[dict[str, Any]]) -> list[str]:
    features: set[str] = set()
    for bin_entry in binaries:
        for feat in bin_entry.get("required_features", []) or []:
            features.add(feat)
    return sorted(features)


def _build_install_command(
    git_url: str,
    crate: str | None,
    features: list[str],
    annotate_features: bool = False,
) -> str:
    parts = ["cargit", "install", git_url]
    if crate:
        parts.append(crate)
    cmd = " ".join(parts)
    if annotate_features and features:
        return f"{cmd} [auto-features: {', '.join(features)}]"
    return cmd


def _load_member_toml(crate_path: Path) -> dict:
    manifest = crate_path / "Cargo.toml"
    if not manifest.exists():
        raise CargitError(f"Missing Cargo.toml in workspace member {crate_path}")

    try:
        import tomllib

        with open(manifest, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        import tomlkit

        with open(manifest, "r", encoding="utf-8") as f:
            return tomlkit.load(f)


def _render_remote_refs(git_url: str, ref_type: str) -> None:
    ref_flag = "--heads" if ref_type == "heads" else "--tags"
    label = "branches" if ref_type == "heads" else "tags"
    result = run_command(["git", "ls-remote", ref_flag, git_url], capture_output=True)
    refs = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            refs.append(parts[1].split("/")[-1])
    if refs:
        rprint(f"[bold]{label.title()}:[/bold] {', '.join(sorted(refs))}")
    else:
        rprint(f"[bold]{label.title()}:[/bold] (none)")
