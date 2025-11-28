"""Core functionality for cargit.

This module contains the main operations for git and cargo management,
including cloning, building, and updating repositories.
"""

import os
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Callable, TypeVar

from rich import print as rprint

# XDG Base Directory paths
CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "cargit"
CONFIG_DIR = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "cargit"
DATA_DIR = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share")) / "cargit"

# Legacy path for migration
OLD_CARGIT_DIR = Path.home() / ".cargit"

# Directory paths
REPOS_DIR = CACHE_DIR  # Git repositories and build artifacts
BIN_DIR = DATA_DIR  # Symlinks to binaries
BINARIES_SAFE_DIR = CACHE_DIR / "binaries"  # Safe location for copied binaries


class CargitError(Exception):
    """Custom exception for cargit operations."""

    pass


T = TypeVar("T")


def with_retry(
    operation: Callable[[], T],
    max_attempts: int = 3,
    delay: float = 1.0,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Execute an operation with retry logic.

    Args:
        operation: Function to execute
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        on_retry: Optional callback called on each retry with (attempt, exception)

    Returns:
        Result of the operation

    Raises:
        The last exception if all attempts fail
    """
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                if on_retry:
                    on_retry(attempt, e)
                time.sleep(delay * attempt)  # Exponential backoff

    raise last_exception  # type: ignore


def ensure_dirs():
    """Ensure required directories exist and handle migration from old structure"""
    import shutil

    # Check if migration is needed from old ~/.cargit
    needs_migration = OLD_CARGIT_DIR.exists() and not CACHE_DIR.exists()

    if needs_migration:
        rprint("[blue]Migrating from old ~/.cargit to XDG directories...[/blue]")

        # Create new directories
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        BINARIES_SAFE_DIR.mkdir(parents=True, exist_ok=True)

        # Migrate repos
        old_repos = OLD_CARGIT_DIR / "repos"
        if old_repos.exists():
            rprint(f"[blue]Moving repositories to {CACHE_DIR}...[/blue]")
            for provider_dir in old_repos.iterdir():
                if provider_dir.is_dir():
                    dest = CACHE_DIR / provider_dir.name
                    if not dest.exists():
                        shutil.move(str(provider_dir), str(CACHE_DIR))

        # Migrate binaries (symlinks)
        old_bin = OLD_CARGIT_DIR / "bin"
        if old_bin.exists():
            rprint(f"[blue]Moving binaries to {DATA_DIR}...[/blue]")
            for binary in old_bin.iterdir():
                dest = DATA_DIR / binary.name
                if not dest.exists():
                    if binary.is_symlink():
                        # Recreate symlink with new target path if needed
                        target = binary.resolve()
                        # Update target path if it points to old location
                        target_str = str(target)
                        if str(OLD_CARGIT_DIR) in target_str:
                            new_target = Path(target_str.replace(str(OLD_CARGIT_DIR / "repos"), str(CACHE_DIR)))
                            dest.symlink_to(new_target)
                        else:
                            dest.symlink_to(target)
                    else:
                        shutil.move(str(binary), str(dest))

        # Database migration is handled in storage.py

        # Remove old directory after successful migration
        try:
            if OLD_CARGIT_DIR.exists():
                # Only remove if empty or only contains db file (which will be migrated by storage.py)
                remaining = list(OLD_CARGIT_DIR.iterdir())
                if not remaining or (len(remaining) == 1 and remaining[0].name == "cargit.db"):
                    rprint("[green]Migration completed successfully![/green]")
        except Exception as e:
            rprint(f"[yellow]Warning: Could not clean up old directory: {e}[/yellow]")

    # Ensure all directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BINARIES_SAFE_DIR.mkdir(parents=True, exist_ok=True)


def run_command(
    cmd: list[str], cwd: Path = None, capture_output: bool = False
) -> subprocess.CompletedProcess:
    """Run a command with error handling"""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=capture_output, text=True, check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        raise CargitError(
            f"Command failed: {' '.join(cmd)}\n{e.stderr if e.stderr else str(e)}"
        )
    except FileNotFoundError:
        raise CargitError(f"Command not found: {cmd[0]}")


def parse_git_url(url: str) -> tuple[str, str, str]:
    """Parse git URL to extract provider, owner, repo name"""
    # Handle different URL formats
    if url.startswith("git@"):
        # SSH format: git@github.com:owner/repo.git
        parts = url.replace("git@", "").replace(":", "/").split("/")
        provider = parts[0]
        owner = parts[1]
        repo = parts[2].replace(".git", "")
    else:
        # HTTPS format: https://github.com/owner/repo.git
        parsed = urllib.parse.urlparse(url)
        provider = parsed.netloc
        path_parts = parsed.path.strip("/").split("/")
        owner = path_parts[0]
        repo = path_parts[1].replace(".git", "")

    return provider, owner, repo


def get_repo_path(git_url: str) -> Path:
    """Get the local path for a repository"""
    provider, owner, repo = parse_git_url(git_url)
    return REPOS_DIR / provider / owner / repo


def get_default_branch(repo_path: Path) -> str:
    """Get the default branch of a repository"""
    try:
        result = run_command(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=repo_path,
            capture_output=True,
        )
        return result.stdout.strip().split("/")[-1]
    except CargitError:
        # Fallback to main/master
        try:
            run_command(
                ["git", "show-ref", "--verify", "refs/remotes/origin/main"],
                cwd=repo_path,
                capture_output=True,
            )
            return "main"
        except CargitError:
            try:
                run_command(
                    ["git", "show-ref", "--verify", "refs/remotes/origin/master"],
                    cwd=repo_path,
                    capture_output=True,
                )
                return "master"
            except CargitError:
                raise CargitError("Could not determine default branch")


def get_current_commit(repo_path: Path) -> str:
    """Get current commit hash"""
    result = run_command(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True
    )
    return result.stdout.strip()


def get_remote_commit(repo_path: Path, branch: str) -> str:
    """Get remote commit hash for branch"""
    result = run_command(
        ["git", "rev-parse", f"origin/{branch}"], cwd=repo_path, capture_output=True
    )
    return result.stdout.strip()


def get_binary_name_from_cargo(repo_path: Path, crate_name: str | None = None) -> str:
    """Extract binary name from Cargo.toml"""
    cargo_toml = repo_path / "Cargo.toml"
    if not cargo_toml.exists():
        raise CargitError("No Cargo.toml found in repository")

    try:
        # For Python >= 3.11
        import tomllib

        use_tomllib = True

        with open(cargo_toml, "rb") as f:
            cargo_data = tomllib.load(f)
    except ImportError:
        # For Python < 3.11
        import tomlkit

        use_tomllib = False

        with open(cargo_toml, "r", encoding="utf-8") as f:
            cargo_data = tomlkit.load(f)

    # Handle workspace configurations
    if "workspace" in cargo_data:
        if crate_name is None:
            # Try to get from default-members first
            if "default-members" in cargo_data["workspace"]:
                default_member = cargo_data["workspace"]["default-members"][0]
                # Extract crate name from path (e.g., "crates/typst-cli" -> "typst-cli")
                crate_name = Path(default_member).name
            else:
                # List available workspace members
                members = cargo_data["workspace"].get("members", [])
                expanded_members = _expand_workspace_members(repo_path, members)
                available_crates = _get_available_crates_from_members(
                    repo_path, expanded_members
                )
                raise CargitError(
                    f"This is a workspace with multiple crates. Please specify which crate to install.\n"
                    f"Available crates: {', '.join(available_crates)}\n"
                    f"Usage: cargit install <git_url> <crate_name>"
                )

        # Expand workspace members (handle glob patterns like "crates/*")
        members = cargo_data["workspace"].get("members", [])
        expanded_members = _expand_workspace_members(repo_path, members)

        # Look for the specific crate in workspace members
        crate_path = None

        # Find the crate path that matches the crate name
        for member_path_str in expanded_members:
            member_path = repo_path / member_path_str
            member_cargo = member_path / "Cargo.toml"

            if member_cargo.exists():
                try:
                    if use_tomllib:
                        with open(member_cargo, "rb") as f:
                            member_data = tomllib.load(f)
                    else:
                        with open(member_cargo, "r", encoding="utf-8") as f:
                            member_data = tomlkit.load(f)

                    if (
                        "package" in member_data
                        and member_data["package"].get("name") == crate_name
                    ):
                        crate_path = member_path
                        break
                except Exception:
                    continue

        if crate_path is None:
            available_crates = _get_available_crates_from_members(
                repo_path, expanded_members
            )
            raise CargitError(
                f"Crate '{crate_name}' not found in workspace.\n"
                f"Available crates: {', '.join(available_crates)}"
            )

        # Get binary name from the specific crate's Cargo.toml
        crate_cargo = crate_path / "Cargo.toml"
        try:
            if use_tomllib:
                with open(crate_cargo, "rb") as f:
                    crate_data = tomllib.load(f)
            else:
                with open(crate_cargo, "r", encoding="utf-8") as f:
                    crate_data = tomlkit.load(f)

            # Check for [[bin]] section first
            if "bin" in crate_data and crate_data["bin"]:
                return crate_data["bin"][0]["name"]

            # Fallback to package name
            if "package" in crate_data and "name" in crate_data["package"]:
                return crate_data["package"]["name"]

        except Exception as e:
            raise CargitError(f"Could not read crate Cargo.toml: {e}")

    else:
        # Regular single-crate repository
        if crate_name is not None:
            rprint(
                f"[yellow]Warning: Crate name '{crate_name}' specified but this is not a workspace. Ignoring.[/yellow]"
            )

        # Check for [[bin]] section first
        if "bin" in cargo_data and cargo_data["bin"]:
            return cargo_data["bin"][0]["name"]

        # Fallback to package name
        if "package" in cargo_data and "name" in cargo_data["package"]:
            return cargo_data["package"]["name"]

    raise CargitError("Could not determine binary name from Cargo.toml")


def _expand_workspace_members(repo_path: Path, members: list[str]) -> list[str]:
    """Expand workspace member patterns like 'crates/*' to actual paths"""
    import glob

    expanded = []
    for member in members:
        if "*" in member:
            # Handle glob patterns
            pattern = repo_path / member
            matches = glob.glob(str(pattern))
            for match in matches:
                # Convert back to relative path
                rel_path = Path(match).relative_to(repo_path)
                expanded.append(str(rel_path))
        else:
            expanded.append(member)

    return expanded


def _get_available_crates_from_members(
    repo_path: Path, members: list[str]
) -> list[str]:
    """Get list of available crate names from workspace members"""
    try:
        # For Python >= 3.11
        import tomllib

        use_tomllib = True
    except ImportError:
        # For Python < 3.11
        import tomlkit

        use_tomllib = False

    available_crates = []
    for member in members:
        member_path = repo_path / member
        member_cargo = member_path / "Cargo.toml"
        if member_cargo.exists():
            try:
                if use_tomllib:
                    with open(member_cargo, "rb") as f:
                        member_data = tomllib.load(f)
                else:
                    with open(member_cargo, "r", encoding="utf-8") as f:
                        member_data = tomlkit.load(f)

                if "package" in member_data:
                    available_crates.append(member_data["package"]["name"])
            except Exception:
                continue

    return available_crates


def _is_shallow_repo(repo_path: Path) -> bool:
    """Check if repository is a shallow clone"""
    shallow_file = repo_path / ".git" / "shallow"
    return shallow_file.exists()


def _convert_to_shallow(repo_path: Path, branch: str):
    """Convert existing repo to shallow clone"""
    rprint("[blue]Converting to shallow clone to save disk space...[/blue]")
    try:
        # Create a new shallow clone with depth 1
        run_command(
            ["git", "fetch", "--depth=1", "origin", f"{branch}:{branch}"],
            cwd=repo_path,
        )

        # Reset to the fetched branch
        run_command(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_path)

        # Clean up unreferenced objects
        run_command(["git", "gc", "--aggressive", "--prune=all"], cwd=repo_path)

    except CargitError as e:
        rprint(f"[yellow]Warning: Could not convert to shallow clone: {e}[/yellow]")


def clone_repository(git_url: str, branch: str | None = None) -> tuple[Path, str]:
    """Clone repository with optimization for minimal disk usage

    Unlike 'cargo install', this keeps the repository and build cache intact,
    enabling fast incremental updates. The repository uses shallow cloning
    (only latest commit) to minimize git history size.
    """
    repo_path = get_repo_path(git_url)

    if repo_path.exists():
        # Repository already exists
        rprint(f"[blue]Repository exists at {repo_path}[/blue]")

        # Determine the branch to use
        if branch is None:
            branch = get_default_branch(repo_path)

        try:
            # Fetch latest changes (shallow fetch if possible)
            if _is_shallow_repo(repo_path):
                rprint("[blue]Fetching latest changes (shallow)...[/blue]")
                run_command(
                    ["git", "fetch", "--depth=1", "origin", branch],
                    cwd=repo_path,
                )
            else:
                rprint("[blue]Fetching latest changes...[/blue]")
                run_command(["git", "fetch", "origin", branch], cwd=repo_path)
                # Optionally convert to shallow to save space
                _convert_to_shallow(repo_path, branch)

            # Get current and remote commits
            current_commit = get_current_commit(repo_path)
            remote_commit = get_remote_commit(repo_path, branch)

            if current_commit != remote_commit:
                rprint(f"[blue]Updating to latest commit on {branch}[/blue]")
                # Hard reset to remote branch, discarding all local changes
                # This preserves the target/ directory with build cache
                run_command(
                    ["git", "reset", "--hard", f"origin/{branch}"],
                    cwd=repo_path,
                )
                # Clean untracked files but preserve target/ directory
                # Note: target/ is in .gitignore so it won't be affected by git clean
                run_command(
                    ["git", "clean", "-fdx"],
                    cwd=repo_path,
                )
            else:
                rprint("[green]Repository is already up to date[/green]")

        except CargitError as e:
            rprint(
                f"[yellow]Warning: Update failed, will use existing state: {e}[/yellow]"
            )

        return repo_path, branch

    # Clone repository with minimal disk usage
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    # Use shallow clone with depth 1 and single branch
    clone_cmd = [
        "git",
        "clone",
        "--depth=1",  # Only fetch the latest commit
        "--single-branch",  # Only fetch the specified branch
        "--no-tags",  # Don't fetch tags to save space
    ]

    if branch:
        clone_cmd.extend(["--branch", branch])

    clone_cmd.extend([git_url, str(repo_path)])

    rprint(f"[blue]Cloning repository (shallow): {git_url}[/blue]")
    run_command(clone_cmd)

    # Determine actual branch after cloning
    if branch is None:
        branch = get_default_branch(repo_path)

    # Run garbage collection to optimize disk space
    rprint("[blue]Optimizing disk space...[/blue]")
    run_command(["git", "gc", "--aggressive", "--prune=all"], cwd=repo_path)

    return repo_path, branch


def build_binary(repo_path: Path, crate_name: str | None = None, alias: str | None = None) -> tuple[Path, float]:
    """Build the binary with cargo, preserving build cache for fast updates.

    Unlike 'cargo install' which removes the repo after building, this keeps
    the repository and target/ directory intact. This means:
    - First build: Compiles all dependencies from scratch
    - Subsequent builds with no changes: Near-instant (cache hit)
    - Updates with changes: Only recompiles changed code and dependencies

    This dependency reuse makes updates significantly faster, especially for
    large projects with many dependencies.

    Args:
        repo_path: Path to the repository
        crate_name: Optional crate name for workspaces
        alias: Optional alias for build time tracking

    Returns:
        Tuple of (binary_path, build_duration_seconds)
    """
    rprint("[blue]Building with cargo (reusing cached dependencies)...[/blue]")

    # Build command
    build_cmd = ["cargo", "build", "--release"]

    # If specific crate is specified, add it to the build command
    if crate_name:
        build_cmd.extend(["--package", crate_name])

    # Track build time
    start_time = time.time()
    run_command(build_cmd, cwd=repo_path)
    build_duration = time.time() - start_time

    target_dir = repo_path / "target" / "release"
    if not target_dir.exists():
        raise CargitError("Build directory not found")

    # Find the binary (usually the same name as the package)
    binary_name = get_binary_name_from_cargo(repo_path, crate_name)
    binary_path = target_dir / binary_name

    if not binary_path.exists():
        # Try to find any executable in the release directory
        executables = [
            f
            for f in target_dir.iterdir()
            if f.is_file() and os.access(f, os.X_OK) and not f.suffix
        ]
        if executables:
            binary_path = executables[0]
            rprint(
                f"[yellow]Expected binary '{binary_name}' not found, using '{binary_path.name}' instead[/yellow]"
            )
        else:
            raise CargitError(f"Built binary not found in {target_dir}")

    # Store build time if alias is provided
    if alias:
        try:
            from .storage import update_build_time
            update_build_time(alias, build_duration)
        except Exception:
            pass  # Don't fail if build time tracking fails

    # Display build time
    if build_duration >= 60:
        minutes = int(build_duration // 60)
        seconds = int(build_duration % 60)
        rprint(f"[dim]Build completed in {minutes}m {seconds}s[/dim]")
    else:
        rprint(f"[dim]Build completed in {build_duration:.1f}s[/dim]")

    return binary_path, build_duration


def install_binary(binary_path: Path, alias: str, install_dir: Path, binary_type: str = "symlink"):
    """Install binary by creating symlink or copy

    Args:
        binary_path: Path to the built binary
        alias: Alias name for the binary
        install_dir: Directory to install to
        binary_type: "symlink" or "copy"
    """
    import shutil

    install_dir.mkdir(parents=True, exist_ok=True)
    target_path = install_dir / alias

    # Remove existing file if it exists
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()

    if binary_type == "copy":
        # Copy binary to install location
        shutil.copy2(binary_path, target_path)
        target_path.chmod(0o755)  # Make executable
        rprint(f"[green]Installed {alias} (copied) -> {binary_path}[/green]")
    else:
        # Create symlink (default)
        target_path.symlink_to(binary_path)
        rprint(f"[green]Installed {alias} (symlink) -> {binary_path}[/green]")

    # Check if install_dir is in PATH
    if str(install_dir) not in os.environ.get("PATH", ""):
        rprint(f"[yellow]Warning: {install_dir} is not in your PATH[/yellow]")
        rprint(
            f'[yellow]Add this to your shell profile: export PATH="{install_dir}:$PATH"[/yellow]'
        )


def update_repository(
    repo_path: Path,
    branch: str | None = None,
    commit: str | None = None,
    tag: str | None = None,
) -> tuple[str, bool]:
    """Update repository to specific branch, commit, or tag

    Behavior:
    - Only branch: Update to latest HEAD of that branch
    - Only commit: Switch to that commit on current branch (detached state)
    - Branch + commit: Switch to that commit, track as on that branch
    - Only tag: Switch to that tag (detached state)

    Returns:
        tuple[str, bool]: (actual_branch, was_updated)
    """
    # Get current commit before update
    current_commit_hash = get_current_commit(repo_path)

    # Always fetch first to ensure remote refs are up-to-date
    # This is cheap and prevents incorrect "up-to-date" messages
    try:
        if _is_shallow_repo(repo_path):
            run_command(["git", "fetch", "--depth=1", "origin"], cwd=repo_path)
        else:
            run_command(["git", "fetch", "origin"], cwd=repo_path)
    except CargitError as e:
        rprint(f"[yellow]Warning: Could not fetch remote updates: {e}[/yellow]")

    # Determine what we're updating to
    if tag:
        # Tag takes precedence
        target_type = "tag"
    elif commit:
        target_type = "commit"
    else:
        # Branch update (default)
        if branch is None:
            branch = get_default_branch(repo_path)
        target_type = "branch"

    # Handle each update type
    if target_type == "tag":
        # For tags, check if we already have it
        try:
            # Try to verify if the tag exists locally
            tag_commit = run_command(
                ["git", "rev-list", "-n", "1", f"tags/{tag}"],
                cwd=repo_path,
                capture_output=True,
            ).stdout.strip()

            # Tag exists, check if we're already on it
            if current_commit_hash == tag_commit:
                rprint(f"[green]Already on tag {tag}[/green]")
                return f"tag:{tag}", False

            # Tag exists but we're not on it, just checkout
            rprint(f"[blue]Checking out existing tag {tag}...[/blue]")
            run_command(["git", "checkout", f"tags/{tag}"], cwd=repo_path)

        except CargitError:
            # Tag doesn't exist locally, need to fetch
            rprint(f"[blue]Fetching tag {tag}...[/blue]")
            if _is_shallow_repo(repo_path):
                run_command(
                    [
                        "git",
                        "fetch",
                        "--depth=1",
                        "origin",
                        f"refs/tags/{tag}:refs/tags/{tag}",
                    ],
                    cwd=repo_path,
                )
            else:
                run_command(
                    ["git", "fetch", "origin", f"refs/tags/{tag}:refs/tags/{tag}"],
                    cwd=repo_path,
                )

            # Checkout the tag
            run_command(["git", "checkout", f"tags/{tag}"], cwd=repo_path)

        # Return tag name as branch for tracking
        actual_branch = f"tag:{tag}"

    elif target_type == "commit":
        # Determine which branch context to use
        if branch is None:
            # No branch specified, try to get current branch
            try:
                current_branch_result = run_command(
                    ["git", "symbolic-ref", "--short", "HEAD"],
                    cwd=repo_path,
                    capture_output=True,
                )
                branch = current_branch_result.stdout.strip()
            except CargitError:
                # We're in detached state, try to extract branch from metadata
                # For now, we'll mark it as detached
                branch = "detached"

        # Check if commit exists locally
        try:
            run_command(
                ["git", "cat-file", "-e", commit],
                cwd=repo_path,
                capture_output=True,
            )
            commit_exists_locally = True
        except CargitError:
            commit_exists_locally = False

        # Check if we're already on this commit
        if current_commit_hash == commit:
            rprint(f"[green]Already on commit {commit[:8]}[/green]")
            return f"commit:{branch}:{commit[:8]}", False

        if not commit_exists_locally:
            # Need to fetch the commit
            if _is_shallow_repo(repo_path):
                rprint(
                    "[blue]Fetching specific commit (may need to unshallow)...[/blue]"
                )

                # Try shallow fetch first with the branch
                if branch and branch != "detached":
                    try:
                        run_command(
                            ["git", "fetch", "--depth=50", "origin", branch],
                            cwd=repo_path,
                        )
                        # Check if we got the commit
                        run_command(
                            ["git", "cat-file", "-e", commit],
                            cwd=repo_path,
                            capture_output=True,
                        )
                        commit_exists_locally = True
                    except CargitError:
                        pass

                if not commit_exists_locally:
                    # Fetch with increasing depth
                    rprint(
                        "[yellow]Commit not in recent history, fetching more...[/yellow]"
                    )
                    try:
                        run_command(
                            ["git", "fetch", "--deepen=100", "origin"],
                            cwd=repo_path,
                        )
                        run_command(
                            ["git", "cat-file", "-e", commit],
                            cwd=repo_path,
                            capture_output=True,
                        )
                        commit_exists_locally = True
                    except CargitError:
                        # Last resort: unshallow completely
                        rprint(
                            "[yellow]Unshallowing repository to find commit...[/yellow]"
                        )
                        run_command(
                            ["git", "fetch", "--unshallow", "origin"],
                            cwd=repo_path,
                        )
            else:
                # Full repo, just fetch normally
                rprint(f"[blue]Fetching to get commit {commit[:8]}...[/blue]")
                run_command(["git", "fetch", "origin"], cwd=repo_path)

        # Checkout the commit
        rprint(f"[blue]Checking out commit {commit[:8]}...[/blue]")
        run_command(["git", "checkout", commit], cwd=repo_path)

        # Return tracking info with branch context
        actual_branch = f"commit:{branch}:{commit[:8]}"

    else:  # branch update
        # Try to get the remote commit hash without fetching
        try:
            remote_commit_hash = get_remote_commit(repo_path, branch)

            # If we already have the latest commit, no need to fetch
            if current_commit_hash == remote_commit_hash:
                rprint(
                    f"[green]{branch} is already up to date (no fetch needed)[/green]"
                )
                return branch, False
        except CargitError:
            # Remote reference doesn't exist locally, need to fetch
            # This case is now less likely due to the fetch at the start, but kept for safety
            pass

        rprint(f"[blue]Checking for updates on branch: {branch}[/blue]")

        # Fetch based on shallow/full repo
        if _is_shallow_repo(repo_path):
            rprint("[blue]Fetching latest changes (shallow)...[/blue]")
            run_command(
                ["git", "fetch", "--depth=1", "origin", branch],
                cwd=repo_path,
            )
        else:
            rprint("[blue]Fetching latest changes...[/blue]")
            run_command(["git", "fetch", "origin", branch], cwd=repo_path)
            # Optionally convert to shallow to save space
            _convert_to_shallow(repo_path, branch)

        # Get remote commit after fetch
        new_remote_commit_hash = get_remote_commit(repo_path, branch)

        # Check if update is needed
        if current_commit_hash == new_remote_commit_hash:
            rprint(f"[green]{branch} is already up to date[/green]")
            return branch, False

        # Update to latest commit on branch
        rprint(
            f"[blue]Updating {branch} from {current_commit_hash[:8]} to {new_remote_commit_hash[:8]}...[/blue]"
        )
        run_command(
            ["git", "reset", "--hard", f"origin/{branch}"],
            cwd=repo_path,
        )

        actual_branch = branch

    # Clean untracked files but preserve target/ and other build artifacts
    # Use -f (force) and -d (directories) but NOT -x (don't remove ignored files)
    # This preserves target/, .cargo/, and other ignored build caches
    run_command(["git", "clean", "-fd"], cwd=repo_path)

    # Check if we actually changed commits
    new_commit_hash = get_current_commit(repo_path)
    was_updated = current_commit_hash != new_commit_hash

    if was_updated:
        rprint(
            f"[green]Updated from {current_commit_hash[:8]} to {new_commit_hash[:8]}[/green]"
        )

    return actual_branch, was_updated


def check_for_updates(
    repo_path: Path,
    branch: str | None = None,
    commit: str | None = None,
    tag: str | None = None,
) -> tuple[bool, str]:
    """Check if updates are available without applying them.

    Args:
        repo_path: Path to git repository
        branch: Branch to check (if None, uses current branch)
        commit: Specific commit to check
        tag: Specific tag to check

    Returns:
        Tuple of (has_update, remote_commit_hash)
    """
    import subprocess

    # Fetch latest changes
    subprocess.run(
        ["git", "fetch", "--all"], cwd=repo_path, check=True, capture_output=True
    )

    # Get current commit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    current_commit = result.stdout.strip()

    # Determine target commit
    if commit:
        # Check specific commit
        remote_commit = commit
    elif tag:
        # Check specific tag
        result = subprocess.run(
            ["git", "rev-parse", f"refs/tags/{tag}"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        remote_commit = result.stdout.strip()
    else:
        # Check branch HEAD
        if branch is None:
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            branch = result.stdout.strip()

        # Get remote commit
        result = subprocess.run(
            ["git", "rev-parse", f"origin/{branch}"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        remote_commit = result.stdout.strip()

    has_update = current_commit != remote_commit
    return has_update, remote_commit



# Cache management utilities

def copy_binary_to_safe_location(binary_path: Path, alias: str) -> Path:
    """Copy binary to safe location in cache

    Returns the path to the copied binary
    """
    import shutil

    BINARIES_SAFE_DIR.mkdir(parents=True, exist_ok=True)
    safe_path = BINARIES_SAFE_DIR / alias

    shutil.copy2(binary_path, safe_path)
    safe_path.chmod(0o755)  # Make executable

    return safe_path


def get_cache_size(path: Path) -> int:
    """Calculate total size of directory recursively in bytes.

    Uses os.scandir() for better performance than Path.rglob().
    """
    import os

    def _scandir_size(dir_path: str) -> int:
        """Recursively calculate directory size using scandir."""
        total = 0
        try:
            with os.scandir(dir_path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            total += _scandir_size(entry.path)
                    except (PermissionError, OSError):
                        continue
        except (PermissionError, OSError):
            pass
        return total

    return _scandir_size(str(path))


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def clean_binary_artifacts(repo_path: Path) -> bool:
    """Run cargo clean on repository

    Returns True if successful
    """
    try:
        run_command(["cargo", "clean"], cwd=repo_path)
        return True
    except CargitError as e:
        rprint(f"[yellow]Warning: Could not clean artifacts: {e}[/yellow]")
        return False


def clean_binary_repo(repo_path: Path) -> bool:
    """Delete entire repository directory

    Returns True if successful
    """
    import shutil

    try:
        if repo_path.exists():
            shutil.rmtree(repo_path)
            return True
        return False
    except (PermissionError, OSError) as e:
        rprint(f"[yellow]Warning: Could not delete repo: {e}[/yellow]")
        return False


def clean_all_cache() -> bool:
    """Remove all cached repositories and build artifacts

    Returns True if successful
    """
    import shutil

    try:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            # Recreate the directory structure
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            BINARIES_SAFE_DIR.mkdir(parents=True, exist_ok=True)
            return True
        return False
    except (PermissionError, OSError) as e:
        rprint(f"[red]Error: Could not clean cache: {e}[/red]")
        return False


def find_orphaned_repos() -> list[tuple[Path, int]]:
    """Find repositories in cache that are not tracked in database.

    Uses os.scandir() for efficient directory traversal.

    Returns list of (repo_path, size_bytes) tuples
    """
    import os
    from .storage import get_all_repo_urls

    tracked_repos = {get_repo_path(url) for url in get_all_repo_urls()}
    orphaned: list[tuple[Path, int]] = []

    if not CACHE_DIR.exists():
        return orphaned

    cache_str = str(CACHE_DIR)

    try:
        with os.scandir(cache_str) as providers:
            for provider_entry in providers:
                if not provider_entry.is_dir() or provider_entry.name == "binaries":
                    continue

                try:
                    with os.scandir(provider_entry.path) as owners:
                        for owner_entry in owners:
                            if not owner_entry.is_dir():
                                continue

                            try:
                                with os.scandir(owner_entry.path) as repos:
                                    for repo_entry in repos:
                                        if not repo_entry.is_dir():
                                            continue

                                        repo_path = Path(repo_entry.path)
                                        git_dir = repo_path / ".git"

                                        if git_dir.exists() and repo_path not in tracked_repos:
                                            size = get_cache_size(repo_path)
                                            orphaned.append((repo_path, size))
                            except (PermissionError, OSError):
                                continue
                except (PermissionError, OSError):
                    continue
    except (PermissionError, OSError):
        pass

    return orphaned


# Parallel operations for sync/update

def fetch_repo_silent(
    repo_path: Path,
    branch: str | None = None,
    max_retries: int = 3,
) -> tuple[bool, str | None]:
    """Fetch repository updates silently (for parallel operations).

    Args:
        repo_path: Path to the repository
        branch: Branch to fetch (None for default)
        max_retries: Number of retry attempts for network failures

    Returns:
        Tuple of (success, error_message)
    """
    if not repo_path.exists():
        return False, "Repository not found"

    try:
        if branch is None:
            branch = get_default_branch(repo_path)
    except CargitError as e:
        return False, str(e)

    def do_fetch():
        if _is_shallow_repo(repo_path):
            run_command(
                ["git", "fetch", "--depth=1", "origin", branch],
                cwd=repo_path,
                capture_output=True,
            )
        else:
            run_command(
                ["git", "fetch", "origin", branch],
                cwd=repo_path,
                capture_output=True,
            )

    try:
        with_retry(do_fetch, max_attempts=max_retries, delay=1.0)
        return True, None
    except CargitError as e:
        return False, str(e)


def check_update_available(repo_path: Path, branch: str | None = None) -> tuple[bool, str, str]:
    """Check if updates are available after fetching.

    Returns (has_update, current_commit, remote_commit)
    """
    try:
        if branch is None:
            branch = get_default_branch(repo_path)

        current = get_current_commit(repo_path)
        remote = get_remote_commit(repo_path, branch)

        return current != remote, current, remote
    except CargitError:
        return False, "", ""


def reset_to_remote(repo_path: Path, branch: str) -> tuple[bool, str | None]:
    """Reset repository to remote branch (for parallel operations).

    Returns (success, error_message)
    """
    try:
        run_command(
            ["git", "reset", "--hard", f"origin/{branch}"],
            cwd=repo_path,
            capture_output=True,
        )
        # Clean untracked files but preserve target/
        run_command(
            ["git", "clean", "-fd"],
            cwd=repo_path,
            capture_output=True,
        )
        return True, None
    except CargitError as e:
        return False, str(e)
