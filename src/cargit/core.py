import os
import subprocess
import urllib.parse
from pathlib import Path

from rich import print as rprint

# Constants
CARGIT_DIR = Path.home() / ".cargit"
REPOS_DIR = CARGIT_DIR / "repos"
BIN_DIR = CARGIT_DIR / "bin"


class CargitError(Exception):
    """Custom exception for cargit operations"""

    pass


def ensure_dirs():
    """Ensure required directories exist"""
    CARGIT_DIR.mkdir(exist_ok=True)
    REPOS_DIR.mkdir(exist_ok=True)
    BIN_DIR.mkdir(exist_ok=True)


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


def clone_repository(git_url: str, branch: str | None = None) -> tuple[Path, str]:
    """Clone repository with optimization for single branch"""
    repo_path = get_repo_path(git_url)

    if repo_path.exists():
        # Repository already exists, just fetch
        rprint(f"[blue]Repository already exists at {repo_path}[/blue]")
        run_command(["git", "fetch", "origin"], cwd=repo_path)

        if branch is None:
            branch = get_default_branch(repo_path)

        # Reset to latest commit to avoid conflicts
        current_commit = get_current_commit(repo_path)
        remote_commit = get_remote_commit(repo_path, branch)

        if current_commit != remote_commit:
            rprint(f"[blue]Updating to latest commit on {branch}[/blue]")
            run_command(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_path)

        return repo_path, branch

    # Clone repository
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    clone_cmd = ["git", "clone"]

    if branch:
        # Clone specific branch only for faster cloning
        clone_cmd.extend(["--single-branch", "--branch", branch])
    else:
        # Clone with minimal history for faster cloning
        clone_cmd.append("--depth=1")

    clone_cmd.extend([git_url, str(repo_path)])

    rprint(f"[blue]Cloning repository: {git_url}[/blue]")
    run_command(clone_cmd)

    if branch is None:
        branch = get_default_branch(repo_path)

    return repo_path, branch


def build_binary(repo_path: Path, crate_name: str | None = None) -> Path:
    """Build the binary with cargo"""
    rprint("[blue]Building with cargo...[/blue]")

    # Build command
    build_cmd = ["cargo", "build", "--release"]

    # If specific crate is specified, add it to the build command
    if crate_name:
        build_cmd.extend(["--package", crate_name])

    run_command(build_cmd, cwd=repo_path)

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

    return binary_path


def install_binary(binary_path: Path, alias: str, install_dir: Path):
    """Install binary by creating symlink"""
    install_dir.mkdir(parents=True, exist_ok=True)
    symlink_path = install_dir / alias

    # Remove existing symlink if it exists
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    # Create symlink
    symlink_path.symlink_to(binary_path)
    rprint(f"[green]Installed {alias} -> {binary_path}[/green]")

    # Check if install_dir is in PATH
    if str(install_dir) not in os.environ.get("PATH", ""):
        rprint(f"[yellow]Warning: {install_dir} is not in your PATH[/yellow]")
        rprint(
            f'[yellow]Add this to your shell profile: export PATH="{install_dir}:$PATH"[/yellow]'
        )
