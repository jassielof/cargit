"""
Core functionality for git operations, building, and installation
"""

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


def get_binary_name_from_cargo(repo_path: Path) -> str:
    """Extract binary name from Cargo.toml"""
    cargo_toml = repo_path / "Cargo.toml"
    if not cargo_toml.exists():
        raise CargitError("No Cargo.toml found in repository")

    try:
        # For Python >= 3.11
        import tomllib

        with open(cargo_toml, "rb") as f:
            cargo_data = tomllib.load(f)
    except ImportError:
        # For Python < 3.11
        import tomlkit

        with open(cargo_toml, "r", encoding="utf-8") as f:
            cargo_data = tomlkit.load(f)

    # Check for [[bin]] section first
    if "bin" in cargo_data and cargo_data["bin"]:
        return cargo_data["bin"][0]["name"]

    # Fallback to package name
    if "package" in cargo_data and "name" in cargo_data["package"]:
        return cargo_data["package"]["name"]

    raise CargitError("Could not determine binary name from Cargo.toml")


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


def build_binary(repo_path: Path) -> Path:
    """Build the binary with cargo"""
    rprint("[blue]Building with cargo...[/blue]")
    run_command(["cargo", "build", "--release"], cwd=repo_path)

    target_dir = repo_path / "target" / "release"
    if not target_dir.exists():
        raise CargitError("Build directory not found")

    # Find the binary (usually the same name as the package)
    binary_name = get_binary_name_from_cargo(repo_path)
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
