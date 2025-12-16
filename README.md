# Cargit

Git-based Cargo binary installer with cached repositories for faster updates.

Unlike `cargo install`, cargit keeps repositories and build artifacts cached, enabling **fast incremental updates**. When dependencies haven't changed, rebuilds are near-instant.

## Features

- **Fast incremental updates**: Cached repos and build artifacts mean only changed code is recompiled
- **Parallel operations**: Sync multiple binaries with parallel git fetch/reset
- **Workspace support**: Install specific crates from Cargo workspaces
- **Cache management**: Clean artifacts to save disk space, rebuild when needed
- **Build tracking**: Track and display build times and statistics
- **XDG compliant**: Follows XDG Base Directory specification

## Dependencies

These must be installed and available in your `$PATH`:

- [Git](https://git-scm.com/)
- [Rust/Cargo](https://rustup.rs/)

## Installation

```bash
# Using uv (recommended)
uv tool install git+https://github.com/jassielof/cargit

# Or with pipx
pipx install git+https://github.com/jassielof/cargit
```

## Quick Start

```bash
# Install a binary
cargit install https://github.com/sharkdp/fd

# Install from a workspace (e.g., typst)
cargit install https://github.com/typst/typst typst-cli

# Check for updates
cargit update --all --check

# Sync all binaries (parallel fetch + sequential builds)
cargit sync

# View status and cache statistics
cargit status
```

## Commands

### `install` - Install a binary from git

```bash
cargit install <git_url> [crate_name] [options]
```

Options:
- `--branch`: Install from a specific branch
- `--alias`: Custom name for the installed binary
- `--dir`: Custom installation directory

Examples:
```bash
# Simple install
cargit install https://github.com/BurntSushi/ripgrep

# Install with alias
cargit install https://github.com/BurntSushi/ripgrep --alias rg

# Install specific crate from workspace
cargit install https://github.com/typst/typst typst-cli

# Install from specific branch
cargit install https://github.com/user/repo --branch develop
```

### `update` - Update installed binaries

```bash
cargit update <alias> [options]
cargit update --all [options]
```

Options:
- `--all`: Update all installed binaries
- `--check`: Check for updates without applying them
- `--branch`: Switch to a different branch
- `--commit`: Pin to a specific commit
- `--tag`: Pin to a specific tag
- `-j, --jobs`: Number of parallel fetch operations (default: 8)

Examples:
```bash
# Update a specific binary
cargit update fd

# Check all for updates (parallel fetch)
cargit update --all --check

# Pin to a specific tag
cargit update typst --tag v0.12.0

# Pin to a specific commit
cargit update ripgrep --commit abc123
```

### `sync` - Sync all binaries efficiently

The recommended way to update everything. Runs in three phases:

1. **Phase 1**: Parallel git fetch (network-bound, configurable parallelism)
2. **Phase 2**: Parallel git reset (disk-bound, only repos with updates)
3. **Phase 3**: Sequential cargo builds (CPU-bound, one at a time)

```bash
cargit sync [options]
```

Options:
- `-j, --jobs`: Number of parallel git operations (default: 8)
- `--fetch-only`: Only fetch, don't reset or build
- `--dry-run`: Preview what would be updated

Examples:
```bash
# Full sync
cargit sync

# Preview updates
cargit sync --dry-run

# Only fetch to see what's available
cargit sync --fetch-only
```

### `status` - Show cache status and statistics

```bash
cargit status
```

Displays:
- Number of installed binaries
- Cache size (repos + build artifacts)
- Build statistics (time tracking)
- Binaries needing attention

### `info` - Inspect a repository

```bash
cargit info <git_url>
```

Shows available crates, workspace structure, and version information without installing.

Example:
```bash
$ cargit info https://github.com/typst/typst

Repository: https://github.com/typst/typst
────────────────────────────────────────────────────────────

Type: Cargo Workspace
Default member: crates/typst-cli

Available crates (15):
  • typst
  • typst-cli (default)
  • typst-eval
  ...

Install with: cargit install https://github.com/typst/typst <crate_name>
```

### `list` - List installed binaries

```bash
cargit list
```

### `clean` - Clean cache to free disk space

```bash
cargit clean [alias] [options]
```

Options:
- `--all`: Remove all cached repositories and build artifacts
- `--artifacts`: Clean build artifacts for a specific binary
- `--repos`: Delete repository for a specific binary
- `--orphaned`: Remove repositories not tracked in database
- `--dry-run`: Preview what would be deleted

Examples:
```bash
# Preview total cache size
cargit clean --all --dry-run

# Clean build artifacts for one binary (keeps repo)
cargit clean fd --artifacts

# Delete entire repository for a binary
cargit clean fd --repos

# Remove orphaned repositories
cargit clean --orphaned
```

When cleaning artifacts or repos, the binary is automatically copied to a safe location. Run `cargit update <alias>` to rebuild.

### `config` - Manage configuration

```bash
cargit config [options]
```

Options:
- `--show`: Display current configuration
- `--init`: Create config file with defaults
- `--edit`: Open config in `$EDITOR`

### Other Commands

```bash
# Show binary location
cargit which fd

# Rename a binary alias
cargit rename fd --to find

# Remove a binary
cargit remove fd
```

## Configuration

Configuration is stored in `~/.config/cargit/config.toml`:

```toml
[parallel]
fetch_jobs = 8      # Parallel git fetch operations
reset_jobs = 8      # Parallel git reset operations

[network]
retry_attempts = 3  # Retry attempts for network failures
retry_delay = 1.0   # Delay between retries (seconds)

[build]
track_time = true   # Track and display build times

[display]
show_progress = true
verbose = false
```

## Directory Structure

Cargit follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html):

| Path | Purpose |
|------|---------|
| `~/.cache/cargit/` | Git repositories and build artifacts |
| `~/.config/cargit/config.toml` | Configuration file |
| `~/.config/cargit/data.sqlite3` | Installation metadata database |
| `~/.local/share/cargit/` | Symlinks to installed binaries |

## Workspace Repositories

For repositories with multiple crates (Cargo workspaces), specify the crate name:

```bash
# Install typst-cli from the typst workspace
cargit install https://github.com/typst/typst typst-cli

# Inspect available crates first
cargit info https://github.com/typst/typst
```

**Note**: Multiple binaries installed from the same repository share that repository and will always be at the same commit. When you update one, all binaries from that repo are affected.

## How It Works

### Why faster than `cargo install`?

`cargo install` downloads the repository, builds it, and then **deletes everything**. Every update starts from scratch.

Cargit keeps the repository and `target/` directory:
- **First install**: Full clone + full build (same as `cargo install`)
- **Updates with no changes**: Near-instant (cache hit)
- **Updates with changes**: Only recompiles what changed

### Shallow Clones

Repositories are cloned with `--depth=1` to minimize disk usage. Only the latest commit is fetched.

### Build Cache Preservation

The `target/` directory is preserved across updates. Cargo's incremental compilation means only changed code is recompiled.

- [ ]: Set a unified target/build directory for all binaries for more efficient disk usage
