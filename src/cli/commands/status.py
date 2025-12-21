"""Status command for cargit."""

from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from cli.core import (
    CACHE_DIR,
    DATA_DIR,
    ensure_dirs,
    get_cache_size,
    format_size,
    get_repo_path,
)
from cli.storage import get_status_summary, get_all_binaries_full


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
    console = Console()

    ensure_dirs()

    summary, binaries, repo_sizes, total_cache_size, bin_dir_size = _collect_status_data()

    _render_status_header()
    _render_summary_panel(console, summary, total_cache_size, bin_dir_size)
    _render_attention(summary)
    _render_binaries_table(console, binaries, repo_sizes)
    _render_status_paths()


def _collect_status_data():
    summary = get_status_summary()
    binaries = get_all_binaries_full()

    total_cache_size = get_cache_size(CACHE_DIR) if CACHE_DIR.exists() else 0
    bin_dir_size = get_cache_size(DATA_DIR) if DATA_DIR.exists() else 0

    repo_sizes: dict[str, int] = {}
    for binary in binaries:
        repo_path = get_repo_path(binary["repo_url"])
        if repo_path.exists() and binary["repo_url"] not in repo_sizes:
            repo_sizes[binary["repo_url"]] = get_cache_size(repo_path)

    return summary, binaries, repo_sizes, total_cache_size, bin_dir_size


def _render_status_header():
    rprint("\n[bold cyan]Cargit Status[/bold cyan]")
    rprint(f"{'─' * 50}\n")


def _render_summary_panel(console, summary, total_cache_size, bin_dir_size):
    summary_text = (
        f"[cyan]Installed binaries:[/cyan] {summary['total_binaries']}\n"
        f"[cyan]Unique repositories:[/cyan] {summary['unique_repos']}\n"
        f"[cyan]Total cache size:[/cyan] {format_size(total_cache_size)}\n"
        f"[cyan]Binary links size:[/cyan] {format_size(bin_dir_size)}"
    )

    if summary["total_builds"] > 0:
        avg_time = summary["overall_avg_build_time"]
        if avg_time:
            avg_str = f"{int(avg_time // 60)}m {int(avg_time % 60)}s" if avg_time >= 60 else f"{avg_time:.1f}s"
            summary_text += f"\n[cyan]Total builds:[/cyan] {summary['total_builds']}"
            summary_text += f"\n[cyan]Avg build time:[/cyan] {avg_str}"

    console.print(Panel(summary_text, title="Summary", border_style="blue"))


def _render_attention(summary):
    needs_attention = []
    if summary["repos_deleted"] > 0:
        needs_attention.append(
            f"[yellow]{summary['repos_deleted']} binary(ies) with deleted repos[/yellow]"
        )
    if summary["artifacts_cleaned"] > 0:
        needs_attention.append(
            f"[yellow]{summary['artifacts_cleaned']} binary(ies) with cleaned artifacts[/yellow]"
        )

    if not needs_attention:
        return

    rprint("\n[bold yellow]⚠ Attention Needed[/bold yellow]")
    for item in needs_attention:
        rprint(f"  • {item}")
    rprint("[dim]Run 'cargit sync' to rebuild affected binaries[/dim]")


def _render_binaries_table(console, binaries, repo_sizes):
    if not binaries:
        return

    rprint("\n[bold]Installed Binaries[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Alias", style="cyan")
    table.add_column("Branch/Ref", style="green")
    table.add_column("Repo Size", justify="right")
    table.add_column("Last Build", justify="right")
    table.add_column("Status")

    for binary in binaries:
        branch_display = _format_branch_display(binary["branch"])
        size_str = _format_repo_size(binary, repo_sizes)
        build_str = _format_build_time(binary)
        status_str = _format_binary_status(binary)

        table.add_row(
            binary["alias"],
            branch_display,
            size_str,
            build_str,
            status_str,
        )

    console.print(table)


def _format_branch_display(branch: str) -> str:
    if branch.startswith("commit:"):
        parts = branch.split(":")
        if len(parts) >= 3:
            return f"📌 {parts[1]}@{parts[2]}"
    if branch.startswith("tag:"):
        return f"🏷️  {branch.split(':', 1)[1]}"
    return branch


def _format_repo_size(binary: dict, repo_sizes: dict[str, int]) -> str:
    repo_size = repo_sizes.get(binary["repo_url"], 0)
    return format_size(repo_size) if repo_size > 0 else "[dim]N/A[/dim]"


def _format_build_time(binary: dict) -> str:
    duration = binary.get("last_build_duration")
    if not duration:
        return "[dim]—[/dim]"
    if duration >= 60:
        return f"{int(duration // 60)}m {int(duration % 60)}s"
    return f"{duration:.1f}s"


def _format_binary_status(binary: dict) -> str:
    if binary.get("repo_deleted"):
        return "[red]repo deleted[/red]"
    if binary.get("artifacts_cleaned"):
        return "[yellow]needs rebuild[/yellow]"
    return "[green]✓[/green]"


def _render_status_paths():
    rprint(f"\n[dim]Cache: {CACHE_DIR}[/dim]")
    rprint(f"[dim]Binaries: {DATA_DIR}[/dim]")
