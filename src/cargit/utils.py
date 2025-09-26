from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def display_installed_table(installed: dict[str, Any]):
    """Display installed binaries in a formatted table"""
    table = Table(title="Installed Binaries")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Repo URL", style="blue")
    table.add_column("Branch", style="green")
    table.add_column("Last Commit", style="yellow")
    table.add_column("Install Dir", style="magenta")

    for name, info in installed.items():
        table.add_row(
            name,
            info["repo_url"],
            info["branch"],
            info["commit"][:8] + "...",
            info["install_dir"],
        )

    console.print(table)
