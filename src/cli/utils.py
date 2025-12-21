from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def display_installed_table(installed: dict[str, Any]):
    """Display installed binaries in a formatted table"""
    table = Table(title="Installed binaries", )
    table.add_column("Alias", style="cyan", no_wrap=True)
    table.add_column("Repository URL", style="blue", highlight=True)
    table.add_column("Branch/Ref", style="green")
    table.add_column("Commit", style="yellow")
    table.add_column("Installation directory", style="magenta")

    for alias, info in installed.items():
        branch_display = info["branch"]

        # Make pinned refs more clear
        if branch_display.startswith("commit:"):
            parts = branch_display.split(":")
            if len(parts) == 3:
                branch_name = parts[1]
                commit_short = parts[2]
                branch_display = f"📌 {branch_name}@{commit_short}"
        elif branch_display.startswith("tag:"):
            tag_name = branch_display.split(":", 1)[1]
            branch_display = f"🏷️  {tag_name}"

        table.add_row(
            alias,
            info["repo_url"],
            branch_display,
            info["commit"][:8] + "...",
            info["install_dir"],
        )

    console.print(table)
