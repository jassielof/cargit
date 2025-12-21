"""List command for cargit."""

from rich import print as rprint

from cli.storage import load_metadata
from cli.utils import display_installed_table


def list_binaries():
    """List installed binaries"""
    metadata = load_metadata()

    if not metadata["installed"]:
        rprint("[yellow]No binaries installed[/yellow]")
        return

    display_installed_table(metadata["installed"])
