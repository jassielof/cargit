"""Config command for cargit."""

import os
import sys

import typer
from rich import print as rprint

from cli.config import CONFIG_FILE, init_config, load_config

app = typer.Typer()


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    init: bool = typer.Option(
        False, "--init", help="Initialize config file with defaults"
    ),
    edit: bool = typer.Option(False, "--edit", help="Open config file in editor"),
):
    """Manage cargit configuration.

    Examples:
        cargit config --show     # Show current configuration
        cargit config --init     # Create config file with defaults
        cargit config --edit     # Open config in $EDITOR
    """
    if not any([show, init, edit]):
        rprint("[red]Error: Must specify one of --show, --init, or --edit[/red]")
        sys.exit(1)

    if show:
        config_data = load_config()
        rprint("[bold cyan]Current Configuration[/bold cyan]")
        rprint(f"[dim]Config file: {CONFIG_FILE}[/dim]")
        rprint(f"[dim]{'─' * 50}[/dim]\n")

        for section, values in config_data.items():
            rprint(f"[bold yellow]\\[{section}][/bold yellow]")
            if isinstance(values, dict):
                for key, value in values.items():
                    rprint(f"  [cyan]{key}[/cyan] = {value}")
            else:
                rprint(f"  {values}")
            rprint()

    elif init:
        config_path = init_config()
        rprint(f"[green]Configuration file created at {config_path}[/green]")

    elif edit:
        if not CONFIG_FILE.exists():
            init_config()
            rprint(f"[blue]Created default config at {CONFIG_FILE}[/blue]")

        editor = os.environ.get("EDITOR", "nano")
        os.system(f"{editor} {CONFIG_FILE}")
