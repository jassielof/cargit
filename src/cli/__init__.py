"""CLI commands for cargit.

This module provides the command-line interface for cargit,
including install, update, sync, clean, status, and config commands.
"""

import typer

from .commands.install import install
from .commands.update import update
from .commands.list import list_binaries
from .commands.status import status
from .commands.info import info
from .commands.config_cmd import config
from .commands.rename import rename
from .commands.remove import remove
from .commands.which import which
from .commands.clean import clean
from .commands.sync import sync

app = typer.Typer(
    help="Git-based cargo binary installer with cached repositories for faster updates",
    no_args_is_help=True,
)

# Register all commands
app.command(no_args_is_help=True)(install)
app.command(no_args_is_help=True)(update)
app.command("list")(list_binaries)
app.command()(status)
app.command()(info)
app.command()(config)
app.command(no_args_is_help=True)(rename)
app.command()(remove)
app.command()(which)
app.command()(clean)
app.command()(sync)
