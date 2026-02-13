"""CLI commands for cargit.

This module provides the command-line interface for cargit,
including install, update, sync, clean, status, and config commands.
"""

import typer

from .commands.clean import app as clean_app
from .commands.config_cmd import app as config_app
from .commands.info import app as info_app
from .commands.install import app as install_app
from .commands.list import app as list_app
from .commands.remove import app as remove_app
from .commands.rename import app as rename_app
from .commands.status import app as status_app
from .commands.sync import app as sync_app
from .commands.update import app as update_app
from .commands.which import app as which_app

app = typer.Typer(
    help="Git-based cargo binary installer with cached repositories for faster updates",
    no_args_is_help=True,
)

# Register all commands
app.add_typer(install_app, no_args_is_help=True)
app.add_typer(update_app, no_args_is_help=True)
app.add_typer(list_app, name="list")
app.add_typer(status_app)
app.add_typer(info_app)
app.add_typer(config_app)
app.add_typer(rename_app)
app.add_typer(remove_app)
app.add_typer(which_app)
app.add_typer(clean_app)
app.add_typer(sync_app)
