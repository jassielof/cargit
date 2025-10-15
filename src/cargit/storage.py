import json
from typing import Any

from rich import print as rprint

from .core import CARGIT_DIR

METADATA_FILE = CARGIT_DIR / "metadata.toml"
# TODO: Make the git repo more efficient by using bare clone or another faster clone, and on every pull or update, also try to remove/reset to the latest commit to avoid storing unnecessary history.


def load_metadata() -> dict[str, Any]:
    """Load metadata from file"""
    if not METADATA_FILE.exists():
        return {"installed": {}}

    # Try TOML first
    if METADATA_FILE.suffix == ".toml":
        try:
            # For Python >= 3.11
            import tomllib

            with open(METADATA_FILE, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            # For Python < 3.11
            import tomlkit

            with open(METADATA_FILE, "r") as f:
                return tomlkit.load(f)
        except Exception as e:
            rprint(f"[red]Error loading TOML metadata: {e}[/red]")
            return {"installed": {}}

    # Fallback to JSON
    json_file = CARGIT_DIR / "metadata.json"
    if json_file.exists():
        try:
            with open(json_file, "r") as f:
                return json.load(f)
        except Exception as e:
            rprint(f"[red]Error loading JSON metadata: {e}[/red]")
            return {"installed": {}}

    return {"installed": {}}


def save_metadata(data: dict[str, Any]):
    """Save metadata to file"""
    try:
        import tomlkit

        with open(METADATA_FILE, "w") as f:
            tomlkit.dump(data, f)
    except ImportError:
        # Fallback to JSON if tomlkit not available
        json_file = CARGIT_DIR / "metadata.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        rprint(
            "[yellow]Warning: Using JSON format. Install tomlkit for TOML support[/yellow]"
        )
