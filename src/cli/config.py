"""Configuration management for cli.

Handles loading and saving user configuration from config.toml.
Uses caching for fast repeated access.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from .core import CONFIG_DIR

CONFIG_FILE = CONFIG_DIR / "config.toml"

# Default configuration values
DEFAULTS: dict[str, Any] = {
    "parallel": {
        "fetch_jobs": 8,      # Number of parallel git fetch operations
        "reset_jobs": 8,      # Number of parallel git reset operations
    },
    "network": {
        "retry_attempts": 3,  # Number of retry attempts for network operations
        "retry_delay": 1.0,   # Delay between retries in seconds
    },
    "build": {
        "track_time": True,   # Track and display build times
    },
    "display": {
        "show_progress": True,  # Show progress bars
        "verbose": False,       # Verbose output (show git/cargo output)
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def _get_config_mtime() -> float | None:
    """Get config file modification time for cache invalidation."""
    try:
        return CONFIG_FILE.stat().st_mtime if CONFIG_FILE.exists() else None
    except OSError:
        return None


_cached_config: dict[str, Any] | None = None
_cached_mtime: float | None = None


def load_config() -> dict[str, Any]:
    """Load configuration from config.toml, merged with defaults.

    Uses caching - config is only re-read if the file has been modified.
    """
    global _cached_config, _cached_mtime

    # Check if cache is still valid
    current_mtime = _get_config_mtime.__wrapped__()  # Bypass lru_cache

    if _cached_config is not None and _cached_mtime == current_mtime:
        return _cached_config

    config = DEFAULTS.copy()

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                user_config = tomllib.load(f)
            config = _deep_merge(DEFAULTS, user_config)
        except Exception:
            # If config file is malformed, use defaults
            pass

    # Update cache
    _cached_config = config
    _cached_mtime = current_mtime

    return config


def invalidate_config_cache() -> None:
    """Invalidate the config cache (call after saving config)."""
    global _cached_config, _cached_mtime
    _cached_config = None
    _cached_mtime = None
    _get_config_mtime.cache_clear()


def save_config(config: dict[str, Any]) -> bool:
    """Save configuration to config.toml."""
    try:
        import tomlkit

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Create a clean TOML document
        doc = tomlkit.document()

        # Add header comment
        doc.add(tomlkit.comment("Cargit configuration file"))
        doc.add(tomlkit.comment("See: https://github.com/jassielof/cargit"))
        doc.add(tomlkit.nl())

        for section, values in config.items():
            if isinstance(values, dict):
                table = tomlkit.table()
                for key, value in values.items():
                    table.add(key, value)
                doc.add(section, table)
            else:
                doc.add(section, values)

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

        # Invalidate cache after saving
        invalidate_config_cache()

        return True
    except Exception:
        return False


@lru_cache(maxsize=32)
def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get a specific config value by dot-separated path.

    Example: get_config_value("parallel.fetch_jobs", 8)

    Results are cached for performance.
    """
    config = load_config()
    keys = key_path.split(".")

    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def init_config() -> Path:
    """Initialize config file with defaults if it doesn't exist.

    Returns path to config file.
    """
    if not CONFIG_FILE.exists():
        save_config(DEFAULTS)
    return CONFIG_FILE


# Convenience functions for common config values (cached via get_config_value)
def get_fetch_jobs() -> int:
    """Get number of parallel fetch jobs."""
    return get_config_value("parallel.fetch_jobs", DEFAULTS["parallel"]["fetch_jobs"])


def get_retry_attempts() -> int:
    """Get number of retry attempts for network operations."""
    return get_config_value("network.retry_attempts", DEFAULTS["network"]["retry_attempts"])


def get_retry_delay() -> float:
    """Get delay between retry attempts."""
    return get_config_value("network.retry_delay", DEFAULTS["network"]["retry_delay"])


def should_track_build_time() -> bool:
    """Check if build time tracking is enabled."""
    return get_config_value("build.track_time", DEFAULTS["build"]["track_time"])


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return get_config_value("display.verbose", DEFAULTS["display"]["verbose"])
