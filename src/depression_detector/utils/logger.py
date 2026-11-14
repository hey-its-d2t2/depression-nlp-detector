"""Centralized structured logging using loguru."""

import sys
from pathlib import Path
from loguru import logger as _logger


def setup_logger(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure application-wide logger with console and optional file sinks."""
    _logger.remove()

    fmt_console = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    fmt_file = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}"
    )

    _logger.add(
        sys.stdout,
        level=log_level,
        format=fmt_console,
        colorize=True,
        enqueue=True,
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_file,
            level=log_level,
            format=fmt_file,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            enqueue=True,
        )


def get_logger(name: str):
    """Return a context-bound logger for the given module name."""
    return _logger.bind(name=name)
