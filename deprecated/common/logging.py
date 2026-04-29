"""Coloured logging for the project.

Provides ``get_logger`` which returns a standard ``logging.Logger`` with
ANSI-coloured level prefixes so that warnings and errors are visible in
terminal output.
"""

import logging
import sys

_RESET = "\033[0m"
_COLOURS = {
    logging.DEBUG: "\033[36m",      # cyan
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33;1m",  # bold yellow
    logging.ERROR: "\033[31;1m",    # bold red
    logging.CRITICAL: "\033[41;1m", # bold white on red
}


class _ColouredFormatter(logging.Formatter):
    """Formatter that wraps the level name in ANSI colour codes."""

    _base_fmt = "%(levelname)s | %(name)s | %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelno, "")
        fmt = f"{colour}%(levelname)s{_RESET} | %(name)s | %(message)s"
        self._style._fmt = fmt
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a coloured logger.  Safe to call multiple times for the same name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_ColouredFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
