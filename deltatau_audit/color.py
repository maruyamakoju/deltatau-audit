"""ANSI color helpers for terminal output.

Auto-disabled when:
  - NO_COLOR env var is set (https://no-color.org/)
  - TERM=dumb
  - stdout is not a TTY (unless FORCE_COLOR is set)

Colors are preserved in GitHub Actions (supports ANSI) and most modern terminals.
"""

import os
import sys


def _supports_color() -> bool:
    """Detect if the current stdout supports ANSI color codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


# ANSI escape code table
_C = {
    "reset":          "\033[0m",
    "bold":           "\033[1m",
    "green":          "\033[32m",
    "bright_green":   "\033[92m",
    "yellow":         "\033[33m",
    "bright_yellow":  "\033[93m",
    "red":            "\033[31m",
    "bright_red":     "\033[91m",
    "gray":           "\033[90m",
    "cyan":           "\033[36m",
    "white":          "\033[97m",
}


def colorize(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes. Returns plain text if colors not supported."""
    if not _supports_color():
        return text
    prefix = "".join(_C.get(c, "") for c in codes)
    return f"{prefix}{text}{_C['reset']}"


def _rj(text: str, width: int) -> str:
    """Right-justify text to width (using actual text length, not display width)."""
    return " " * max(0, width - len(text)) + text


# ── Rating-specific helpers ──────────────────────────────────────────

_RATING_CODES = {
    "PASS":     ("bright_green",),
    "MILD":     ("green",),
    "DEGRADED": ("bright_yellow",),
    "FAIL":     ("bright_red", "bold"),
    "N/A":      ("gray",),
    "UNKNOWN":  ("gray",),
}


def rating_codes(rating: str) -> tuple:
    """Return ANSI codes for a robustness rating string."""
    return _RATING_CODES.get(rating, ("reset",))


def colored_rating(rating: str, width: int = 0) -> str:
    """Return colored + optionally right-justified rating string."""
    text = _rj(rating, width) if width else rating
    return colorize(text, *rating_codes(rating))


def section_header(text: str) -> str:
    """Bold cyan section header."""
    return colorize(text, "cyan", "bold")


def dim(text: str) -> str:
    """Dimmed (gray) text."""
    return colorize(text, "gray")


def bold(text: str) -> str:
    """Bold text."""
    return colorize(text, "bold")


def ok(text: str) -> str:
    """Bright green text (for positive values, improvements)."""
    return colorize(text, "bright_green")


def warn(text: str) -> str:
    """Yellow text (for warnings)."""
    return colorize(text, "bright_yellow")


def err(text: str) -> str:
    """Bright red text (for errors, degraded values)."""
    return colorize(text, "bright_red")
