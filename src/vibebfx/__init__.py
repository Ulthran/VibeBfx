"""VibeBfx: DeepAgents-powered bioinformatics workflows."""

from importlib.metadata import version

__all__ = ["__version__"]

try:  # pragma: no cover - best effort metadata lookup
    __version__ = version("vibebfx")
except Exception:  # pragma: no cover - during editable installs
    __version__ = "0.1.0"
