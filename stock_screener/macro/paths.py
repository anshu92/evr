from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repository root (parent of `stock_screener/`)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def default_memory_sqlite() -> Path:
    return repo_root() / "data_runtime" / "macro_memory.sqlite"
