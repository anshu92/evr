from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable


TICKER_ALLOWED_RE = re.compile(r"^[A-Z0-9.\-^=]{1,20}$")


def get_logger(name: str = "stock_screener", level: str | int = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_ticker(raw: str) -> str | None:
    """Validate/sanitize external symbols before using in logic/outputs."""

    if raw is None:
        return None
    t = raw.strip().upper()
    if not t:
        return None
    if not TICKER_ALLOWED_RE.fullmatch(t):
        return None
    return t


def stable_hash(items: Iterable[str]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


@dataclass(frozen=True)
class Universe:
    tickers: list[str]
    meta: dict[str, Any]


