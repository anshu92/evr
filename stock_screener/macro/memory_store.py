"""SQLite durable memory for macro stories, themes, surfaced names, and outcomes."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def connect(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS stories (
            story_id TEXT PRIMARY KEY,
            story_hash TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            summary_short TEXT,
            source_type TEXT,
            source_name TEXT,
            source_url TEXT,
            published_utc TEXT,
            ingested_utc TEXT NOT NULL,
            reliability_tier TEXT,
            event_type TEXT,
            event_key TEXT,
            novelty_score REAL,
            staleness_hours REAL,
            is_scheduled_event INTEGER DEFAULT 0,
            surprise_direction TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_stories_hash ON stories(story_hash);

        CREATE TABLE IF NOT EXISTS themes (
            theme_id TEXT PRIMARY KEY,
            theme_key TEXT NOT NULL,
            direction TEXT,
            horizon TEXT,
            confidence REAL,
            watch_until_utc TEXT,
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL,
            source_count INTEGER DEFAULT 1,
            status TEXT DEFAULT 'active',
            source_mix TEXT,
            event_window TEXT,
            surprise_direction TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_themes_key ON themes(theme_key, status);

        CREATE TABLE IF NOT EXISTS story_theme_links (
            story_id TEXT NOT NULL,
            theme_id TEXT NOT NULL,
            link_confidence REAL,
            link_reason TEXT,
            PRIMARY KEY (story_id, theme_id),
            FOREIGN KEY (story_id) REFERENCES stories(story_id),
            FOREIGN KEY (theme_id) REFERENCES themes(theme_id)
        );

        CREATE TABLE IF NOT EXISTS surfaced_names (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            theme_id TEXT NOT NULL,
            as_of_utc TEXT NOT NULL,
            ticker TEXT NOT NULL,
            asset_type TEXT,
            basket_key TEXT,
            rank INTEGER,
            score REAL,
            match_reason TEXT,
            anchor_etf TEXT,
            retrieval_mode TEXT,
            theme_cluster TEXT,
            portfolio_action TEXT,
            FOREIGN KEY (theme_id) REFERENCES themes(theme_id)
        );
        CREATE INDEX IF NOT EXISTS idx_surfaced_theme ON surfaced_names(theme_id, as_of_utc, rank);

        CREATE TABLE IF NOT EXISTS theme_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            theme_id TEXT,
            ticker TEXT NOT NULL,
            entry_utc TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            return_abs REAL,
            return_vs_anchor REAL,
            return_vs_spy REAL,
            return_vs_cash REAL,
            event_type TEXT,
            theme_cluster TEXT,
            FOREIGN KEY (theme_id) REFERENCES themes(theme_id)
        );
        CREATE INDEX IF NOT EXISTS idx_outcomes ON theme_outcomes(theme_id, ticker, horizon_days);
        """
    )
    conn.commit()


def insert_story(conn: sqlite3.Connection, row: dict[str, Any]) -> str | None:
    """Insert story if new. Returns story_id or None if duplicate hash."""
    story_hash = row["story_hash"]
    cur = conn.execute("SELECT story_id FROM stories WHERE story_hash = ?", (story_hash,))
    existing = cur.fetchone()
    if existing:
        return str(existing[0])

    story_id = story_hash
    conn.execute(
        """
        INSERT INTO stories (
            story_id, story_hash, title, summary_short, source_type, source_name, source_url,
            published_utc, ingested_utc, reliability_tier, event_type, event_key,
            novelty_score, staleness_hours, is_scheduled_event, surprise_direction
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            story_id,
            story_hash,
            row.get("title", ""),
            row.get("summary_short", ""),
            row.get("source_type", ""),
            row.get("source_name", ""),
            row.get("source_url", ""),
            row.get("published_utc"),
            row.get("ingested_utc") or _utc_iso(),
            row.get("reliability_tier", "tier2"),
            row.get("event_type"),
            row.get("event_key"),
            row.get("novelty_score"),
            row.get("staleness_hours"),
            1 if row.get("is_scheduled_event") else 0,
            row.get("surprise_direction"),
        ),
    )
    conn.commit()
    return story_id


def upsert_theme(conn: sqlite3.Connection, row: dict[str, Any]) -> str:
    theme_id = str(row["theme_id"])
    now = _utc_iso()
    cur = conn.execute("SELECT theme_id FROM themes WHERE theme_id = ?", (theme_id,))
    if cur.fetchone():
        conn.execute(
            """
            UPDATE themes SET
                last_seen_utc = ?,
                source_count = source_count + 1,
                confidence = COALESCE(?, confidence),
                watch_until_utc = COALESCE(?, watch_until_utc)
            WHERE theme_id = ?
            """,
            (now, row.get("confidence"), row.get("watch_until_utc"), theme_id),
        )
    else:
        conn.execute(
            """
            INSERT INTO themes (
                theme_id, theme_key, direction, horizon, confidence, watch_until_utc,
                first_seen_utc, last_seen_utc, source_count, status, source_mix, event_window, surprise_direction
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                theme_id,
                row.get("theme_key", ""),
                row.get("direction"),
                row.get("horizon"),
                row.get("confidence", 0.5),
                row.get("watch_until_utc"),
                now,
                now,
                1,
                row.get("status", "active"),
                row.get("source_mix"),
                row.get("event_window"),
                row.get("surprise_direction"),
            ),
        )
    conn.commit()
    return theme_id


def link_story_theme(
    conn: sqlite3.Connection,
    story_id: str,
    theme_id: str,
    link_confidence: float,
    link_reason: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO story_theme_links (story_id, theme_id, link_confidence, link_reason)
        VALUES (?,?,?,?)
        """,
        (story_id, theme_id, link_confidence, link_reason),
    )
    conn.commit()


def insert_surfaced_batch(conn: sqlite3.Connection, theme_id: str, rows: list[dict[str, Any]]) -> None:
    as_of = _utc_iso()
    for r in rows:
        conn.execute(
            """
            INSERT INTO surfaced_names (
                theme_id, as_of_utc, ticker, asset_type, basket_key, rank, score,
                match_reason, anchor_etf, retrieval_mode, theme_cluster, portfolio_action
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                theme_id,
                as_of,
                r.get("ticker", ""),
                r.get("asset_type", ""),
                r.get("basket_key", ""),
                int(r.get("rank", 0)),
                float(r.get("score", 0.0)),
                r.get("match_reason", ""),
                r.get("anchor_etf", ""),
                r.get("retrieval_mode", "hybrid"),
                r.get("theme_cluster", ""),
                r.get("portfolio_action", ""),
            ),
        )
    conn.commit()


def export_json_snapshot(conn: sqlite3.Connection, path: Path) -> None:
    """Compact JSON export for artifacts."""
    out: dict[str, Any] = {"exported_utc": _utc_iso(), "stories": [], "themes": [], "surfaced": []}
    for row in conn.execute("SELECT * FROM stories ORDER BY ingested_utc DESC LIMIT 200"):
        out["stories"].append(dict(row))
    for row in conn.execute("SELECT * FROM themes ORDER BY last_seen_utc DESC LIMIT 100"):
        out["themes"].append(dict(row))
    for row in conn.execute("SELECT * FROM surfaced_names ORDER BY id DESC LIMIT 200"):
        out["surfaced"].append(dict(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def recent_stories(conn: sqlite3.Connection, limit: int = 500) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM stories ORDER BY datetime(COALESCE(published_utc, ingested_utc)) DESC LIMIT ?",
        (limit,),
    )
    return list(cur.fetchall())
