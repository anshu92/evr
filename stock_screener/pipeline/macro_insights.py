"""Macro news + hybrid retrieval + standalone macro portfolio pipeline."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stock_screener.macro.classify import classify_article
from stock_screener.macro.config import MacroConfig
from stock_screener.macro.evaluate import update_theme_outcomes
from stock_screener.macro.macro_sources import fetch_macro_headlines
from stock_screener.macro.memory_store import connect, export_json_snapshot, init_schema, insert_surfaced_batch, insert_story, link_story_theme, upsert_theme
from stock_screener.macro.paths import default_memory_sqlite, repo_root
from stock_screener.macro.macro_llm import refine_macro_targets_with_llm
from stock_screener.macro.portfolio_engine import apply_macro_trades, build_target_weights
from stock_screener.macro.retrieve import retrieve_for_baskets
from stock_screener.reporting.macro_render import render_macro_reports


def _started_ok(started: float, cfg: MacroConfig) -> bool:
    return (time.time() - started) / 60.0 < cfg.max_runtime_minutes


def run_macro_insights(cfg: MacroConfig | None = None, logger: Any | None = None) -> None:
    """End-to-end macro run: ingest, classify, memory, retrieve, portfolio, render."""
    import logging

    log = logger or logging.getLogger(__name__)
    cfg = cfg or MacroConfig.from_env()
    started = time.time()
    reports_dir = Path(cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mem_path = Path(cfg.memory_sqlite_path)
    if not mem_path.is_absolute():
        mem_path = repo_root() / mem_path
    conn = connect(mem_path)
    init_schema(conn)

    if not _started_ok(started, cfg):
        log.warning("Macro pipeline: runtime budget exceeded before start")
        return

    articles = fetch_macro_headlines(max_per_source=18, max_age_days=3)
    log.info("Macro ingest: %d headlines", len(articles))

    prior_path = cache_dir / "macro_prior_titles.json"
    prior_titles: set[str] = set()
    if prior_path.exists():
        try:
            prior_titles = set(json.loads(prior_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, OSError):
            prior_titles = set()

    merged_story_text: list[str] = []
    all_ranked: list[dict[str, Any]] = []
    primary_theme_key = "macro_mixed"

    for art in articles[:40]:
        if not _started_ok(started, cfg):
            break
        row = {
            "story_hash": art["story_hash"],
            "title": art["title"],
            "summary_short": art.get("summary", "")[:200] or art["title"][:200],
            "source_type": art.get("source_type", ""),
            "source_name": art.get("publisher", ""),
            "source_url": art.get("link", ""),
            "published_utc": art.get("publish_date"),
            "ingested_utc": datetime.now(tz=timezone.utc).isoformat(),
            "reliability_tier": art.get("reliability_tier", "tier2"),
        }
        story_id = insert_story(conn, row)
        if not story_id:
            continue

        cl = classify_article(art, prior_titles=prior_titles)
        primary_theme_key = str(cl.get("theme_key", primary_theme_key))
        merged_story_text.append(cl["summary_short"])
        theme_row = {
            "theme_id": cl["theme_id"],
            "theme_key": cl["theme_key"],
            "direction": cl.get("direction"),
            "horizon": cl.get("horizon"),
            "confidence": cl.get("confidence"),
            "watch_until_utc": cl.get("watch_until_utc"),
            "surprise_direction": cl.get("surprise_direction"),
        }
        upsert_theme(conn, theme_row)
        link_story_theme(conn, story_id, cl["theme_id"], float(cl.get("confidence", 0.5)), "rules+v1")

        story_row_update = {
            "story_hash": art["story_hash"],
            "title": art["title"],
            "summary_short": cl["summary_short"],
            "source_type": art.get("source_type", ""),
            "source_name": art.get("publisher", ""),
            "source_url": art.get("link", ""),
            "published_utc": art.get("publish_date"),
            "ingested_utc": datetime.now(tz=timezone.utc).isoformat(),
            "reliability_tier": art.get("reliability_tier", "tier2"),
            "event_type": cl.get("event_type"),
            "event_key": cl.get("event_key"),
            "novelty_score": cl.get("novelty_score"),
            "staleness_hours": cl.get("staleness_hours"),
            "is_scheduled_event": cl.get("is_scheduled_event"),
            "surprise_direction": cl.get("surprise_direction"),
        }
        conn.execute(
            """
            UPDATE stories SET summary_short=?, event_type=?, event_key=?, novelty_score=?,
            staleness_hours=?, is_scheduled_event=?, surprise_direction=?
            WHERE story_id = ?
            """,
            (
                story_row_update["summary_short"],
                story_row_update["event_type"],
                story_row_update["event_key"],
                story_row_update["novelty_score"],
                story_row_update["staleness_hours"],
                1 if story_row_update["is_scheduled_event"] else 0,
                story_row_update["surprise_direction"],
                story_id,
            ),
        )
        conn.commit()

        ranked = retrieve_for_baskets(
            story_text=cl["summary_short"] + " " + art["title"],
            basket_keys=cl.get("basket_keys") or [],
            min_adv_cad=cfg.min_adv_cad,
            fx_usdcad=None,
            max_names_per_basket=5,
        )
        for r in ranked[:8]:
            r = dict(r)
            r["portfolio_action"] = "candidate"
            all_ranked.append(r)
        insert_surfaced_batch(conn, cl["theme_id"], ranked[:8])

    prior_titles.update(a["title"].lower()[:120] for a in articles[:40])
    prior_path.write_text(json.dumps(sorted(prior_titles)[-400:]), encoding="utf-8")

    all_ranked.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    rule_targets = build_target_weights(ranked=all_ranked, cfg=cfg)
    targets, llm_agent_meta, ranked_for_reports = refine_macro_targets_with_llm(
        cfg=cfg,
        all_ranked=all_ranked,
        articles=articles,
        primary_theme_key=primary_theme_key,
        rule_targets=rule_targets,
        log=log,
    )
    log.info("Macro targets: %s", [t["ticker"] for t in targets])

    state, trade_actions = apply_macro_trades(cfg=cfg, targets=targets, theme_key=primary_theme_key, logger=log)

    try:
        update_theme_outcomes(conn)
    except Exception as e:
        log.debug("theme outcomes update skipped: %s", e)

    export_json_snapshot(conn, reports_dir / "macro_memory_export.json")

    render_macro_reports(
        reports_dir=reports_dir,
        articles=articles,
        ranked=ranked_for_reports,
        targets=targets,
        trade_actions=trade_actions,
        portfolio_state=state,
        memory_path=str(mem_path),
        run_utc=datetime.now(tz=timezone.utc).isoformat(),
        logger=log,
        llm_agent=llm_agent_meta,
    )

    meta = {
        "run_utc": datetime.now(tz=timezone.utc).isoformat(),
        "n_headlines": len(articles),
        "n_ranked": len(ranked_for_reports),
        "targets": targets,
        "llm_agent": llm_agent_meta,
    }
    (cache_dir / "last_macro_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    conn.close()
    log.info("Macro pipeline finished")
