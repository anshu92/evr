from __future__ import annotations

from pathlib import Path

from stock_screener.macro.classify import classify_article
from stock_screener.macro.memory_store import connect, init_schema, insert_story
from stock_screener.macro.config import MacroConfig
from stock_screener.macro.macro_llm import refine_macro_targets_with_llm
from stock_screener.macro.retrieve import retrieve_for_baskets
from stock_screener.reporting.email_style import html_escape


def test_html_escape():
    assert "&amp;" in html_escape("a & b")
    assert "<" not in html_escape("<script>")


def test_classify_article_oil():
    art = {
        "title": "Oil jumps as OPEC signals supply cut",
        "summary": "",
        "publish_date": "2026-01-15T12:00:00+00:00",
        "story_hash": "abc",
    }
    cl = classify_article(art, prior_titles=set())
    assert "oil_up_energy" in cl["basket_keys"]
    assert cl["theme_id"]


def test_retrieve_for_baskets_deterministic(monkeypatch):
    monkeypatch.setattr(
        "stock_screener.macro.retrieve.last_close_cad",
        lambda *a, **k: 100.0,
    )
    rows = retrieve_for_baskets(
        story_text="crude oil prices surge energy stocks",
        basket_keys=["oil_up_energy"],
        min_adv_cad=1.0,
        fx_usdcad=1.35,
        max_names_per_basket=4,
    )
    assert rows
    assert rows[0]["ticker"] in {"XLE", "XOM", "USO", "CVX", "COP", "SLB"}


def test_refine_macro_llm_disabled(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MACRO_LLM_AGENT_ENABLED", raising=False)
    monkeypatch.delenv("LLM_AGENT_ENABLED", raising=False)
    monkeypatch.setenv("MACRO_PORTFOLIO_STATE_PATH", str(tmp_path / "mps.json"))
    cfg = MacroConfig.from_env()
    ranked = [
        {
            "ticker": "XLE",
            "score": 0.9,
            "basket_key": "oil_up_energy",
            "theme_cluster": "energy",
            "asset_type": "etf",
            "match_reason": "rule",
        }
    ]
    rule_targets = [
        {
            "ticker": "XLE",
            "weight": 0.2,
            "basket_key": "oil_up_energy",
            "theme_cluster": "energy",
            "score": 0.9,
            "asset_type": "etf",
        }
    ]
    t, meta, r = refine_macro_targets_with_llm(
        cfg=cfg,
        all_ranked=ranked,
        articles=[{"title": "Oil rises", "publisher": "test"}],
        primary_theme_key="oil_up_energy",
        rule_targets=rule_targets,
        log=None,
    )
    assert meta["status"] == "disabled"
    assert t == rule_targets
    assert r == ranked


def test_refine_macro_llm_skipped_without_agent_key(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MACRO_LLM_AGENT_ENABLED", "1")
    monkeypatch.setenv("MACRO_PORTFOLIO_STATE_PATH", str(tmp_path / "mps2.json"))
    monkeypatch.setattr(
        "stock_screener.agents.config.get_agent_config",
        lambda: {"api_key": ""},
    )
    cfg = MacroConfig.from_env()
    assert cfg.llm_agent_enabled is True
    ranked = [
        {
            "ticker": "XLF",
            "score": 0.85,
            "basket_key": "rates_up_banks",
            "theme_cluster": "rates",
            "asset_type": "etf",
            "match_reason": "rule",
        }
    ]
    rule_targets = [dict(ranked[0], weight=0.15)]
    t, meta, _ = refine_macro_targets_with_llm(
        cfg=cfg,
        all_ranked=ranked,
        articles=[{"title": "Rates", "publisher": "t"}],
        primary_theme_key="rates",
        rule_targets=rule_targets,
        log=None,
    )
    assert meta["status"] == "skipped"
    assert t == rule_targets


def test_memory_sqlite_roundtrip(tmp_path: Path):
    db = tmp_path / "m.sqlite"
    conn = connect(db)
    init_schema(conn)
    sid = insert_story(
        conn,
        {
            "story_hash": "h1",
            "title": "Test headline",
            "summary_short": "sum",
            "source_type": "test",
            "source_name": "t",
            "source_url": "https://example.com",
            "published_utc": None,
            "ingested_utc": "2026-01-01T00:00:00+00:00",
            "reliability_tier": "tier2",
        },
    )
    assert sid
    cur = conn.execute("SELECT COUNT(*) AS c FROM stories")
    assert int(cur.fetchone()["c"]) == 1
    conn.close()
