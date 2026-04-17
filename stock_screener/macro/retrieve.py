"""Theme-first hybrid retrieval: rules + tag overlap + pseudo-embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stock_screener.macro.paths import data_dir
from stock_screener.macro.prices import last_close_cad


def load_universe() -> dict[str, dict[str, Any]]:
    path = data_dir() / "macro_exposure_universe.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("ticker"):
            out[str(r["ticker"]).upper()] = r
    return out


def _pseudo_embed(text: str, dim: int = 96) -> np.ndarray:
    """Deterministic bag-of-bytes embedding for cosine ranking without external APIs."""
    v = np.zeros(dim, dtype=np.float64)
    for i, b in enumerate(text.lower().encode("utf-8", errors="ignore")[:800]):
        v[i % dim] += (float(b) - 96.0) / 128.0
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _tag_overlap_score(story: str, exposures: list[str]) -> float:
    if not exposures:
        return 0.0
    s = story.lower()
    hits = sum(1 for e in exposures if e.lower() in s or e.lower().replace("_", " ") in s)
    return min(1.0, hits / max(1, len(exposures)))


def _liquidity_score(ticker: str, min_adv: float, fx: float | None) -> float:
    px = last_close_cad(ticker, fx_usdcad=fx)
    if px is None or px <= 0:
        return 0.2
    try:
        import yfinance as yf

        h = yf.Ticker(ticker).history(period="30d", auto_adjust=True)
        if h is None or h.empty:
            return 0.3
        vol = float(h["Volume"].iloc[-5:].mean())
        adv = vol * px
        if adv >= min_adv * 2:
            return 1.0
        if adv >= min_adv:
            return 0.7
        return 0.4
    except Exception:
        return 0.4


def retrieve_for_baskets(
    *,
    story_text: str,
    basket_keys: list[str],
    min_adv_cad: float,
    fx_usdcad: float | None,
    max_names_per_basket: int = 6,
) -> list[dict[str, Any]]:
    """Return ranked name dicts across all candidate baskets."""
    baskets_path = data_dir() / "macro_baskets.json"
    baskets = {b["basket_key"]: b for b in json.loads(baskets_path.read_text(encoding="utf-8")) if isinstance(b, dict)}
    universe = load_universe()
    q = _pseudo_embed(story_text)
    results: list[dict[str, Any]] = []
    rank_global = 0

    for bk in basket_keys:
        bdef = baskets.get(bk)
        if not bdef:
            continue
        tickers = list(dict.fromkeys((bdef.get("eligible_etfs") or []) + (bdef.get("eligible_stocks") or [])))
        anchor = str(bdef.get("anchor_etf", "")).upper()
        theme_cluster = str(bdef.get("theme_cluster", ""))

        scored: list[tuple[float, dict[str, Any]]] = []
        for t in tickers:
            t = str(t).upper()
            meta = universe.get(t, {})
            exposures = meta.get("macro_exposures") if isinstance(meta.get("macro_exposures"), list) else []
            profile = json.dumps(meta, sort_keys=True) if meta else t
            emb_t = _pseudo_embed(profile + " " + t)
            emb_sim = max(0.0, _cosine(q, emb_t))
            if t == anchor:
                rule = 1.0
            elif meta.get("asset_type") == "etf":
                rule = 0.88
            else:
                rule = 0.72
            tag = _tag_overlap_score(story_text, [str(x) for x in exposures])
            liq = _liquidity_score(t, min_adv_cad, fx_usdcad)
            novelty = 0.5
            score = (
                0.45 * rule
                + 0.20 * emb_sim
                + 0.15 * tag
                + 0.15 * liq
                + 0.05 * novelty
            )
            asset_type = str(meta.get("asset_type", "stock" if len(t) <= 4 else "etf"))
            scored.append(
                (
                    score,
                    {
                        "ticker": t,
                        "asset_type": asset_type,
                        "basket_key": bk,
                        "anchor_etf": anchor,
                        "theme_cluster": theme_cluster,
                        "score": round(score, 4),
                        "match_reason": f"rule=1.0 emb={emb_sim:.2f} tag={tag:.2f} liq={liq:.2f}",
                        "rank": 0,
                        "retrieval_mode": "hybrid",
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        for i, (_, row) in enumerate(scored[:max_names_per_basket], start=1):
            row["rank"] = i
            rank_global += 1
            row["global_order"] = rank_global
            results.append(row)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
