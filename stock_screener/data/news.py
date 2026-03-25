"""Fetch news articles and compute sentiment features per ticker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False


def _get_vader() -> Any:
    if not _VADER_AVAILABLE:
        return None
    return SentimentIntensityAnalyzer()


def fetch_ticker_news(ticker: str, logger=None) -> list[dict]:
    """Fetch recent news articles for a single ticker via yfinance.

    Returns list of dicts with keys: title, publisher, publish_date, link.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        raw = t.news or []
        articles = []
        for item in raw:
            content = item.get("content") or item
            title = (
                content.get("title")
                or item.get("title")
                or ""
            )
            pub_date = content.get("pubDate") or item.get("providerPublishTime")
            publisher = ""
            if isinstance(content.get("provider"), dict):
                publisher = content["provider"].get("displayName", "")
            elif isinstance(item.get("publisher"), str):
                publisher = item["publisher"]

            if not title:
                continue
            articles.append({
                "title": str(title),
                "publisher": str(publisher),
                "publish_date": str(pub_date) if pub_date else None,
                "link": content.get("canonicalUrl", {}).get("url") or item.get("link", ""),
            })
        return articles
    except Exception as e:
        if logger:
            logger.warning("Failed to fetch news for %s: %s", ticker, e)
        return []


def compute_news_sentiment(articles: list[dict]) -> dict[str, float]:
    """Score a list of articles using VADER and return aggregate metrics.

    Returns dict with keys:
      - sentiment_avg: mean compound score (-1 to +1)
      - sentiment_pos_ratio: fraction of articles with positive sentiment
      - article_count: number of articles scored
    """
    vader = _get_vader()
    if vader is None or not articles:
        return {
            "sentiment_avg": float("nan"),
            "sentiment_pos_ratio": float("nan"),
            "article_count": 0,
        }

    scores = []
    for a in articles:
        title = a.get("title", "")
        if not title:
            continue
        vs = vader.polarity_scores(title)
        scores.append(vs["compound"])

    if not scores:
        return {
            "sentiment_avg": float("nan"),
            "sentiment_pos_ratio": float("nan"),
            "article_count": 0,
        }

    import numpy as np
    return {
        "sentiment_avg": float(np.mean(scores)),
        "sentiment_pos_ratio": float(np.mean([1.0 if s > 0.05 else 0.0 for s in scores])),
        "article_count": len(scores),
    }


def fetch_news_sentiment_features(
    tickers: list[str],
    cache_dir: Path | None = None,
    cache_ttl_hours: int = 24,
    logger=None,
) -> pd.DataFrame:
    """Fetch news and compute sentiment features for a list of tickers.

    Returns DataFrame indexed by ticker with columns:
      - news_sentiment_avg: VADER compound score (-1 to +1)
      - news_volume_5d: number of articles (proxy for attention)
      - news_sentiment_pos_ratio: fraction of positive articles
    """
    rows: list[dict] = []

    for t in tickers:
        # Check cache
        cached = _load_cache(t, cache_dir, cache_ttl_hours) if cache_dir else None
        if cached is not None:
            rows.append(cached)
            continue

        articles = fetch_ticker_news(t, logger=logger)
        sentiment = compute_news_sentiment(articles)

        row = {
            "ticker": t,
            "news_sentiment_avg": sentiment["sentiment_avg"],
            "news_volume_5d": float(sentiment["article_count"]),
            "news_sentiment_pos_ratio": sentiment["sentiment_pos_ratio"],
        }
        rows.append(row)

        if cache_dir:
            _save_cache(t, row, cache_dir)

    if not rows:
        return pd.DataFrame(
            columns=["news_sentiment_avg", "news_volume_5d", "news_sentiment_pos_ratio"],
        )

    df = pd.DataFrame(rows).set_index("ticker")
    if logger:
        n_with_news = (df["news_volume_5d"] > 0).sum()
        logger.info(
            "News sentiment: %d/%d tickers have articles, avg_sentiment=%.3f",
            n_with_news, len(df),
            df["news_sentiment_avg"].mean() if not df["news_sentiment_avg"].isna().all() else 0,
        )
    return df


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / "news" / f"{ticker}.json"


def _load_cache(ticker: str, cache_dir: Path, ttl_hours: int) -> dict | None:
    import json
    p = _cache_path(ticker, cache_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = data.get("_cached_utc")
        if ts:
            cached_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = (datetime.now(tz=timezone.utc) - cached_dt).total_seconds() / 3600.0
            if age > ttl_hours:
                return None
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        return None


def _save_cache(ticker: str, row: dict, cache_dir: Path) -> None:
    import json
    p = _cache_path(ticker, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**row, "_cached_utc": datetime.now(tz=timezone.utc).isoformat()}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
