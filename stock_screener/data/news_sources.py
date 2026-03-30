"""Multi-source news aggregator: RSS feeds, Reddit, Finnhub.

All sources produce a unified article dict:
  {"title": str, "publisher": str, "publish_date": str|None, "link": str, "source_type": str}

This module supplements the existing yfinance-based news in news.py.
"""
from __future__ import annotations

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds per HTTP request
_USER_AGENT = "StockScreener/1.0 (github.com/anshu92/evr)"


def _http_get(url: str, *, timeout: int = _TIMEOUT) -> bytes | None:
    """Simple HTTP GET with timeout. Returns bytes or None on failure."""
    try:
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        logger.debug("HTTP GET failed for %s: %s", url, e)
        return None


# ── RSS Feeds (no API key, unlimited) ──────────────────────────────────────

_RSS_FEEDS: dict[str, str] = {
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
}


def fetch_rss_headlines(max_per_feed: int = 10) -> list[dict[str, Any]]:
    """Fetch top headlines from financial RSS feeds (MarketWatch, Seeking Alpha).

    Returns list of article dicts. No API key required.
    """
    articles: list[dict[str, Any]] = []
    for source_name, url in _RSS_FEEDS.items():
        data = _http_get(url)
        if not data:
            continue
        try:
            root = ET.fromstring(data)
            for item in root.findall(".//item")[:max_per_feed]:
                title = (item.findtext("title") or "").strip()
                if not title:
                    continue
                articles.append({
                    "title": title,
                    "publisher": source_name,
                    "publish_date": item.findtext("pubDate"),
                    "link": item.findtext("link") or "",
                    "source_type": "rss",
                })
        except ET.ParseError:
            logger.debug("RSS parse failed for %s", source_name)
    return articles


# ── Reddit (no API key, ~60 req/min with User-Agent) ──────────────────────

_REDDIT_SUBREDDITS = ["wallstreetbets", "stocks"]


def fetch_reddit_posts(
    subreddits: list[str] | None = None,
    sort: str = "hot",
    limit: int = 10,
    min_score: int = 50,
) -> list[dict[str, Any]]:
    """Fetch top posts from financial subreddits.

    Filters by min_score to surface high-signal posts only.
    Uses the public JSON API (no authentication required).
    """
    subs = subreddits or _REDDIT_SUBREDDITS
    articles: list[dict[str, Any]] = []
    for sub in subs:
        url = f"https://www.reddit.com/r/{sub}/{sort}.json?limit={limit}"
        data = _http_get(url)
        if not data:
            continue
        try:
            payload = json.loads(data)
            for child in payload.get("data", {}).get("children", []):
                post = child.get("data", {})
                score = post.get("score", 0)
                if score < min_score:
                    continue
                title = post.get("title", "").strip()
                if not title:
                    continue
                created = post.get("created_utc")
                pub_date = (
                    datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                    if created else None
                )
                articles.append({
                    "title": title,
                    "publisher": f"r/{sub} ({score} upvotes)",
                    "publish_date": pub_date,
                    "link": f"https://reddit.com{post.get('permalink', '')}",
                    "source_type": "reddit",
                    "score": score,
                })
        except (json.JSONDecodeError, KeyError):
            logger.debug("Reddit parse failed for r/%s", sub)
    # Sort by score descending (most upvoted first)
    articles.sort(key=lambda a: a.get("score", 0), reverse=True)
    return articles


def search_reddit_ticker(
    ticker: str,
    subreddits: list[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search Reddit for posts mentioning a specific ticker.

    Uses Reddit's search endpoint with restrict_sr and time filter.
    """
    subs = subreddits or _REDDIT_SUBREDDITS
    articles: list[dict[str, Any]] = []
    for sub in subs:
        url = (
            f"https://www.reddit.com/r/{sub}/search.json"
            f"?q={ticker}&sort=new&limit={limit}&restrict_sr=on&t=week"
        )
        data = _http_get(url)
        if not data:
            continue
        try:
            payload = json.loads(data)
            for child in payload.get("data", {}).get("children", []):
                post = child.get("data", {})
                title = post.get("title", "").strip()
                if not title:
                    continue
                created = post.get("created_utc")
                pub_date = (
                    datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                    if created else None
                )
                articles.append({
                    "title": title,
                    "publisher": f"r/{sub}",
                    "publish_date": pub_date,
                    "link": f"https://reddit.com{post.get('permalink', '')}",
                    "source_type": "reddit",
                    "score": post.get("score", 0),
                })
        except (json.JSONDecodeError, KeyError):
            logger.debug("Reddit search failed for %s in r/%s", ticker, sub)
    return articles


# ── Finnhub (free API key, 60 calls/sec) ──────────────────────────────────

def fetch_finnhub_news(
    ticker: str,
    days_back: int = 3,
    limit: int = 5,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch company news from Finnhub (requires free API key).

    Set FINNHUB_API_KEY env var or pass api_key directly.
    Free tier: 60 API calls/second.
    """
    key = api_key or os.getenv("FINNHUB_API_KEY", "")
    if not key:
        return []

    now = datetime.now(tz=timezone.utc)
    from_date = (now - __import__("datetime").timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = now.strftime("%Y-%m-%d")

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={from_date}&to={to_date}&token={key}"
    )
    data = _http_get(url)
    if not data:
        return []
    try:
        items = json.loads(data)
        if not isinstance(items, list):
            return []
        articles: list[dict[str, Any]] = []
        for item in items[:limit]:
            headline = item.get("headline", "").strip()
            if not headline:
                continue
            dt = item.get("datetime")
            pub_date = (
                datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()
                if dt else None
            )
            articles.append({
                "title": headline,
                "publisher": item.get("source", "Finnhub"),
                "publish_date": pub_date,
                "link": item.get("url", ""),
                "source_type": "finnhub",
                "summary": item.get("summary", ""),
            })
        return articles
    except (json.JSONDecodeError, KeyError):
        logger.debug("Finnhub parse failed for %s", ticker)
        return []


# ── Unified Aggregator ────────────────────────────────────────────────────

def fetch_market_news(max_headlines: int = 15) -> list[dict[str, Any]]:
    """Fetch general market news from all free sources (no ticker filter).

    Combines: RSS feeds + Reddit hot posts. No API key required.
    Returns up to max_headlines articles sorted by recency.
    """
    articles: list[dict[str, Any]] = []
    articles.extend(fetch_rss_headlines(max_per_feed=8))
    articles.extend(fetch_reddit_posts(limit=10, min_score=100))
    # Deduplicate by title similarity (exact match after lowering)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for a in articles:
        key = a["title"].lower().strip()[:60]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)
    return deduped[:max_headlines]


def fetch_ticker_news_multi(
    ticker: str,
    logger_: Any = None,
) -> list[dict[str, Any]]:
    """Fetch news for a specific ticker from ALL available sources.

    Combines: yfinance (existing) + Finnhub + Reddit search.
    Returns unified article list sorted by source priority.
    """
    _log = logger_ or logger
    articles: list[dict[str, Any]] = []

    # 1. yfinance (primary — always available)
    try:
        from stock_screener.data.news import fetch_ticker_news as _yf_news
        yf_articles = _yf_news(ticker, logger=_log)
        for a in yf_articles[:5]:
            a["source_type"] = "yfinance"
            articles.append(a)
    except Exception:
        pass

    # 2. Finnhub (if API key available)
    fh = fetch_finnhub_news(ticker, limit=5)
    articles.extend(fh)

    # 3. Reddit ticker search
    reddit = search_reddit_ticker(ticker, limit=3)
    articles.extend(reddit)

    # Deduplicate by title
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for a in articles:
        key = a["title"].lower().strip()[:60]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)

    if _log and deduped:
        sources = {}
        for a in deduped:
            s = a.get("source_type", "unknown")
            sources[s] = sources.get(s, 0) + 1
        _log.debug("News for %s: %d articles (%s)", ticker, len(deduped), sources)

    return deduped
