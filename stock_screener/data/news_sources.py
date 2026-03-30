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
_REDDIT_CONGRESS_SUBS = ["CongressStockWatcher"]


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


# ── Congressional Trading (Reddit r/CongressStockWatcher) ─────────────────

_TICKER_RE = re.compile(r"\$([A-Z]{2,5})\b")
_MAX_AGE_DAYS = 14  # Default: ignore posts older than 2 weeks


def _is_recent(pub_date: str | None, max_age_days: int = _MAX_AGE_DAYS) -> bool:
    """Check if a publish date is within max_age_days of now."""
    if not pub_date:
        return True  # No date = assume recent
    try:
        dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        age = (datetime.now(tz=timezone.utc) - dt).total_seconds() / 86400
        return age <= max_age_days
    except (ValueError, TypeError):
        return True


def fetch_congress_trades(
    limit: int = 25, min_score: int = 1, max_age_days: int = _MAX_AGE_DAYS,
) -> list[dict[str, Any]]:
    """Fetch recent congressional trading posts from r/CongressStockWatcher.

    Returns article dicts with source_type="congress", filtered to max_age_days.
    Posts include congress member trades, insider buys, and political trading signals.
    No API key required.
    """
    articles: list[dict[str, Any]] = []
    for sub in _REDDIT_CONGRESS_SUBS:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit={limit}"
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
                if not _is_recent(pub_date, max_age_days):
                    continue
                # Extract tickers from title ($TICKER pattern)
                tickers_found = _TICKER_RE.findall(title)
                articles.append({
                    "title": title,
                    "publisher": f"r/{sub}",
                    "publish_date": pub_date,
                    "link": f"https://reddit.com{post.get('permalink', '')}",
                    "source_type": "congress",
                    "score": score,
                    "tickers": tickers_found,
                })
        except (json.JSONDecodeError, KeyError):
            logger.debug("Reddit congress parse failed for r/%s", sub)
    return articles


def get_congress_traded_tickers(max_age_days: int = _MAX_AGE_DAYS) -> list[str]:
    """Extract unique tickers from recent congressional trades.

    Returns list of ticker symbols that congress members have recently traded.
    Useful for adding to the screening universe as high-signal candidates.
    """
    trades = fetch_congress_trades(limit=25, min_score=1, max_age_days=max_age_days)
    tickers: list[str] = []
    seen: set[str] = set()
    for t in trades:
        for ticker in t.get("tickers", []):
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers


def get_congress_trades_for_ticker(
    ticker: str, limit: int = 25, max_age_days: int = _MAX_AGE_DAYS,
) -> list[dict[str, Any]]:
    """Get congressional trading posts that mention a specific ticker.

    Searches post titles for $TICKER pattern and ticker name mentions.
    Filtered to max_age_days.
    """
    all_posts = fetch_congress_trades(limit=limit, min_score=1, max_age_days=max_age_days)
    ticker_upper = ticker.upper()
    matches: list[dict[str, Any]] = []
    for post in all_posts:
        # Match by $TICKER tag
        if ticker_upper in post.get("tickers", []):
            matches.append(post)
            continue
        # Match by ticker appearing in title
        if f" {ticker_upper} " in f" {post['title'].upper()} ":
            matches.append(post)
    return matches


# ── Wild Card Ticker Extraction ────────────────────────────────────────────

def get_wildcard_tickers(max_age_days: int = 7, min_reddit_score: int = 200) -> list[dict[str, Any]]:
    """Extract high-signal tickers from Reddit and news that deserve LLM evaluation.

    Returns list of dicts: {"ticker": str, "source": str, "reason": str, "score": int}
    Sorted by signal strength (Reddit upvotes, congress trade).
    """
    seen: set[str] = set()
    wildcards: list[dict[str, Any]] = []

    # 1. Congressional trades (highest signal)
    congress = fetch_congress_trades(limit=25, min_score=1, max_age_days=max_age_days)
    for post in congress:
        for ticker in post.get("tickers", []):
            if ticker in seen:
                continue
            seen.add(ticker)
            wildcards.append({
                "ticker": ticker,
                "source": "congress",
                "reason": post["title"][:80],
                "score": post.get("score", 0) + 1000,  # Congress trades get priority boost
            })

    # 2. Reddit trending (high-upvote posts with $TICKER mentions)
    for sub in _REDDIT_SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit=25"
        data = _http_get(url)
        if not data:
            continue
        try:
            payload = json.loads(data)
            for child in payload.get("data", {}).get("children", []):
                post = child.get("data", {})
                score = post.get("score", 0)
                if score < min_reddit_score:
                    continue
                title = post.get("title", "")
                tickers_found = _TICKER_RE.findall(title)
                created = post.get("created_utc")
                if created:
                    pub_date = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                    if not _is_recent(pub_date, max_age_days):
                        continue
                for ticker in tickers_found:
                    if ticker in seen:
                        continue
                    seen.add(ticker)
                    wildcards.append({
                        "ticker": ticker,
                        "source": f"r/{sub}",
                        "reason": title[:80],
                        "score": score,
                    })
        except (json.JSONDecodeError, KeyError):
            pass

    # 3. RSS headline tickers (MarketWatch, Seeking Alpha)
    rss = fetch_rss_headlines(max_per_feed=10)
    for article in rss:
        title = article.get("title", "")
        tickers_found = _TICKER_RE.findall(title)
        if not _is_recent(article.get("publish_date"), max_age_days):
            continue
        for ticker in tickers_found:
            if ticker in seen:
                continue
            seen.add(ticker)
            wildcards.append({
                "ticker": ticker,
                "source": article.get("publisher", "RSS"),
                "reason": title[:80],
                "score": 50,  # RSS gets base score
            })

    # Sort by score descending
    wildcards.sort(key=lambda x: x["score"], reverse=True)
    return wildcards


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

def fetch_market_news(max_headlines: int = 20, max_age_days: int = 7) -> list[dict[str, Any]]:
    """Fetch general market news from all free sources (no ticker filter).

    Combines: Congressional trades (priority) + RSS feeds + Reddit hot posts.
    Congress trades are listed first since they're high-signal for trading.
    All articles filtered to max_age_days. No API key required.
    """
    # Congress trades first (highest signal)
    congress = fetch_congress_trades(limit=15, min_score=1, max_age_days=max_age_days)
    rss = [a for a in fetch_rss_headlines(max_per_feed=8) if _is_recent(a.get("publish_date"), max_age_days)]
    reddit = [a for a in fetch_reddit_posts(limit=10, min_score=100) if _is_recent(a.get("publish_date"), max_age_days)]

    # Congress first, then RSS, then Reddit
    articles: list[dict[str, Any]] = congress + rss + reddit

    # Deduplicate by title similarity
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
    max_age_days: int = 7,
) -> list[dict[str, Any]]:
    """Fetch news for a specific ticker from ALL available sources.

    Combines: yfinance + Finnhub + Reddit search + Congressional trades.
    Filtered to max_age_days. Returns unified article list.
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

    # 4. Congressional trading (if any member traded this ticker)
    congress = get_congress_trades_for_ticker(ticker, max_age_days=max_age_days)
    articles.extend(congress)

    # Filter by age + deduplicate by title
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for a in articles:
        if not _is_recent(a.get("publish_date"), max_age_days):
            continue
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
