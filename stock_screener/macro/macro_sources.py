"""Macro-oriented headline ingestion: RSS, optional Finnhub general news, existing financial RSS."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from stock_screener.data.news_sources import fetch_rss_headlines, _is_recent

logger = logging.getLogger(__name__)

_USER_AGENT = "StockScreenerMacro/1.0 (github.com/anshu92/evr)"


def _story_hash(title: str, link: str, publish_date: str | None) -> str:
    base = f"{title}|{link}|{publish_date or ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def _parse_pub_iso(pub: str | None) -> str | None:
    if not pub:
        return None
    try:
        dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        pass
    try:
        from email.utils import parsedate_to_datetime

        dt2 = parsedate_to_datetime(pub)
        if dt2.tzinfo is None:
            dt2 = dt2.replace(tzinfo=timezone.utc)
        return dt2.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def fetch_finnhub_general_news(
    *,
    max_headlines: int = 25,
    max_age_days: int = 3,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Finnhub market-wide news (category=general). HTTPS only."""
    key = api_key or os.getenv("FINNHUB_API_KEY", "")
    if not key:
        return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={key}"
    try:
        resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=20)
        resp.raise_for_status()
        items = resp.json()
    except Exception as e:
        logger.debug("Finnhub general news failed: %s", e)
        return []
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    now = datetime.now(tz=timezone.utc)
    for item in items[: max_headlines * 2]:
        headline = str(item.get("headline", "")).strip()
        if not headline:
            continue
        dt = item.get("datetime")
        pub_date = (
            datetime.fromtimestamp(int(dt), tz=timezone.utc).isoformat()
            if dt is not None
            else None
        )
        if pub_date and not _is_recent(pub_date, max_age_days):
            continue
        out.append(
            {
                "title": headline,
                "publisher": str(item.get("source", "Finnhub")),
                "publish_date": pub_date,
                "link": str(item.get("url", "")),
                "source_type": "finnhub_general",
                "summary": str(item.get("summary", "")),
                "reliability_tier": "tier2",
            }
        )
        if len(out) >= max_headlines:
            break
    return out


def fetch_bls_news_rss(max_items: int = 15) -> list[dict[str, Any]]:
    """BLS news RSS (macro labor/inflation releases)."""
    url = "https://www.bls.gov/feed/bls_news.xml"
    articles: list[dict[str, Any]] = []
    try:
        resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        logger.debug("BLS RSS failed: %s", e)
        return articles
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(resp.content)
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            pub = _parse_pub_iso(item.findtext("pubDate"))
            articles.append(
                {
                    "title": title,
                    "publisher": "BLS",
                    "publish_date": pub,
                    "link": item.findtext("link") or "",
                    "source_type": "rss_bls",
                    "summary": "",
                    "reliability_tier": "tier1",
                }
            )
    except Exception as e:
        logger.debug("BLS RSS parse failed: %s", e)
    return articles


def normalize_article(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized story dict with story_hash."""
    title = str(raw.get("title", "")).strip()
    link = str(raw.get("link", "")).strip()
    pub = raw.get("publish_date")
    pub_iso = _parse_pub_iso(str(pub)) if pub else None
    story_hash = _story_hash(title, link, pub_iso)
    return {
        "title": title,
        "publisher": str(raw.get("publisher", "")).strip() or "unknown",
        "publish_date": pub_iso,
        "link": link,
        "source_type": str(raw.get("source_type", "unknown")),
        "summary": str(raw.get("summary", "")).strip(),
        "reliability_tier": str(raw.get("reliability_tier", "tier2")),
        "story_hash": story_hash,
    }


def dedupe_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for a in articles:
        key = a.get("title", "").lower().strip()[:72]
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def fetch_macro_headlines(
    *,
    max_per_source: int = 20,
    max_age_days: int = 3,
) -> list[dict[str, Any]]:
    """Aggregate macro-relevant headlines from allowed HTTPS sources."""
    merged: list[dict[str, Any]] = []

    for a in fetch_bls_news_rss(max_items=max_per_source):
        merged.append(a)

    for a in fetch_rss_headlines(max_per_feed=max_per_source):
        a = dict(a)
        a.setdefault("reliability_tier", "tier2")
        merged.append(a)

    for a in fetch_finnhub_general_news(max_headlines=max_per_source, max_age_days=max_age_days):
        merged.append(a)

    filtered: list[dict[str, Any]] = []
    for a in merged:
        if a.get("publish_date") and not _is_recent(a.get("publish_date"), max_age_days):
            continue
        filtered.append(normalize_article(a))

    return dedupe_articles(filtered)
