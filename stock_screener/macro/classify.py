"""Rule-based macro theme classification and novelty hints."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any


def _day_bucket(iso: str | None) -> str:
    if not iso:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def detect_scheduled_event(title: str) -> tuple[str | None, str | None]:
    patterns: list[tuple[str, str]] = [
        (r"\bCPI\b", "CPI"),
        (r"\bPCE\b", "PCE"),
        (r"\bFOMC\b|\bFed\b|\bFederal Reserve\b", "FOMC"),
        (r"\bNFP\b|\bnon[- ]farm\b|\bpayrolls\b", "NFP"),
        (r"\bGDP\b", "GDP"),
        (r"\bEIA\b|\bcrude inventories\b|\boil inventories\b", "EIA"),
        (r"\bISM\b", "ISM"),
        (r"\bretail sales\b", "RETAIL_SALES"),
        (r"\bunemployment\b|\bjobs report\b", "LABOR"),
    ]
    t = title
    for pat, key in patterns:
        if re.search(pat, t, flags=re.IGNORECASE):
            return key, f"{key.lower()}:{_day_bucket(None)}"
    return None, None


def _surprise_direction(title: str) -> str | None:
    tl = title.lower()
    if any(x in tl for x in ["hotter", "above expectations", "beats", "stronger than expected"]):
        return "upside"
    if any(x in tl for x in ["cooler", "below expectations", "misses", "weaker than expected"]):
        return "downside"
    return None


def _basket_keyword_hits(text: str) -> list[str]:
    t = text.lower()
    hits: list[str] = []
    rules: list[tuple[str, list[str]]] = [
        ("oil_up_energy", ["crude", "wti", "opec", "oil price", "brent", "oil jumps", "oil rises"]),
        ("oil_down_consumers", ["oil falls", "oil drops", "cheaper oil"]),
        ("semicap_ai_capex", ["nvidia", "chip", "semiconductor", "ai boom", "gpu", "foundry"]),
        ("cloud_ai_power_demand", ["datacenter", "data center", "hyperscaler", "cloud capex"]),
        ("rates_up_banks", ["yield surge", "yields jump", "rates rise", "hawkish fed", "higher rates"]),
        ("rates_down_duration", ["yield plunge", "yields fall", "dovish fed", "rate cut", "cuts rates"]),
        ("usd_strength_exporters", ["dollar index", "dxy", "strong dollar", "usd strength"]),
        ("usd_weakness_multinationals", ["weak dollar", "dollar weak", "usd falls"]),
        ("gold_real_rates_down", ["gold", "bullion", "xau"]),
        ("copper_industrials_china", ["copper", "china demand", "china stimulus"]),
        ("housing_rates_down", ["homebuilder", "housing starts", "mortgage rates fall", "mortgage"]),
        ("credit_stress_defensive", ["credit crunch", "default cycle", "recession fear", "stagflation"]),
        ("regional_banks_credit_relief", ["regional bank", "svb", "bank stress"]),
        ("defense_spending", ["pentagon", "defense contract", "military"]),
        ("uranium_nuclear", ["uranium", "nuclear power", "smr"]),
        ("natgas_power_utilities", ["natural gas", "natgas", "utilities surge"]),
    ]
    for basket_key, kws in rules:
        if any(kw in t for kw in kws):
            hits.append(basket_key)
    return hits


def classify_article(article: dict[str, Any], *, prior_titles: set[str]) -> dict[str, Any]:
    """Return classification payload for one normalized article."""
    title = article.get("title", "")
    summary = str(article.get("summary", ""))
    text = f"{title} {summary}"
    event_type, event_key = detect_scheduled_event(title)
    surprise = _surprise_direction(title)
    basket_hits = _basket_keyword_hits(text)

    # If scheduled macro headline but no basket keyword, map coarsely
    if event_type == "CPI" and not basket_hits:
        basket_hits = ["credit_stress_defensive", "rates_down_duration"]
    if event_type == "FOMC" and not basket_hits:
        basket_hits = ["rates_down_duration", "rates_up_banks"]
    if event_type == "NFP" and not basket_hits:
        basket_hits = ["credit_stress_defensive", "rates_up_banks"]

    if not basket_hits:
        basket_hits = ["credit_stress_defensive"]

    theme_key = basket_hits[0].replace("_", " ")
    direction = "neutral"
    if surprise == "upside":
        direction = "firm"
    elif surprise == "downside":
        direction = "soft"

    # Novelty: simple title overlap with prior run titles
    tnorm = title.lower().strip()[:120]
    is_recycled = any(
        tnorm in p or p in tnorm for p in prior_titles if len(p) > 20 and len(tnorm) > 20
    )
    novelty_score = 0.3 if is_recycled else 0.9

    summary_short = (title[:140] + "…") if len(title) > 140 else title
    confidence = 0.45 + 0.1 * min(len(basket_hits), 3)
    if event_type:
        confidence += 0.15
    confidence = min(confidence, 0.95)

    day = _day_bucket(article.get("publish_date"))
    theme_id = hashlib.sha256(f"{basket_hits[0]}|{direction}|{day}".encode()).hexdigest()[:20]

    return {
        "theme_id": theme_id,
        "theme_key": basket_hits[0],
        "basket_keys": list(dict.fromkeys(basket_hits))[:4],
        "direction": direction,
        "horizon": "days",
        "confidence": confidence,
        "summary_short": summary_short,
        "watch_until_utc": None,
        "event_type": event_type,
        "event_key": event_key,
        "surprise_direction": surprise,
        "novelty_score": novelty_score,
        "staleness_hours": None,
        "is_scheduled_event": bool(event_type),
        "is_recycled_commentary": is_recycled,
    }
