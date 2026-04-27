# Real-Time Copy-Trade Signal Workflow Research

This workflow should discover near-real-time portfolio trades from opt-in traders, signal platforms, or market-flow feeds, convert them into normalized trade signals, and feed those signals into the existing screener as a capped conviction overlay. It should not blindly mirror another portfolio.

## Bottom Line

True real-time portfolio trades are not generally public. They are available only when the portfolio owner opts into a copy-trading/signal platform, or when a platform has a legal brokerage/advisory relationship that lets it distribute and execute model changes. Public filings remain useful for research, but they are not real time.

Best real-time candidates:

- **Regulated copy-trading apps** such as eToro CopyTrader and dub. These can mirror another investor's portfolio actions in real time inside their own app/brokerage environment.
- **Signal marketplaces** such as Collective2, MetaTrader Signals, ZuluTrade, and TradeStation TradingApp Store. These expose strategy signals or auto-trading hooks, but asset coverage and reliability vary.
- **Opt-in creator/model portfolios** from broker/advisor platforms. These are the cleanest route if the creator explicitly publishes trades or model updates.
- **Real-time options/equity flow feeds** such as OPRA-derived options flow, block/sweep detectors, and dark-pool/equity print scanners. These do not identify a portfolio manager, but can infer institutional activity.

Not real time:

- **13F institutional holdings** are the best starting point for proven fund managers. They are official, structured, broad, and free, but quarterly and delayed up to 45 days after quarter end.
- **Schedule 13D/13G activist ownership** is more timely and can flag concentrated, high-conviction stakes, but only once beneficial ownership crosses 5%.
- **Form 4 insider transactions** is timely and useful for corporate insider buying, but it tracks insiders of the issuer, not profitable external portfolio managers.
- **Congressional Periodic Transaction Reports** are public and can be profitable to study, but amounts are disclosed in ranges and trades may be reported up to 45 days after the transaction.
- **N-PORT fund holdings** can help track registered funds, but SEC data sets are quarterly releases of monthly holdings, so they are better for research than near-real-time copying.

## Source Evaluation

### A. Opt-In Copy-Trading Platforms

Use for: actual real-time mirroring of portfolios.

Examples:

- eToro CopyTrader: public product docs state copied trades are mirrored automatically in real time/proportion. U.S. availability and copied asset coverage vary by account, state, and program.
- dub: support docs describe auto-copying creator portfolio actions, rebalances, and liquidations through dub Financial. This is closer to the requested product, but currently appears app/platform-bound rather than a public developer data feed.
- Autopilot: app-store description claims real-time mirroring of expert, AI, hedge-fund-tracker, thematic, and politician-tracker portfolios inside a connected brokerage workflow.

Major limitations:

- Most do not expose a public API for extracting creator trades into this repo.
- You usually must execute inside their platform, not harvest signals externally.
- Availability depends on jurisdiction, brokerage, account type, and product eligibility.
- Performance leaderboards can be biased by short histories, risk, leverage, and survivorship.

Implementation role:

- Best path if the goal is **real copy execution**: use the platform directly.
- Best repo path: only ingest opt-in/exported activity if terms permit it, or model our own "creator portfolio" in this repo.

### B. Signal Marketplaces and Auto-Trading Networks

Use for: real-time buy/sell signals from strategy creators.

Examples:

- Collective2: supports stocks, options, futures, and forex; has AutoTrade and APIs for subscribing to strategies and requesting real-time buy/sell signal information.
- MetaTrader Signals: real-time copying of signal-provider deals, mainly forex/CFD/crypto/metals depending on broker.
- ZuluTrade: social/copy trading of leader actions.
- TradeStation TradingApp Store: marketplace for strategies/apps inside TradeStation.

Major limitations:

- Many strategies are not stock-only and may use leverage, options, futures, forex, or CFDs.
- Published returns may be hypothetical/simulated or affected by slippage.
- API availability, redistribution rights, and broker compatibility need review before integration.

Implementation role:

- More practical than scraping copy-trading apps if API access is available.
- Use as `SignalProviderEvent` inputs with strict risk caps and slippage assumptions.

### C. Real-Time Market Flow Feeds

Use for: inferring institutional activity when portfolio identity is unavailable.

Examples:

- OPRA-derived options prints/quotes via vendors.
- Unusual options flow, sweep, block, and dark-pool/equity print services.
- Exchange/ATS/FINRA market transparency feeds where available.

Major limitations:

- These are trades, not portfolio-level trades.
- The buyer/seller and portfolio owner are usually unknown.
- Direction is inferred, not guaranteed.
- Data licensing can be expensive and restrictive.

Implementation role:

- Useful as a separate `flow_score`, not as "copying a profitable portfolio."
- Best for short-horizon confirmation, especially when paired with the existing ML screener.

### D. Public Disclosures

Use for: lagged research, manager ranking, and fallback idea generation.

These sources are not real time, but they can rank managers and validate whether real-time signal sources are worth copying.

### 1. SEC 13F Holdings

Use for: lagged "superinvestor" and hedge-fund copy portfolios.

What it gives:

- Manager name and CIK.
- Security name/class, CUSIP, shares, and market value.
- Quarterly long U.S. exchange-traded equity-style holdings.

Major limitations:

- Filed within 45 days after quarter end, so positions can be stale.
- Generally excludes shorts, many international ordinary shares, private holdings, and intra-quarter trading.
- Some managers can receive confidential treatment for selected positions.
- CUSIP-to-ticker mapping is required.

Implementation role:

- Primary source for slow-copy, high-conviction ideas.
- Score **changes** in holdings, not just raw ownership.
- Favor managers with persistent historical alpha after disclosure lag.

### 2. Schedule 13D / 13G

Use for: lagged activist/passive 5% ownership events.

What it gives:

- Investors who acquire beneficial ownership over 5% of a voting equity class.
- 13D is especially useful when the investor may influence control or strategy.

Major limitations:

- Only captures large ownership thresholds.
- Not a full portfolio feed.
- Some 13G filers are passive and less informative.

Implementation role:

- Event-driven signal source.
- Higher weight for 13D than 13G.
- Require liquidity and price filters before trade consideration.

### 3. Form 4 Insider Transactions

Use for: near-real-time issuer-insider conviction, especially open-market purchases.

What it gives:

- Officers, directors, and 10% beneficial owners must report many ownership changes quickly.
- Transaction codes distinguish open-market buys/sells from grants, option exercises, gifts, and tax-related transactions.

Major limitations:

- Insiders are not external portfolio managers.
- Sales are noisy because they may reflect diversification, tax, or 10b5-1 plans.
- Existing repo code currently uses yfinance insider data, which is weaker than parsing SEC ownership XML directly.

Implementation role:

- Upgrade `stock_screener/data/insiders.py` to SEC primary-source Form 4 XML.
- Treat open-market purchases (`P`) and cluster buying as positive overlays.
- Treat routine sales as low/zero signal unless unusually large and not plan-based.

### 4. Congressional PTRs

Use for: lagged optional "political trade disclosure" signal.

What it gives:

- Member, spouse, and dependent-child trades above reporting thresholds.
- Purchases, sales, exchanges, and some option activity.

Major limitations:

- Reports can lag the transaction by up to 45 days.
- Amounts are ranges, not exact sizes.
- Ticker normalization from official filings can be messy.
- Ethical and reputational risk is higher; this should be transparent and optional.

Implementation role:

- Replace the current Reddit-derived congress ticker hook with primary House/Senate disclosure parsing or a paid/curated API.
- Keep it opt-in via config, and cap its signal contribution.

### 5. N-PORT Fund Holdings

Use for: registered fund replication research.

What it gives:

- Monthly portfolio holdings for registered management companies and relevant ETFs/funds, published through quarterly SEC data sets.

Major limitations:

- Better for backtesting and factor extraction than near-term copying.
- Fund strategies can be diversified, index-like, or constrained in ways that are hard to copy.

Implementation role:

- Later-stage research source, not v1.

## Proposed Repo Workflow

Add a new logical pipeline:

```text
real-time provider feeds -> normalized provider events -> provider ranking -> copy-trade signals
-> screener overlay -> portfolio manager -> reports/state
```

Recommended files:

- `stock_screener/data/copy_trade_providers.py`
  - Provider adapters for permitted APIs/exports such as Collective2, broker model portfolios, or an internal creator portfolio feed.
  - Normalize to `CopyTradeEvent`.
- `stock_screener/data/market_flow.py`
  - Optional real-time options/equity flow adapter if a licensed feed is available.
  - Normalize to `FlowEvent`.
- `stock_screener/data/sec_filings.py`
  - Fallback research/disclosure helper with declared user agent, rate limiting, retry/backoff.
  - Fetch submission history and filing documents.
- `stock_screener/data/portfolio_disclosures.py`
  - Lagged 13F, 13D/G, and optional N-PORT parsers.
  - Normalize to `DisclosureEvent`.
- `stock_screener/signals/copy_trading.py`
  - Provider universe, live/lagged performance scoring, signal scoring, and caps.
- `stock_screener/pipeline/copy_trades.py`
  - Standalone intraday/daily signal materialization.
  - Writes `cache/copy_trade_signals.parquet` and `reports/copy_trade_signals.csv/json`.
- `stock_screener/cli.py`
  - Add `copy-trades` command.
- `.github/workflows/copy-trade-signals.yml.disabled`
  - Scheduled intraday only for permitted real-time providers; daily/weekly for lagged disclosures.

## Signal Rules

Use conservative rules first:

- Include only liquid common stocks already passing repo price/liquidity filters.
- Ignore any provider event older than a configurable lag threshold.
- Require explicit source type: `copy_platform`, `signal_marketplace`, `internal_creator`, `market_flow`, or `public_disclosure`.
- For copy-platform/signal events:
  - `buy` or `increase`: positive, scaled by provider quality and trade freshness.
  - `sell` or `decrease`: negative or exit-review signal.
  - `rebalance`: smaller signal unless position delta is large.
- For market-flow events:
  - keep separate from copy events and require repeated/clustered activity before scoring.
- For 13F:
  - `new_position`: positive.
  - `increased_position`: positive, scaled by percent increase and portfolio weight.
  - `trimmed_position`: neutral/slightly negative.
  - `sold_out`: negative.
- For 13D:
  - strong positive event, decayed over time.
- For 13G:
  - weaker positive event unless the filer is on the high-conviction manager allowlist.
- For Form 4:
  - open-market purchases and cluster buys only.
- For Congress:
  - optional, small cap, range-midpoint sizing only.

Portfolio integration:

- Add `copy_trade_score` and `copy_trade_signal_count` feature columns.
- Blend as an overlay after ML scoring, with a small default cap such as 10%-15% of final score.
- Never bypass existing entry, liquidity, volatility, concentration, exposure, and rebalance controls.
- Emit audit data showing provider, strategy, source type, event time, observed time, lag seconds/minutes, ticker, side, size/weight delta, and score contribution.

## Manager Ranking

Do not define "most profitable" by raw recent return. Rank real-time providers using executable, slippage-aware performance:

- Use only providers with sufficient live track record.
- Compute net return after estimated slippage, spread, fees, and borrow/option costs where relevant.
- Penalize leverage, max drawdown, volatility, turnover, short track records, martingale sizing, and strategy crowding.
- Track signal latency from provider event to our observable/possible execution time.
- Prefer providers with transparent holdings, own-capital participation, stable risk, and low turnover.

For public-disclosure managers, use lag-aware backtests:

- Build historical 13F snapshots by manager.
- Reconstruct hypothetical portfolios using only information available after the actual filing date.
- Score excess return versus SPY/sector benchmarks over 1, 3, and 6 months after disclosure.
- Penalize high turnover, concentrated one-hit records, stale filings, and survivorship bias.
- Use a small curated allowlist for v1, then graduate to data-driven rankings once backtests exist.

## V1 Recommendation

For true real-time copying, start with an API-accessible signal marketplace or an internal opt-in model portfolio feed. Do not start by scraping eToro/dub/Autopilot unless their terms explicitly allow it.

Implemented v1 target in this repo:

- Collective2 World API access audit + signal polling via `python -m stock_screener.cli collective2-copy`.
- Enabled GitHub Actions workflow at `.github/workflows/collective2-copy-trades.yml`.
- Paper-only USD model portfolio with default initial cash of `$4,000`.
- Long stock/ETF signals only; no live brokerage execution and no Collective2 order-submission calls.

V1 deliverables:

1. Provider interface: `CopyTradeEvent(provider, strategy_id, ticker, side, quantity_or_weight_delta, event_time, confidence, source_url)`.
2. One permitted provider adapter, preferably Collective2 API or a local/internal JSON feed.
3. Provider ranking with drawdown/turnover/slippage penalties.
4. `copy_trade_score` feature export.
5. Dry-run report comparing copied signals against repo ML scores and existing portfolio constraints.
6. Portfolio overlay disabled by default until dry-run results are reviewed.
7. Optional disclosure parsers as slower fallback/research inputs, not the core real-time path.

## Operational Notes

- Do not scrape logged-in apps or redistribute copied trades unless the provider's terms allow it.
- For broker-linked execution, confirm regulatory/account suitability and API permissions first.
- Cache raw provider events and normalized outputs to avoid duplicate signals.
- Store enough metadata to reproduce every signal.
- Default workflow should be report-only. Portfolio enforcement should require an explicit variable such as `COPY_TRADE_OVERLAY_ENABLED=1`.
- Treat this as an idea-generation overlay, not automated fiduciary advice.
