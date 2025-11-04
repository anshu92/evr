# Automatic Position Monitoring & Closing

## Overview

The scanner now **automatically monitors all open positions** and closes them when they hit stop-loss or take-profit targets. This happens at the start of every scan run, ensuring your portfolio is always up-to-date.

## How It Works

### 1. **Monitoring Trigger**
Every time you run `python official_scanner.py`, the scanner:
1. Fetches current prices for all open positions
2. Checks if any position hit its stop or target
3. Automatically closes positions that meet exit conditions
4. Updates portfolio state with P&L calculations

### 2. **Exit Conditions**

#### Long Positions (Entry > Stop)
- **Stop Loss Hit**: Current price ≤ Stop price → Close as `"STOPPED_OUT"`
- **Target Hit**: Current price ≥ Target price → Close as `"TARGET_HIT"`

#### Short Positions (Entry < Stop)
- **Stop Loss Hit**: Current price ≥ Stop price → Close as `"STOPPED_OUT"`
- **Target Hit**: Current price ≤ Target price → Close as `"TARGET_HIT"`

### 3. **P&L Calculation**
When a position closes:
```python
# Calculate profit/loss
pnl = (exit_price - entry_price) × shares

# Return capital to available
available_capital += (shares × entry_price) + pnl

# Update total portfolio value
total_capital += pnl
```

## Example Output

When you run the scanner with open positions:

```
📊 Monitoring Open Positions...
Monitored: 3 | Closed: 2 | Stopped Out: 1 | Targets Hit: 1 | Errors: 0

✓ Closed LOB @ $29.15 (STOPPED_OUT) - Entry: $31.29, P&L: -$2,074.86
✓ Closed AWI @ $209.50 (TARGET_HIT) - Entry: $191.56, P&L: +$5,075.82

💼 Current Portfolio Status:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric               ┃ Value           ┃ Details                        ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Total Capital        │ $103,000.96     │ Initial: $1000, P&L: $3,000.96 │
│ Available Capital    │ $88,543.21      │ 86.0% of total                 │
│ Allocated Capital    │ $14,457.75      │ 14.0% of total                 │
│ Total Return         │ 3.00%           │ $3,000.96 absolute             │
│ Open Positions       │ 1               │ Risk: $14,457.75               │
│ Closed Positions     │ 2               │ Win Rate: 50.0%                │
│ Run Count            │ 8               │ Last: 2025-11-04 12:15         │
└──────────────────────┴─────────────────┴────────────────────────────────┘
```

## Key Features

### ✅ Duplicate Prevention
The scanner now prevents duplicate positions:
- Checks if position already exists for ticker + setup combination
- Only adds new position if ticker/setup is unique
- Logs when duplicates are skipped

### 📊 Monitoring Stats
Each scan reports:
- **Monitored**: Number of positions checked
- **Closed**: Total positions closed
- **Stopped Out**: Positions that hit stop-loss
- **Targets Hit**: Positions that hit take-profit
- **Errors**: Failed price fetches

### 💰 Accurate Capital Tracking
- **Allocated Capital**: Total value of open positions (shares × entry_price)
- **Available Capital**: Cash available for new positions
- **Total Capital**: Allocated + Available + cumulative P&L

### 📈 Performance History
Every position close is recorded in `performance_history` for analytics:
```json
{
  "timestamp": "2025-11-04T12:15:30",
  "total_capital": 103000.96,
  "total_pnl": 3000.96,
  "total_return_pct": 0.03,
  "open_positions": 1,
  "run_count": 8
}
```

## Code Implementation

### New Methods in `PortfolioManager`

#### `monitor_and_close_positions(data_fetcher)`
Main monitoring function that:
1. Gets all open positions
2. Fetches current prices
3. Checks exit conditions
4. Closes positions automatically

**Returns:**
```python
{
    'monitored': 3,
    'closed': 2,
    'stopped_out': 1,
    'targets_hit': 1,
    'errors': 0
}
```

#### `_find_existing_position(ticker, setup)`
Checks if an open position already exists:
- Prevents duplicate positions
- Searches by ticker + setup combination
- Only checks OPEN positions

**Returns:** `PortfolioPosition` or `None`

#### `close_position(ticker, exit_price, reason)`
Closes a position and updates portfolio:
- Calculates P&L
- Updates capital allocation
- Records performance
- Saves state to JSON

**Reasons:**
- `"MANUAL"` - User closed
- `"STOPPED_OUT"` - Hit stop-loss
- `"TARGET_HIT"` - Hit take-profit target

## Integration Points

### Main Scan Flow
```python
# 1. Monitor positions (NEW!)
monitor_results = portfolio_manager.monitor_and_close_positions(data_fetcher)

# 2. Display current state
display_portfolio_status()

# 3. Scan for new opportunities
trade_plans = scan_tickers(tickers)

# 4. Add new positions (with duplicate check!)
for plan in trade_plans:
    portfolio_manager.add_position(plan)  # Checks for duplicates
```

## Position State Transitions

```
     ┌─────────┐
     │  OPEN   │◄─── add_position()
     └────┬────┘
          │
          ├─ Current Price ≤ Stop (Long)  ─────► STOPPED_OUT
          │
          ├─ Current Price ≥ Target (Long) ────► TARGET_HIT
          │
          └─ Manual close() ───────────────────► MANUAL
```

## Files Modified

### `/Users/sahooa3/Documents/git/evr/official_scanner.py`

**PortfolioManager class:**
- ✅ Added `monitor_and_close_positions()` method (line 1113-1200)
- ✅ Added `_find_existing_position()` helper (line 1098-1111)
- ✅ Updated `add_position()` with duplicate checking (line 1057-1096)
- ✅ Added logger initialization (line 967)

**Main function:**
- ✅ Integrated monitoring before each scan (line 5043-5063)
- ✅ Display monitoring results with stats

## Benefits

1. **Automatic Risk Management**: No manual monitoring needed
2. **Accurate P&L Tracking**: Every exit is recorded with reason
3. **Capital Efficiency**: Freed capital immediately available
4. **Duplicate Prevention**: No accidental double positions
5. **Performance Analytics**: Historical tracking of all trades
6. **Clear Logging**: See exactly when and why positions closed

## Testing

### Test Automatic Closing

1. **Open some positions** by running the scanner:
   ```bash
   python official_scanner.py
   ```

2. **Check portfolio_state.json** - note open positions

3. **Wait for price movement** (or manually edit entry/stop in JSON for testing)

4. **Run scanner again**:
   ```bash
   python official_scanner.py
   ```

5. **Observe automatic closes** in the output:
   ```
   📊 Monitoring Open Positions...
   ✓ Closed TICKER @ $XX.XX (STOPPED_OUT) - Entry: $YY.YY, P&L: $ZZZ
   ```

### Test Duplicate Prevention

1. Run scanner to add positions
2. Run scanner again immediately
3. Same tickers should be skipped (not added twice)
4. Check logs for "Position already exists" messages

## Troubleshooting

### Position Not Closing
- **Issue**: Position should have closed but didn't
- **Cause**: Price data fetch failed
- **Check**: Monitor results show `Errors: N`
- **Fix**: Ticker might be delisted or data unavailable

### Wrong Direction Detection
- **Issue**: Long position treated as short (or vice versa)
- **Cause**: Entry price < Stop price (should be Entry > Stop for long)
- **Fix**: Check position.entry_price vs position.stop_price relationship

### Duplicate Positions Still Appearing
- **Issue**: Same ticker appears multiple times
- **Cause**: Different setup names for same ticker
- **Example**: `LOB` with `rsi_oversold` AND `stoch_oversold` are both allowed
- **Solution**: This is intentional - different setups are different positions

## Future Enhancements

Potential improvements:
- [ ] Trailing stop-loss adjustment
- [ ] Partial position closes (scale out)
- [ ] Time-based stops (close after N days)
- [ ] Volatility-adjusted stop widening
- [ ] Email/SMS alerts on position closes
- [ ] Live price monitoring (WebSocket integration)

## Summary

The automatic position monitoring system:
- ✅ Monitors all open positions every scan
- ✅ Auto-closes on stop/target hits
- ✅ Prevents duplicate positions
- ✅ Tracks P&L accurately
- ✅ Provides detailed logging
- ✅ Updates capital allocation in real-time

**Your portfolio is now self-managing!** 🎯

