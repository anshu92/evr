# Position Management Enhancements

## Overview

The EVR scanner now includes intelligent position management with two powerful features:

1. **Time-Based Exits** (Option 3) - Automatically close positions after a maximum holding period
2. **Position Replacement Logic** (Option 2) - Replace underperforming positions with better opportunities

## Features

### 1. Time-Based Exits

Positions are automatically closed after exceeding the maximum holding period, preventing capital from being tied up in stagnant positions.

**Default:** 7 days

**Configuration:**
```bash
python official_scanner.py --max-holding-days 5
```

**How it works:**
- Every scan checks how many days each position has been held
- If `days_held >= max_holding_days`, the position is closed at current market price
- Exit reason: `TIME_EXIT`
- P&L is calculated and capital is returned to available pool

**Example output:**
```
⏰ Time exit ORN @ $10.95 (held 7 days) - Entry: $10.88, P&L: $+1.40
```

### 2. Position Replacement Logic

When the portfolio is full, the scanner compares new opportunities against existing positions and replaces underperforming ones if significantly better opportunities appear.

**Default:** Enabled, 20% improvement threshold

**Configuration:**
```bash
# Enable with custom threshold (15% improvement required)
python official_scanner.py --enable-replacement --replacement-threshold 0.15

# Disable replacement
python official_scanner.py --no-replacement
```

**How it works:**

1. **Portfolio Full Check**: When portfolio has max positions (5 by default), evaluate replacement
2. **Score Calculation**: Calculate EVR score for each position:
   ```
   EVR_score = P(Win) × Expected_Return × Kelly_Fraction
   ```
3. **Compare Opportunities**: Compare top 3 new recommendations against existing positions
4. **Threshold Check**: Replace if new opportunity is ≥20% better (default)
5. **Limit Churn**: Maximum 2 replacements per scan to avoid excessive trading
6. **Close & Replace**: Close underperforming position, add new position

**Example comparison:**

| Position | Ticker | EVR Score | Days Held | Status |
|----------|--------|-----------|-----------|--------|
| Current  | AMH    | 0.0113    | 6 days    | Weakest position |
| New      | NOC    | 0.0220    | -         | +94% better → **REPLACE!** |

**Example output:**
```
⚠️  Portfolio full (5/5 positions)
🔄 Evaluating position replacement opportunities...
🔄 Replaced AMH (EVR: 0.011, held 6d) with NOC (EVR: 0.022, +94.2% better) - P&L: $-0.45
✅ Replaced 1 position(s)
```

## Usage Examples

### Default Settings
```bash
# Time-based exits after 7 days, replacement enabled with 20% threshold
python official_scanner.py
```

### Conservative (Hold Longer, Higher Threshold)
```bash
# Hold up to 14 days, require 30% improvement to replace
python official_scanner.py --max-holding-days 14 --replacement-threshold 0.30
```

### Aggressive (Quick Exits, Lower Threshold)
```bash
# Exit after 3 days, replace with 10% improvement
python official_scanner.py --max-holding-days 3 --replacement-threshold 0.10
```

### Disable Replacement (Time Exits Only)
```bash
# Only use time-based exits, no position replacement
python official_scanner.py --no-replacement
```

## Position Exit Reasons

The scanner now tracks multiple exit reasons:

| Reason | Description | Trigger |
|--------|-------------|---------|
| `STOPPED_OUT` | Stop loss hit | Price ≤ Stop (Long) or Price ≥ Stop (Short) |
| `TARGET_HIT` | Take profit hit | Price ≥ Target (Long) or Price ≤ Target (Short) |
| `TIME_EXIT` | Max holding period | days_held ≥ max_holding_days |
| `REPLACED` | Position upgraded | Better opportunity found (EVR improvement ≥ threshold) |
| `MANUAL` | Manual close | User intervention |

## Monitoring Output

Enhanced monitoring display shows all exit types:

```
📊 Monitoring Open Positions...
Monitored: 5 | Closed: 2 | Stopped Out: 0 | Targets Hit: 1 | Time Exits: 1 | Errors: 0
```

## Configuration Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-holding-days` | 7 | Days before time exit |
| `--enable-replacement` | True | Enable position replacement |
| `--no-replacement` | - | Disable position replacement |
| `--replacement-threshold` | 0.20 | Min improvement to replace (20%) |

## Benefits

### Time-Based Exits
✅ Prevents capital stagnation  
✅ Forces portfolio turnover  
✅ Ensures fresh opportunities  
✅ Reduces opportunity cost  

### Position Replacement
✅ Upgrades portfolio dynamically  
✅ Maintains best opportunities  
✅ Adapts to market changes  
✅ Maximizes expected value  
✅ Limits over-trading (max 2 per run)  

## Example Scenarios

### Scenario 1: Position Times Out
```
Day 1: Open AMH @ $31.45, Expected Return: 2.87%
Day 7: No stop/target hit, but held 7 days → TIME_EXIT
Result: Close @ $31.60, P&L: +$0.90, Capital available for new opportunities
```

### Scenario 2: Better Opportunity Appears
```
Current: AMH (EVR: 0.011, Expected Return: 2.87%, 6 days old)
New Scan: NOC (EVR: 0.022, Expected Return: 2.94%, 8 signals)
Improvement: +94% → REPLACE!
Result: Close AMH, Open NOC
```

### Scenario 3: No Replacement (Below Threshold)
```
Current: HP (EVR: 0.030, Expected Return: 5.21%, 2 days old)
New Scan: MIDD (EVR: 0.033, Expected Return: 3.60%, 7 signals)
Improvement: +10% → Below 20% threshold → KEEP HP
Result: No changes
```

## Safety Features

1. **Max Replacements per Run**: Limited to 2 to avoid excessive churn
2. **Improvement Threshold**: Requires significant improvement (default 20%)
3. **Days Held Consideration**: Shown in replacement decision
4. **P&L Tracking**: All exits record P&L for performance analysis
5. **Error Handling**: Graceful fallback if data fetch fails

## Integration with Existing Features

Both features work seamlessly with:
- ✅ Stop loss monitoring
- ✅ Target hit monitoring  
- ✅ Position sizing
- ✅ Capital management
- ✅ Performance tracking
- ✅ Portfolio state persistence

## Real-World Impact

**Before:**
- Portfolio stuck with AMH (2.87% expected return, 6 days old)
- NOC opportunity (2.94%, 8 signals, EVR 0.744) ignored
- No action taken → Portfolio stagnates

**After:**
- System identifies NOC is 94% better than AMH
- Closes AMH at current price (captures small gain/loss)
- Opens NOC position automatically
- Portfolio now has highest EVR opportunities

## Best Practices

1. **Start Conservative**: Use defaults (7 days, 20% threshold)
2. **Monitor Results**: Track replacement frequency and P&L
3. **Adjust Gradually**: Lower threshold if too few replacements, raise if too many
4. **Match Strategy**: Day traders use shorter holds, swing traders use longer
5. **Review Performance**: Check if time exits and replacements improve returns

## Future Enhancements

Potential additions:
- Volatility-adjusted holding periods
- Setup-specific time limits
- Dynamic threshold based on market conditions
- Machine learning for optimal replacement timing
- Risk-adjusted replacement scores

---

**Note:** These features are designed to keep your portfolio dynamic and optimized. Start with defaults and adjust based on your trading style and results.

