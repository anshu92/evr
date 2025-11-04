# Setup Name Mapping Fix

## Problem

The trained parameters weren't being used because of a **setup name mismatch** between the scanner and the trainer:

### Scanner Setup Names (official_scanner.py)
- `rsi_oversold`
- `rsi_overbought`
- `macd_bullish`
- `bb_breakout`
- `stoch_oversold`
- `williams_oversold`
- `cci_oversold`
- `support_bounce`
- `hammer_pattern`
- `engulfing_pattern`
- `strong_uptrend`
- etc.

### Trainer Setup Names (historical_parameter_trainer.py)
- `RSI_Oversold_Long`
- `RSI_Overbought_Short`
- `MACD_Cross_Long`
- `BB_Bounce_Long`
- `Trend_Following_Long`
- `Mean_Reversion_Short`

## Solution

Added setup name normalization in `parameter_integration.py` to map scanner setup names to trainer setup names:

### Key Changes

1. **Added `_normalize_setup_name()` method** (line 85-132)
   - Normalizes setup names by removing underscores and spaces
   - Maps scanner names to corresponding trainer names
   - Returns list of possible matches to try

2. **Enhanced `get_setup_parameters()` method** (line 134-159)
   - First tries exact match
   - Falls back to normalized name matching
   - Logs when a mapping is used

3. **Added missing `List` import** (line 12)
   - Fixed `NameError: name 'List' is not defined`

## Setup Name Mappings

| Scanner Setup | Trainer Setup(s) |
|--------------|------------------|
| `rsi_oversold` | `RSI_Oversold_Long` |
| `rsi_overbought` | `RSI_Overbought_Short`, `Mean_Reversion_Short` |
| `macd_bullish`, `macd_cross` | `MACD_Cross_Long` |
| `bb_breakout`, `bb_bounce` | `BB_Bounce_Long` |
| `strong_uptrend`, `trend_following`, `ma_crossover` | `Trend_Following_Long` |
| `volume_momentum`, `volatility_breakout` | `Trend_Following_Long` |
| `stoch_oversold`, `williams_oversold`, `cci_oversold` | `RSI_Oversold_Long` |
| `stoch_overbought`, `williams_overbought`, `cci_overbought` | `Mean_Reversion_Short` |
| `support_bounce`, `hammer_pattern`, `doji_pattern` | `RSI_Oversold_Long` |
| `engulfing_pattern` | `Trend_Following_Long` |
| `resistance_rejection` | `Mean_Reversion_Short` |
| `mean_reversion` | `Mean_Reversion_Short` |

## Expected Results

After this fix, when you run `python official_scanner.py`, you should see:

1. **Non-50% P(Win) values** based on trained parameters:
   - `BB_Bounce_Long`: ~70.7% win rate ✅
   - `Trend_Following_Long`: ~54.6% win rate ✅
   - `RSI_Oversold_Long`: varies
   - `MACD_Cross_Long`: ~45% win rate
   - `Mean_Reversion_Short`: ~23% win rate (avoid!)

2. **Debug logs** showing name mapping (if debug logging is enabled):
   ```
   Mapped setup 'rsi_oversold' to trained setup 'RSI_Oversold_Long'
   Mapped setup 'bb_breakout' to trained setup 'BB_Bounce_Long'
   ```

3. **Varying Expected Returns** based on actual historical performance instead of uniform values

## How It Works

When the scanner generates a signal with setup name `rsi_oversold`:

1. `EnhancedRollingBayes.estimate('rsi_oversold', ...)` is called
2. Calls `param_loader.get_setup_parameters('rsi_oversold')`
3. Exact match fails (no `rsi_oversold` in trained params)
4. Calls `_normalize_setup_name('rsi_oversold')`
   - Normalizes to `'rsioversold'`
   - Looks up in mapping → returns `['RSI_Oversold_Long']`
5. Finds `RSI_Oversold_Long` in trained parameters ✅
6. Returns trained `p_win`, `avg_win`, `avg_loss` for that setup
7. Blends with real trading data (if any)
8. Since `num_trades = 0`, uses 100% trained parameters

## Testing

Run the scanner and check the P(Win) column:

```bash
python official_scanner.py
```

You should now see **varying P(Win) values** like:
- 70.7% (for BB bounce setups)
- 54.6% (for trend following)
- 51.6% (global average)
- 45% (for MACD cross)
- 23% (for mean reversion shorts)

Instead of uniform 50% for everything!

## Files Modified

1. `/Users/sahooa3/Documents/git/evr/parameter_integration.py`
   - Added setup name normalization
   - Fixed missing `List` import
   - Enhanced `get_setup_parameters()` with fallback matching

2. `/Users/sahooa3/Documents/git/evr/historical_parameter_trainer.py`
   - Previously fixed: Added missing `TaskProgressColumn` import

## Notes

- The mapping is **many-to-one**: Multiple scanner setups can map to the same trainer setup
- This is intentional because the trainer uses higher-level strategy categories
- For example, `stoch_oversold`, `williams_oversold`, and `rsi_oversold` all map to `RSI_Oversold_Long`
- This provides better statistical power by combining similar setups

