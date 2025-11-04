# Capital and Position Size Fixes

## Problems Fixed

### 1. **Capital Was $100,000 Instead of $1,000** 💰

**Problem:**
- Portfolio had `total_capital: 100000`
- Positions totaled $92,496
- Should only have $1,000 capital!

**Root Cause:**
- `display_portfolio_status()` was calling `update_allocation(total_capital)` 
- This kept resetting capital to whatever was in the state file
- Once it got set to 100,000 (somehow), it stayed that way

**Solution:**
- ✅ Removed `update_allocation()` call from `display_portfolio_status()`
- ✅ Added `increment_run_count()` method to update run count without changing capital
- ✅ Reset `portfolio_state.json` to start fresh with $1,000

---

### 2. **position_size Should NOT Equal shares** 📊

**Problem:**
```json
{
  "position_size": 969,  // Was: number of shares
  "shares": 969,         // Also: number of shares (redundant!)
}
```

**What They Should Be:**
- `position_size`: **Dollar value** of position (shares × price)
- `shares`: **Number of shares** to buy

**Example - Correct Values:**
```json
{
  "ticker": "LOB",
  "entry_price": 31.29,
  "shares": 1,              // Number of shares
  "position_size": 31.29,   // Dollar value ($31.29 × 1 share)
  "kelly_fraction": 0.05    // 5% of $1000 = $50, so 1 share @ $31.29
}
```

**Root Cause:**
1. `TradePlan` dataclass had no `shares` field
2. `size_position()` calculated both `position_size` (dollars) and `shares` separately
3. But only `position_size` was stored in `TradePlan`
4. `add_position()` then treated `position_size` as if it was shares count!

**Solution:**
✅ Added `shares: int = 0` field to `TradePlan` dataclass
✅ Updated `create_trade_plan()` to store `shares` from `size_position()`
✅ Fixed `add_position()` to use `position_size` for capital check (dollars)
✅ Fixed `close_position()` to use `position_size` when returning capital

---

## Code Changes

### 1. **TradePlan Dataclass** (line 160-192)

**Before:**
```python
@dataclass
class TradePlan:
    position_size: float  # AMBIGUOUS!
    # No shares field!
```

**After:**
```python
@dataclass
class TradePlan:
    position_size: float  # Dollar value of position
    shares: int = 0       # Number of shares to buy
```

---

### 2. **create_trade_plan()** (line 2427-2447)

**Before:**
```python
# Calculate position size
position_size, shares = self.kelly_sizing.size_position(...)

# Create TradePlan
trade_plan = TradePlan(
    ...
    position_size=position_size,
    # shares NOT included! ❌
)
```

**After:**
```python
# Calculate position size
position_size, shares = self.kelly_sizing.size_position(...)

# Create TradePlan
trade_plan = TradePlan(
    ...
    position_size=position_size,  # Dollars
    shares=shares,                 # Number of shares ✅
)
```

---

### 3. **add_position()** (line 1064-1095)

**Before:**
```python
# WRONG: Calculated capital from shares × price
actual_capital_deployed = trade_plan.position_size * trade_plan.entry

# WRONG: Used position_size as shares
shares=int(trade_plan.position_size)
```

**After:**
```python
# Check if we have enough capital (position_size is in dollars)
if self.state.available_capital < trade_plan.position_size:
    return False

position = PortfolioPosition(
    ...
    position_size=trade_plan.position_size,  # Dollar value ✅
    shares=trade_plan.shares,                 # Number of shares ✅
)

# Deduct the dollar value from available capital
self.state.allocated_capital += trade_plan.position_size
self.state.available_capital -= trade_plan.position_size
```

---

### 4. **close_position()** (line 1201-1228)

**Before:**
```python
# Calculate P&L (WRONG: treated position_size as shares)
position.pnl = (exit_price - position.entry_price) * position.position_size

# Return capital (WRONG: recalculated from shares × price)
actual_capital_deployed = position.position_size * position.entry_price
self.state.available_capital += actual_capital_deployed + position.pnl
```

**After:**
```python
# Calculate P&L based on shares ✅
position.pnl = (exit_price - position.entry_price) * position.shares

# Return the original position_size (dollars) plus P&L ✅
self.state.available_capital += position.position_size + position.pnl
self.state.allocated_capital -= position.position_size
```

---

### 5. **display_portfolio_status()** (line 3485)

**Before:**
```python
self.console.print(table)

# WRONG: This kept resetting capital!
self.console.print(f"\n[blue]Updating allocation to ${summary['total_capital']:,.2f}...[/blue]")
self.portfolio_manager.update_allocation(summary['total_capital'])
self.console.print(f"[green]✅ Allocation updated successfully![/green]")
```

**After:**
```python
self.console.print(table)
# That's it! Just display, don't modify.
```

---

### 6. **New: increment_run_count()** (line 1058-1062)

```python
def increment_run_count(self) -> None:
    """Increment run count and update timestamp."""
    self.state.run_count += 1
    self.state.last_updated = datetime.now()
    self._save_state()
```

Called in main flow (line 5056-5057):
```python
# Increment run count
scanner.portfolio_manager.increment_run_count()
```

---

## Example: Correct Position Sizing

With **$1,000 capital** and **5% Kelly fraction**:

### Example Position 1: LOB @ $31.29
```python
kelly_fraction = 0.05  # 5%
position_size_dollars = $1000 × 0.05 = $50
shares = $50 / $31.29 = 1.59 → 1 share (integer)

# Stored as:
{
  "ticker": "LOB",
  "position_size": 50.0,      # $50 position
  "shares": 1,                 # Buy 1 share
  "entry_price": 31.29,        # @ $31.29
  "actual_cost": 31.29         # Actual: $31.29
}
```

**Capital after opening:**
- Available: $1000 - $31.29 = $968.71
- Allocated: $31.29
- Total: $1000

---

### Example Position 2: GGLL @ $83.90
```python
kelly_fraction = 0.076  # 7.6% (higher due to better setup)
position_size_dollars = $968.71 × 0.076 = $73.62
shares = $73.62 / $83.90 = 0.88 → 0 shares (TOO SMALL!)

# Position would be SKIPPED because:
# - Can't buy fractional shares
# - 0 shares = $0 position
```

---

### Example Position 3: MTEK @ $1.63
```python
kelly_fraction = 0.072  # 7.2%
position_size_dollars = $968.71 × 0.072 = $69.75
shares = $69.75 / $1.63 = 42.79 → 42 shares

# Stored as:
{
  "ticker": "MTEK",
  "position_size": 69.75,      # $69.75 position
  "shares": 42,                # Buy 42 shares
  "entry_price": 1.63,         # @ $1.63
  "actual_cost": 68.46         # Actual: $68.46
}
```

**Capital after opening:**
- Available: $968.71 - $68.46 = $900.25
- Allocated: $31.29 + $68.46 = $99.75
- Total: $1000

---

## Testing the Fix

### 1. Reset Portfolio
```bash
# portfolio_state.json is now:
{
  "total_capital": 1000,
  "available_capital": 1000.0,
  "allocated_capital": 0.0,
  "positions": []
}
```

### 2. Run Scanner
```bash
python official_scanner.py
```

### 3. Check Results

**Expected Behavior:**
```
💼 Current Portfolio Status:
Total Capital: $1,000.00
Available Capital: $900.25
Allocated Capital: $99.75

Open Positions: 2
- LOB: 1 shares @ $31.29 (position_size: $50.00) ✅
- MTEK: 42 shares @ $1.63 (position_size: $69.75) ✅
```

**Verify in JSON:**
```json
{
  "total_capital": 1000,
  "available_capital": 900.25,
  "allocated_capital": 99.75,
  "positions": [
    {
      "ticker": "LOB",
      "position_size": 50.0,     // Dollars ✅
      "shares": 1,                // Count ✅
      "entry_price": 31.29
    },
    {
      "ticker": "MTEK",
      "position_size": 69.75,    // Dollars ✅
      "shares": 42,               // Count ✅
      "entry_price": 1.63
    }
  ]
}
```

---

## Capital Flow Example

### Initial State
```
Total Capital: $1,000
Available: $1,000
Allocated: $0
```

### After Opening Position 1 (LOB)
```
Action: Buy 1 share @ $31.29

Total Capital: $1,000
Available: $1,000 - $31.29 = $968.71
Allocated: $31.29
```

### After Opening Position 2 (MTEK)
```
Action: Buy 42 shares @ $1.63 = $68.46

Total Capital: $1,000
Available: $968.71 - $68.46 = $900.25
Allocated: $31.29 + $68.46 = $99.75
```

### After Closing Position 1 (LOB exits @ $35.00)
```
Action: Sell 1 share @ $35.00
P&L: ($35.00 - $31.29) × 1 = +$3.71

Total Capital: $1,000 + $3.71 = $1,003.71
Available: $900.25 + $31.29 + $3.71 = $935.25
Allocated: $99.75 - $31.29 = $68.46
```

**Position value returned:** `position_size + pnl = $31.29 + $3.71 = $35.00` ✅

---

## Summary

| Field | Meaning | Unit | Example |
|-------|---------|------|---------|
| `position_size` | Dollar value of position | $ | $50.00 |
| `shares` | Number of shares | count | 1 |
| `entry_price` | Price per share | $/share | $31.29 |
| `kelly_fraction` | Position sizing % | % | 0.05 (5%) |
| `risk_dollars` | $ at risk (stop distance × shares) | $ | $2.58 |

**Relationship:**
```
kelly_fraction × available_capital = target position size
target position size / entry_price = shares (rounded down to integer)
shares × entry_price = actual position_size
```

---

## Files Modified

1. ✅ `official_scanner.py` - Fixed TradePlan, add_position(), close_position(), display_portfolio_status()
2. ✅ `portfolio_state.json` - Reset to $1,000 capital

**Total Changes:** ~60 lines modified across 6 methods

---

## Key Takeaways

1. **position_size = DOLLARS, shares = COUNT** 📊
2. **Start with $1,000, not $100,000** 💰
3. **Kelly sizing works on available capital** 🎯
4. **Capital tracking must be accurate** ✅
5. **Display status ≠ Modify state** 🚫

Your portfolio is now correctly sized for $1,000 capital! 🎉

