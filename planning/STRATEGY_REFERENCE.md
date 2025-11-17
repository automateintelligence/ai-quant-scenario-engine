# Strategy Reference Guide

## Available Strategies

### Stock Strategies

#### 1. `stock_basic` (Original Placeholder)
**File**: `quant_scenario_engine/strategies/stock_basic.py`
**Logic**: Simple dual moving average crossover
**Signal**: `1` when short_ma > long_ma, `-1` otherwise
**Status**: Placeholder - no real trading rules

#### 2. `stock_sma_trend` (Canonical Trend Following)
**File**: `quant_scenario_engine/strategies/stock_sma_trend.py`
**Logic**: 20/50 SMA trend following with position sizing
**Signal**: Long when SMA(20) > SMA(50), flat otherwise
**Position Sizing**: Targets $500-$1,000 daily P&L based on expected 2% daily move

**Parameters**:
- `short_window`: Short MA window (default: 20)
- `long_window`: Long MA window (default: 50)
- `target_profit_usd`: Target daily profit (default: 750)
- `expected_daily_move_pct`: Expected daily move (default: 0.02 = 2%)
- `max_position_usd`: Maximum position size (default: 50,000)

**Example**:
```bash
python -m quant_scenario_engine.cli.main compare \
  --symbol AAPL \
  --strategy stock_sma_trend \
  --option-strategy call_basic \
  --paths 1000 --steps 60 --seed 42
```

#### 3. `stock_rsi_reversion` (Mean Reversion)
**File**: `quant_scenario_engine/strategies/stock_rsi_reversion.py`
**Logic**: RSI(14) oversold mean reversion with position sizing
**Signal**: Long when RSI < 30, flat when RSI >= 50
**Position Sizing**: Targets $500-$1,000 daily P&L
**Requirements**: Requires `rsi` feature in features dict

**Parameters**:
- `oversold_threshold`: RSI oversold level (default: 30)
- `exit_threshold`: RSI exit level (default: 50)
- `target_profit_usd`: Target daily profit (default: 750)
- `expected_daily_move_pct`: Expected daily move (default: 0.02)
- `max_position_usd`: Maximum position size (default: 50,000)

**Example**:
```bash
python -m quant_scenario_engine.cli.main compare \
  --symbol AAPL \
  --strategy stock_rsi_reversion \
  --option-strategy option_atm_call_momentum \
  --paths 1000 --steps 60 --seed 42
```

**Note**: Currently requires RSI feature to be provided in features dict. Feature calculation pipeline needs implementation.

---

### Option Strategies

#### 4. `call_basic` (Original Placeholder)
**File**: `quant_scenario_engine/strategies/option_call.py`
**Logic**: Simple momentum (price vs rolling mean)
**Signal**: `1` when price >= rolling_mean, `0` otherwise
**Status**: Placeholder - no option-specific logic

#### 5. `option_atm_call_momentum` (ATM Call Momentum)
**File**: `quant_scenario_engine/strategies/option_atm_call_momentum.py`
**Logic**: ATM call momentum with dual SMA confirmation
**Signal**: Long call when price > SMA(20) AND SMA(20) > SMA(50)
**Position Sizing**: Targets $500-$1,000 daily P&L with assumed delta=0.5

**Parameters**:
- `short_window`: Short MA window (default: 20)
- `long_window`: Long MA window (default: 50)
- `target_profit_usd`: Target daily profit (default: 750)
- `expected_daily_move_pct`: Expected daily move (default: 0.02)
- `max_position_usd`: Maximum position size (default: 25,000)
- `assumed_delta`: Assumed option delta (default: 0.5)
- `max_iv_rank`: Max IV rank filter (default: 60.0, requires `iv_rank` feature)

**Example**:
```bash
python -m quant_scenario_engine.cli.main compare \
  --symbol AAPL \
  --strategy stock_sma_trend \
  --option-strategy option_atm_call_momentum \
  --strike 180 --iv 0.25 --maturity-days 30 \
  --paths 1000 --steps 60 --seed 42
```

#### 6. `option_atm_put_rsi` (ATM Put RSI)
**File**: `quant_scenario_engine/strategies/option_atm_put_rsi.py`
**Logic**: ATM put bought on deep oversold RSI
**Signal**: Long put when RSI < 25, flat otherwise
**Position Sizing**: Targets $500-$1,000 daily P&L with assumed delta=0.5
**Requirements**: Requires `rsi` feature in features dict

**Parameters**:
- `oversold_threshold`: RSI oversold level (default: 25)
- `target_profit_usd`: Target daily profit (default: 750)
- `expected_daily_move_pct`: Expected daily move (default: 0.03 = 3%)
- `max_position_usd`: Maximum position size (default: 25,000)
- `assumed_delta`: Assumed option delta (default: 0.5)

**Example**:
```bash
python -m quant_scenario_engine.cli.main compare \
  --symbol AAPL \
  --strategy stock_rsi_reversion \
  --option-strategy option_atm_put_rsi \
  --strike 180 --iv 0.25 --maturity-days 21 \
  --paths 1000 --steps 60 --seed 42
```

**Note**: Currently requires RSI feature to be provided in features dict. Feature calculation pipeline needs implementation.

---

## Position Sizing

All canonical strategies use shared position sizing utilities from `quant_scenario_engine/strategies/position_sizing.py`.

### Stock Position Sizing
Formula: `shares ≈ target_profit / (price * expected_daily_move_pct)`

### Option Position Sizing
Formula: `contracts ≈ target_profit / (100 * delta * price * expected_daily_move_pct)`

Both respect `max_position_usd` caps to limit risk exposure.

---

## Strategy Registration

Strategies are registered in `quant_scenario_engine/strategies/factory.py`:

```python
from quant_scenario_engine.strategies.factory import get_strategy

# Load stock strategy
stock_strat = get_strategy("stock_sma_trend", kind="stock")

# Load option strategy (requires OptionSpec)
from quant_scenario_engine.models.options import OptionSpec
option_spec = OptionSpec(
    option_type="call",
    strike=100.0,
    maturity_days=30,
    implied_vol=0.2,
    risk_free_rate=0.01,
    contracts=1
)
option_strat = get_strategy("option_atm_call_momentum", kind="option", option_spec=option_spec)
```

---

## Testing Strategy Implementations

### Quick Test (50 paths, 30 steps)
```bash
python -m quant_scenario_engine.cli.main compare \
  --symbol TEST --s0 100 --paths 50 --steps 30 \
  --strategy stock_sma_trend \
  --option-strategy option_atm_call_momentum \
  --strike 100 --iv 0.2 --seed 42
```

### Full Backtest (1000 paths, 60 steps)
```bash
python -m quant_scenario_engine.cli.main compare \
  --symbol AAPL --s0 180 --paths 1000 --steps 60 \
  --strategy stock_sma_trend \
  --option-strategy option_atm_call_momentum \
  --strike 180 --iv 0.25 --maturity-days 30 \
  --distribution laplace --seed 42
```

---

## Feature Computation System ✅

**Implemented**: `quant_scenario_engine/features/technical.py` provides comprehensive technical indicator calculations.

### Automatic Feature Computation

Features are **automatically computed** when running compare:
- RSI strategies get RSI calculated automatically from price paths
- Feature computation happens transparently in `run_compare()`
- No manual feature calculation needed for standard workflows

### Available Technical Indicators

**Computed by `compute_all_features()`**:
- **RSI**: Relative Strength Index (period 14)
- **SMA**: Simple Moving Averages (20, 50, 200)
- **EMA**: Exponential Moving Averages (12, 26)
- **MACD**: Moving Average Convergence Divergence + signal + histogram
- **Bollinger Bands**: Upper, middle, lower bands
- **ATR**: Average True Range
- **Stochastic**: %K and %D oscillators

### Usage

**Option 1: Automatic (Recommended)**
```python
result = run_compare(
    stock_strategy="stock_rsi_reversion",  # Requires RSI
    option_strategy="option_atm_put_rsi",  # Requires RSI
    compute_features=True,  # Default: automatically computes all features
    # ... other params ...
)
```

**Option 2: Pre-computed Features**
```python
from quant_scenario_engine.features.technical import compute_all_features

features = compute_all_features(price_paths, fillna=True)
# Customize or add features here...

result = run_compare(
    features=features,  # Use pre-computed features
    compute_features=False,
    # ... other params ...
)
```

**Option 3: Custom Features**
```python
from quant_scenario_engine.features.technical import compute_rsi

features = {
    "rsi": compute_rsi(price_paths, period=14),
    # Add only what you need...
}

result = run_compare(
    features=features,
    compute_features=False,
    # ... other params ...
)
```

### Strategies Requiring Features

| Strategy | Required Features | Optional Features |
|----------|-------------------|-------------------|
| `stock_rsi_reversion` | `rsi` | None |
| `option_atm_put_rsi` | `rsi` | None |
| `option_atm_call_momentum` | None | `iv_rank` (IV rank filter) |
| All other strategies | None | None |

---

## Strategy Comparison Matrix

| Strategy | Type | Logic | Position Sizing | Features Required |
|----------|------|-------|-----------------|-------------------|
| `stock_basic` | Stock | Dual MA | None (placeholder) | None |
| `stock_sma_trend` | Stock | 20/50 SMA trend | $500-$1k target | None |
| `stock_rsi_reversion` | Stock | RSI oversold | $500-$1k target | `rsi` |
| `call_basic` | Option | Simple momentum | None (placeholder) | None |
| `option_atm_call_momentum` | Option | Call + dual SMA | $500-$1k target | None (`iv_rank` optional) |
| `option_atm_put_rsi` | Option | Put + RSI oversold | $500-$1k target | `rsi` |

---

## Files Added

### Strategies
1. `quant_scenario_engine/strategies/position_sizing.py` - Position sizing utilities
2. `quant_scenario_engine/strategies/stock_sma_trend.py` - SMA trend following
3. `quant_scenario_engine/strategies/stock_rsi_reversion.py` - RSI mean reversion
4. `quant_scenario_engine/strategies/option_atm_call_momentum.py` - ATM call momentum
5. `quant_scenario_engine/strategies/option_atm_put_rsi.py` - ATM put RSI

### Feature Computation
6. `quant_scenario_engine/features/technical.py` - Technical indicator calculations (RSI, SMA, EMA, MACD, Bollinger Bands, ATR, Stochastic)

### Tests
7. `tests/integration/test_strategies_with_features.py` - Integration tests for strategies with feature computation

**Modified**:
- `quant_scenario_engine/strategies/factory.py` - Registered new strategies
- `quant_scenario_engine/simulation/compare.py` - Added automatic feature computation
