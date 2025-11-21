# StrategyOptimizer User Guide

**Complete CLI Workflow for Option Strategy Discovery**

## Table of Contents
1. [Quick Start](#quick-start)
2. [Configuration Setup](#configuration-setup)
3. [Complete Workflow](#complete-workflow)
4. [Understanding Output](#understanding-output)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

**What You Need**:
- One stock ticker (e.g., NVDA, TSLA, SPY)
- A price action regime assumption (e.g., "strong-bullish", "neutral")
- 5 minutes to set up your config file

**What Happens Automatically**:
- ✅ Fetches option chain data from Schwab API (with yfinance fallback)
- ✅ Models stock pricing from historical data using distribution fit testing
- ✅ Auto-filters and selects option spread candidates based on regime
- ✅ Generates option greeks and pricing models
- ✅ Performs vector-based selection/optimization using custom scoring
- ✅ Measures trading horizon and expiration pricing for each candidate
- ✅ Down-selects to Top-100 and Top-10 candidates
- ✅ Returns Top-10 with performance diagnostics and analysis summary

**Simple Example**:
```bash
# 1. Create config file (see below)
# 2. Run optimizer
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1

# Done! Top-10 strategies ranked and ready in seconds
```

---

## Configuration Setup

### Step 1: Create Your Config File

Create `my_config.yml` in your project root:

```yaml
# ====================================================================
# STRATEGY OPTIMIZER CONFIGURATION
# ====================================================================

# --------------------------------------------------------------------
# REGIME DEFINITIONS - Map qualitative market views to MC parameters
# --------------------------------------------------------------------
regimes:
  # Neutral: Flat price action, low volatility
  neutral:
    mean_daily_return: 0.0     # 0% expected daily move
    daily_vol: 0.01            # 1% daily volatility
    skew: 0.0                  # Symmetric distribution
    kurtosis_excess: 1.0       # Moderate tail thickness

  # Strong Neutral: Very tight range, minimal movement
  strong-neutral:
    mean_daily_return: 0.0
    daily_vol: 0.005           # 0.5% daily volatility
    skew: 0.0
    kurtosis_excess: 0.5

  # Low Volatility: Stable, predictable moves
  low-volatility:
    mean_daily_return: 0.0
    daily_vol: 0.008           # 0.8% daily volatility
    skew: 0.0
    kurtosis_excess: 0.8

  # Volatility Direction Uncertain: Wide swings either way
  volatility-dir-uncertain:
    mean_daily_return: 0.0
    daily_vol: 0.03            # 3% daily volatility
    skew: 0.0                  # No directional bias
    kurtosis_excess: 2.0       # Fat tails

  # Mild Bearish: Slight downward drift
  mild-bearish:
    mean_daily_return: -0.01   # -1% expected daily move
    daily_vol: 0.02            # 2% daily volatility
    skew: -0.3                 # Left skew (downside bias)
    kurtosis_excess: 1.5

  # Strong Bearish: Significant downward movement
  strong-bearish:
    mean_daily_return: -0.03   # -3% expected daily move
    daily_vol: 0.04            # 4% daily volatility
    skew: -0.5                 # Strong left skew
    kurtosis_excess: 2.5       # Very fat tails

  # Mild Bullish: Moderate upward drift
  mild-bullish:
    mean_daily_return: 0.01    # +1% expected daily move
    daily_vol: 0.02            # 2% daily volatility
    skew: 0.3                  # Right skew (upside bias)
    kurtosis_excess: 1.5

  # Strong Bullish: Aggressive upward movement
  strong-bullish:
    mean_daily_return: 0.02    # +2% expected daily move
    daily_vol: 0.03            # 3% daily volatility
    skew: 0.5                  # Strong right skew
    kurtosis_excess: 2.0       # Fat tails

# --------------------------------------------------------------------
# MONTE CARLO SIMULATION SETTINGS
# --------------------------------------------------------------------
mc:
  num_paths: 5000              # Number of MC paths (5k baseline, up to 20k adaptive)
  seed: 42                     # Random seed for reproducibility
  bars_per_day: 1              # Timesteps per trading day (1 = daily bars)
  adaptive_paths: true         # Enable adaptive path doubling for high variance
  max_paths: 20000             # Maximum paths in adaptive mode
  ci_threshold_pnl: 100        # CI width threshold for E[PnL] ($100)
  ci_threshold_pop: 0.03       # CI width threshold for POP (3 percentage points)

# --------------------------------------------------------------------
# FILTERING & CONSTRAINTS
# --------------------------------------------------------------------
filters:
  # Stage 0: Expiry Selection
  min_dte: 7                   # Minimum days to expiration
  max_dte: 45                  # Maximum days to expiration
  expiry_count_target: 4       # Target number of expiries (3-5 range)

  # Stage 1: Strike Selection
  moneyness_min: 0.85          # Min strike/spot ratio (15% OTM puts)
  moneyness_max: 1.15          # Max strike/spot ratio (15% OTM calls)
  min_volume: 100              # Minimum option volume
  min_open_interest: 100       # Minimum open interest
  max_bid_ask_spread_pct: 0.15 # Max bid-ask spread (15% of mid)

  # Stage 2: Structure Generation
  vertical_width_min: 1        # Min vertical spread width (strikes)
  vertical_width_max: 3        # Max vertical spread width (strikes)

  # Stage 3: Hard Constraints (Analytic Prefilter)
  max_capital: 15000           # Maximum capital per trade ($15k)
  max_loss_pct: 0.50           # Max loss as % of capital (50% for SHORT premium)
  min_expected_pnl: 500        # Minimum expected profit ($500)
  min_pop_breakeven: 0.55      # Minimum POP at breakeven (55%)
  min_pop_target: 0.30         # Minimum POP at profit target (30%)
  top_k_per_type: 20           # Top-K candidates per structure type

# --------------------------------------------------------------------
# SCORING WEIGHTS (Intraday-Spreads Composite Score)
# --------------------------------------------------------------------
scoring:
  # Positive Contributions
  w_pop: 0.35                  # Weight for probability of profit
  w_roc: 0.30                  # Weight for return on capital
  w_theta: 0.10                # Weight for positive time decay

  # Penalties (Negative Contributions)
  w_tail: 0.15                 # Penalty for tail risk (max loss)
  w_delta: 0.05                # Penalty for delta misalignment (neutral target)
  w_gamma: 0.03                # Penalty for gamma instability
  w_vega: 0.02                 # Penalty for vega exposure

  # Normalization Scales
  delta_target: 0.0            # Target delta (0 for neutral spreads)
  delta_scale: 0.5             # Delta normalization scale
  gamma_scale: 0.1             # Gamma normalization scale
  vega_scale: 50.0             # Vega normalization scale
  theta_scale: 30.0            # Theta normalization scale

  # Profit Target
  profit_target: 500           # Profit target for POP_target ($500)

# --------------------------------------------------------------------
# TRANSACTION COSTS
# --------------------------------------------------------------------
costs:
  commission_per_contract: 0.65   # Commission per option contract ($0.65)
  fill_mode: "pay_spread"         # Entry fill: buy at ask, sell at bid

# --------------------------------------------------------------------
# DATA SOURCE
# --------------------------------------------------------------------
data_source:
  primary: "schwab"            # Primary data provider (Schwab API)
  fallback: "yfinance"         # Fallback provider (Yahoo Finance)
  cache_ttl: 300               # Cache time-to-live (5 minutes)
  timeout: 30                  # API timeout (30 seconds)
  retry_attempts: 3            # Retry attempts on failure

# --------------------------------------------------------------------
# DISTRIBUTION MODEL
# --------------------------------------------------------------------
distribution:
  engine: "student_t"          # Distribution engine (student_t, garch_t, laplacian)
  df_default: 5                # Degrees of freedom for Student-t (heavier tails)

# --------------------------------------------------------------------
# PRICING MODEL
# --------------------------------------------------------------------
pricing:
  model: "black_scholes"       # Pricing model (black_scholes, bjerksund_stensland)
  iv_evolution: "constant"     # IV evolution (constant, sticky_delta, custom)
  risk_free_rate: 0.04         # Risk-free rate (4%)
  dividend_yield: 0.0          # Dividend yield (0% default)
```

### Step 2: Customize Your Config

**For Conservative Traders** (Higher POP, Lower Risk):
```yaml
filters:
  max_capital: 10000           # Reduce capital
  max_loss_pct: 0.30           # Tighter loss limit (30%)
  min_pop_breakeven: 0.65      # Higher POP requirement (65%)

scoring:
  w_pop: 0.50                  # Increase POP weight
  w_roc: 0.20                  # Decrease ROC weight
  w_tail: 0.20                 # Increase tail penalty
```

**For Aggressive Traders** (Higher ROC, Accept More Risk):
```yaml
filters:
  max_capital: 20000           # Increase capital
  max_loss_pct: 0.60           # Looser loss limit (60%)
  min_pop_breakeven: 0.50      # Lower POP requirement (50%)

scoring:
  w_pop: 0.25                  # Decrease POP weight
  w_roc: 0.40                  # Increase ROC weight
  w_tail: 0.10                 # Decrease tail penalty
```

---

## Complete Workflow

### Step 1: Run the Optimizer

```bash
# Basic usage
qse optimize-strategy \
  --ticker NVDA \
  --regime strong-bullish \
  --trade-horizon 1 \
  --config my_config.yml

# With custom overrides
qse optimize-strategy \
  --ticker NVDA \
  --regime strong-bullish \
  --trade-horizon 1 \
  --override "mc.num_paths=10000" \
  --override "filters.max_capital=20000" \
  --override "scoring.w_pop=0.50"

# For 3-day holding period
qse optimize-strategy \
  --ticker TSLA \
  --regime neutral \
  --trade-horizon 3 \
  --config my_config.yml
```

### Step 2: What Happens Automatically

#### Stage 0: Data Collection & Expiry Selection
```
[INFO] Fetching option chain for NVDA from Schwab API...
[INFO] Received 1,247 option contracts across 8 expiries
[INFO] Stage 0: Selected 4 expiries with DTE in [7, 45] range
  ✓ Expiry 1: 7 DTE (2025-11-28)
  ✓ Expiry 2: 14 DTE (2025-12-05)
  ✓ Expiry 3: 21 DTE (2025-12-12)
  ✓ Expiry 4: 35 DTE (2025-12-26)
```

#### Stage 1: Strike Filtering (Moneyness & Liquidity)
```
[INFO] Stage 1: Filtering strikes by moneyness [0.85, 1.15] and liquidity
  Current spot: $500.00
  Moneyness window: [$425.00, $575.00]
  ✓ Retained 10 strikes per expiry meeting liquidity criteria
    - Min volume: 100 ✓
    - Min OI: 100 ✓
    - Max spread: 15% ✓
```

#### Stage 2: Structure Generation
```
[INFO] Stage 2: Generating candidate structures
  ✓ Verticals (width 1-3): 180 candidates
  ✓ Iron Condors: 120 candidates
  ✓ Straddles: 30 candidates
  ✓ Strangles: 45 candidates
  Total candidates: 375
```

#### Stage 3: Analytic Prefilter (Hard Constraints)
```
[INFO] Stage 3: Applying analytic prefilter with Black-Scholes
  Using Normal distribution approximation for quick scoring
  Hard filters:
    ✓ Capital ≤ $15,000
    ✓ MaxLoss/Capital ≤ 50%
    ✓ E[PnL] ≥ $500
    ✓ POP_breakeven ≥ 55%
    ✓ POP_target ≥ 30%

  Rejection breakdown:
    - 77 rejected by capital filter (>$15,000)
    - 33 rejected by MaxLoss filter (>50%)
    - 76 rejected by E[PnL] filter (<$500)
    - 35 rejected by POP filter (<55%)

  ✓ 154 candidates passed hard filters
  ✓ Selected top-105 for Stage 4 (30 verticals, 40 condors, 20 straddles, 15 strangles)
```

#### Stage 4: Full Monte Carlo Scoring
```
[INFO] Stage 4: Running full MC scoring with GARCH-t distribution
  Regime: strong-bullish (mean=2%, vol=3%, skew=0.5)
  MC paths: 5,000 baseline (adaptive up to 20,000 if needed)
  Pricer: Black-Scholes with constant IV
  Horizon: 1 trading day

  Progress: [████████████████████████████] 105/105 candidates scored

  ✓ All 105 candidates scored
  ✓ Top-10 ranked by composite score (intraday-spreads)
  ✓ Top-100 cached for analysis

  Runtime: 4.9 seconds
```

### Step 3: Review Top-10 Output

The optimizer returns a JSON result with three main sections:

1. **top10**: Top 10 ranked strategies (displayed below)
2. **top100**: Top 100 cached strategies (stored internally)
3. **diagnostics**: Stage counts, rejection breakdowns, runtime

**Example Top-10 Output**:
```json
{
  "top10": [
    {
      "rank": 1,
      "structure_type": "iron_condor",
      "legs": [
        {"strike": 480, "type": "put", "side": "sell", "quantity": 1, "fill_price": 8.20},
        {"strike": 485, "type": "put", "side": "buy", "quantity": 1, "fill_price": 9.50},
        {"strike": 515, "type": "call", "side": "sell", "quantity": 1, "fill_price": 7.80},
        {"strike": 520, "type": "call", "side": "buy", "quantity": 1, "fill_price": 6.50}
      ],
      "metrics": {
        "expected_pnl": 642,
        "ci_pnl": [580, 704],
        "pop_breakeven": 0.74,
        "pop_target": 0.68,
        "roc": 0.042,
        "capital": 15000,
        "max_loss": 4350,
        "var_5pct": -1200,
        "cvar_5pct": -1850,
        "greeks": {
          "delta": -0.03,
          "theta": 28.0,
          "gamma": -0.04,
          "vega": -18.0
        },
        "score": 0.782,
        "score_breakdown": {
          "pop_contribution": 0.274,
          "roc_contribution": 0.235,
          "theta_contribution": 0.093,
          "tail_penalty": -0.145,
          "delta_penalty": -0.015,
          "gamma_penalty": -0.012,
          "vega_penalty": -0.007
        },
        "mc_paths": 5000
      },
      "diagnostics": {
        "entry_cost": 89.60,
        "exit_cost_estimate": 92.00,
        "net_credit": 1500,
        "breakeven_upper": 516.50,
        "breakeven_lower": 483.50,
        "max_profit": 1500,
        "max_loss": 3500,
        "days_to_expiry": 21
      }
    },
    {
      "rank": 2,
      "structure_type": "short_strangle",
      "legs": [
        {"strike": 475, "type": "put", "side": "sell", "quantity": 1, "fill_price": 12.50},
        {"strike": 525, "type": "call", "side": "sell", "quantity": 1, "fill_price": 11.80}
      ],
      "metrics": {
        "expected_pnl": 580,
        "ci_pnl": [510, 650],
        "pop_breakeven": 0.68,
        "pop_target": 0.62,
        "roc": 0.038,
        "capital": 15000,
        "max_loss": 6500,
        "var_5pct": -2100,
        "cvar_5pct": -3200,
        "greeks": {
          "delta": -0.02,
          "theta": 32.0,
          "gamma": -0.06,
          "vega": -22.0
        },
        "score": 0.745,
        "score_breakdown": {
          "pop_contribution": 0.245,
          "roc_contribution": 0.228,
          "theta_contribution": 0.107,
          "tail_penalty": -0.195,
          "delta_penalty": -0.010,
          "gamma_penalty": -0.018,
          "vega_penalty": -0.009
        },
        "mc_paths": 5000
      }
    }
    // ... ranks 3-10 ...
  ],
  "diagnostics": {
    "stage_counts": {
      "Stage 0 (expiries)": 4,
      "Stage 1 (strikes)": 10,
      "Stage 2 (structures)": 375,
      "Stage 3 (survivors)": 105,
      "Stage 4 (MC scored)": 105
    },
    "rejections": {
      "capital_filter": 77,
      "max_loss_filter": 33,
      "epnl_filter": 76,
      "pop_filter": 35
    },
    "runtime_seconds": 4.9,
    "trade_horizon_days": 1,
    "regime": "strong-bullish",
    "ticker": "NVDA",
    "spot_price": 500.0
  }
}
```

---

## Understanding Output

### Key Metrics Explained

#### Expected P&L (`expected_pnl`)
- **What**: Average profit/loss across all MC paths
- **Interpretation**: Higher is better; $500-$1000 typical for intraday spreads
- **Confidence Interval** (`ci_pnl`): 95% CI; narrower = more certain estimate

#### Probability of Profit (`pop_breakeven`, `pop_target`)
- **POP Breakeven**: Chance of making $0.01 or more
- **POP Target**: Chance of hitting profit target ($500 default)
- **Interpretation**: Higher is better; 60-80% typical for neutral spreads

#### Return on Capital (`roc`)
- **What**: Expected profit as % of capital deployed
- **Interpretation**: Higher is better; 3-5% typical for 1-day horizon
- **Annualized**: ~750-1250% if compounded (unrealistic but illustrative)

#### Tail Risk Metrics
- **Max Loss**: Worst-case loss from worst MC path
- **VaR 5%**: 5th percentile loss (95% of paths do better)
- **CVaR 5%**: Average loss in worst 5% of paths

#### Greeks
- **Delta**: Directional exposure ($1 move = Delta × $100 P&L per contract)
  - Target: ~0 for neutral spreads
  - Range: -1 (full bearish) to +1 (full bullish)
- **Theta**: Daily time decay (positive = earning decay)
  - Target: +$20-$40/day for short premium
- **Gamma**: Delta sensitivity (how fast delta changes)
  - Penalty for high values (rapid swings)
- **Vega**: IV sensitivity (positive = profit from IV increase)
  - Penalty for large exposure (IV risk)

#### Composite Score (`score`)
- **Range**: 0.0 (worst) to 1.0 (best)
- **Typical Winners**: 0.70-0.85
- **Breakdown**: Shows contribution of each component
  - Positive: POP, ROC, Theta
  - Negative: Tail, Delta, Gamma, Vega penalties

### Trade Structure Types

#### Iron Condor (Most Common Winner)
```
                Call Spread
                 ↓     ↓
    -----[Buy]---[Sell]---[Sell]---[Buy]-----
         $480    $485    $515    $520
                  ↑     ↑
                Put Spread
```
- **Strategy**: Sell OTM call spread + sell OTM put spread
- **Bias**: Neutral (profit from low movement)
- **Max Profit**: Net credit received
- **Max Loss**: Spread width - credit
- **Best For**: Low volatility, range-bound stocks

#### Short Strangle
```
    -----[Sell]---------------[Sell]-----
         $475 Put            $525 Call
```
- **Strategy**: Sell OTM put + sell OTM call
- **Bias**: Neutral (profit from low movement)
- **Max Profit**: Total premium collected
- **Max Loss**: Theoretically unlimited (very large in practice)
- **Best For**: High IV, expect IV crush

#### Short Straddle
```
    -------------[Sell Both]-------------
                ATM $500
```
- **Strategy**: Sell ATM call + sell ATM put
- **Bias**: Strong neutral (profit from minimal movement)
- **Max Profit**: Total premium collected
- **Max Loss**: Theoretically unlimited
- **Best For**: Very low volatility, tight range

#### Vertical Spread (Call/Put)
```
Call Vertical:               Put Vertical:
    ---[Buy]---[Sell]---     ---[Sell]---[Buy]---
       $495    $505             $495     $485
```
- **Strategy**: Buy one strike, sell another (same expiry)
- **Bias**: Directional (bullish call spread, bearish put spread)
- **Max Profit**: Credit received (credit spread) or Spread width - debit (debit spread)
- **Max Loss**: Debit paid or Spread width - credit
- **Best For**: Directional plays with defined risk

---

## Advanced Usage

### Override Config from CLI

```bash
# Increase MC paths for higher accuracy
qse optimize-strategy --ticker SPY --regime neutral --trade-horizon 1 \
  --override "mc.num_paths=10000"

# Relax capital constraint
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1 \
  --override "filters.max_capital=20000"

# Prioritize POP over ROC
qse optimize-strategy --ticker TSLA --regime mild-bearish --trade-horizon 1 \
  --override "scoring.w_pop=0.50" \
  --override "scoring.w_roc=0.20"

# Multiple overrides at once
qse optimize-strategy --ticker AAPL --regime low-volatility --trade-horizon 1 \
  --override "mc.num_paths=15000" \
  --override "filters.max_capital=25000" \
  --override "filters.min_pop_breakeven=0.70" \
  --override "scoring.w_pop=0.45"
```

### Multi-Day Horizons

```bash
# 3-day income trade
qse optimize-strategy --ticker SPY --regime neutral --trade-horizon 3

# 5-day trend play
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 5
```

**Interpretation for H > 1**:
- Metrics computed over full horizon (E[PnL_3day], POP_3day)
- Regime parameters compound via multi-step MC (not single jump)
- Theta reported as daily average over horizon
- Option maturities at exit: Original DTE - H

### Batch Processing

```bash
# Create batch script
cat > batch_optimize.sh <<'EOF'
#!/bin/bash
for ticker in NVDA TSLA AAPL MSFT SPY; do
  echo "Optimizing $ticker..."
  qse optimize-strategy \
    --ticker $ticker \
    --regime strong-bullish \
    --trade-horizon 1 \
    --config my_config.yml \
    > results_${ticker}.json
done
EOF

chmod +x batch_optimize.sh
./batch_optimize.sh
```

### Custom Regime Definitions

Add your own regimes to `my_config.yml`:

```yaml
regimes:
  # Custom: Explosive breakout expected
  explosive-breakout:
    mean_daily_return: 0.05    # +5% expected move
    daily_vol: 0.06            # 6% volatility
    skew: 0.8                  # Strong right skew
    kurtosis_excess: 3.0       # Very fat tails

  # Custom: Post-earnings drift
  post-earnings-drift:
    mean_daily_return: 0.015   # +1.5% expected move
    daily_vol: 0.025           # 2.5% volatility
    skew: 0.4                  # Moderate right skew
    kurtosis_excess: 2.0
```

Then use:
```bash
qse optimize-strategy --ticker NVDA --regime explosive-breakout --trade-horizon 1
```

---

## Troubleshooting

### Common Issues

#### 1. Empty Results (No Candidates Pass Filters)

**Symptom**:
```json
{
  "top10": [],
  "diagnostics": {
    "rejections": {
      "capital_filter": 342,
      "epnl_filter": 28,
      "pop_filter": 5
    }
  }
}
```

**Cause**: Filters too strict for available option chain

**Solution A** - Relax capital constraint:
```bash
qse optimize-strategy --ticker AAPL --regime neutral --trade-horizon 1 \
  --override "filters.max_capital=20000"
```

**Solution B** - Lower POP requirement:
```bash
qse optimize-strategy --ticker AAPL --regime neutral --trade-horizon 1 \
  --override "filters.min_pop_breakeven=0.50"
```

**Solution C** - Reduce min E[PnL]:
```bash
qse optimize-strategy --ticker AAPL --regime neutral --trade-horizon 1 \
  --override "filters.min_expected_pnl=300"
```

#### 2. Schwab API Timeout

**Symptom**:
```
[ERROR] Schwab API timeout after 30 seconds
[WARN] Falling back to yfinance provider
[INFO] yfinance: Fetched option chain for NVDA (247 contracts)
```

**Cause**: Network issues or Schwab API rate limiting

**Action**: No action needed - fallback automatic

**Prevention**: Increase timeout in config:
```yaml
data_source:
  timeout: 60  # Increase to 60 seconds
```

#### 3. Adaptive Paths Not Converging

**Symptom**:
```
[WARN] Trade #3: E[PnL] CI remains wide ($400-$900) even at 20k paths
[WARN] High path sensitivity detected - estimates may be unstable
```

**Cause**: Trade has high path-dependent variance (e.g., gamma-heavy structure)

**Action**: Increase max paths or accept uncertainty:
```bash
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1 \
  --override "mc.max_paths=50000"
```

#### 4. Runtime Too Slow (>30 seconds)

**Cause**: Too many MC paths or too many survivors from Stage 3

**Solution A** - Reduce MC paths for testing:
```bash
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1 \
  --override "mc.num_paths=2000"
```

**Solution B** - Stricter Stage 3 filters:
```bash
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1 \
  --override "filters.top_k_per_type=10"  # Reduce from 20 to 10
```

**Solution C** - Tighter moneyness window:
```bash
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1 \
  --override "filters.moneyness_min=0.90" \
  --override "filters.moneyness_max=1.10"
```

#### 5. Unrealistic E[PnL] Estimates

**Symptom**: E[PnL] = $5,000 for 1-day trade (seems too high)

**Cause**: Regime parameters too extreme or incorrect structure type

**Check 1** - Verify regime parameters:
```yaml
regimes:
  strong-bullish:
    mean_daily_return: 0.02  # Should be 1-3% max for daily
    daily_vol: 0.03          # Should be 1-5% for typical stocks
```

**Check 2** - Verify structure makes sense for regime:
- Neutral regime → Iron Condors, Short Strangles
- Bullish regime → Call Spreads, Bull Put Spreads
- Bearish regime → Put Spreads, Bear Call Spreads

#### 6. Missing Strikes After Stage 1

**Symptom**:
```
[INFO] Stage 1: Retained 3 strikes (expected 8-12)
[WARN] Low strike count may limit candidate generation
```

**Cause**: Liquidity filters too strict or illiquid underlying

**Solution** - Relax liquidity requirements:
```bash
qse optimize-strategy --ticker XYZ --regime neutral --trade-horizon 1 \
  --override "filters.min_volume=50" \
  --override "filters.min_open_interest=50" \
  --override "filters.max_bid_ask_spread_pct=0.25"
```

---

## Performance Expectations

### Runtime Targets

| Mode | MC Paths | Typical Runtime | Use Case |
|------|----------|----------------|----------|
| Fast Test | 1,000 | 2-3 seconds | Quick iteration, testing configs |
| Standard | 5,000 | 4-6 seconds | Default production mode |
| High Accuracy | 10,000 | 8-12 seconds | Important trades, low tolerance for error |
| Maximum | 20,000 | 15-25 seconds | Critical trades, maximum confidence |

### Candidate Progression (Typical)

| Stage | Count | Description |
|-------|-------|-------------|
| Stage 0 | 4 expiries | Selected from 6-10 available |
| Stage 1 | 10 strikes/expiry | Filtered from 15-20 available |
| Stage 2 | 300-500 structures | Generated from combinations |
| Stage 3 | 100-200 survivors | After hard filters + top-K |
| Stage 4 | 100-200 scored | Full MC on all survivors |
| Top-100 | 100 cached | Stored for analysis |
| Top-10 | 10 displayed | Final ranked output |

---

## Future Enhancements (Coming Soon)

### Visualization & UI
- **Profit Probability Plots**: Visual P&L distributions for each strategy
- **Grafana Dashboards**: Real-time monitoring of live positions
- **Streamlit UI**: Interactive strategy explorer with parameter tuning
- **mplfinance Charts**: Option payoff diagrams with underlying price paths

### Live Position Monitoring
```bash
# Monitor a selected trade in real-time
qse monitor \
  --position top10_rank1.json \
  --interval 5min \
  --alert-profit-target 500 \
  --alert-stop-loss -150
```

### Enhanced Distribution Models
- **Regime-Conditioned Bootstrap**: Sample from historical bars matching current regime
- **ML Conditional Models**: Gradient-boosted quantile regression for return distributions
- **GARCH-t**: Full volatility clustering with Student-t innovations

### Advanced Pricing
- **Bjerksund-Stensland**: American exercise approximation (default upgrade)
- **Heston Stochastic Vol**: Path-dependent IV dynamics
- **IV Surface Interpolation**: Bilinear interpolation from grid data

### Additional Structure Types
- Butterflies, Calendars, Diagonals
- Ratio Spreads, Back Spreads
- Custom user-defined structures

---

## Quick Reference Card

### Essential Commands
```bash
# Basic optimization
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1

# With overrides
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1 \
  --override "mc.num_paths=10000" \
  --override "filters.max_capital=20000"

# Multi-day horizon
qse optimize-strategy --ticker SPY --regime neutral --trade-horizon 3

# Custom config
qse optimize-strategy --ticker TSLA --regime mild-bullish --trade-horizon 1 \
  --config custom_config.yml
```

### Key Config Sections
```yaml
regimes:          # Market outlook definitions
mc:               # Monte Carlo settings
filters:          # Hard constraints (Stage 0-3)
scoring:          # Composite score weights
costs:            # Transaction costs
data_source:      # API configuration
distribution:     # Return distribution model
pricing:          # Option pricing model
```

### Regime Cheat Sheet
| Regime | Mean Return | Volatility | Use When |
|--------|-------------|------------|----------|
| neutral | 0% | 1% | Sideways, range-bound |
| strong-neutral | 0% | 0.5% | Very tight range |
| low-volatility | 0% | 0.8% | Calm, predictable |
| volatility-dir-uncertain | 0% | 3% | High vol, no direction |
| mild-bearish | -1% | 2% | Slight downtrend |
| strong-bearish | -3% | 4% | Significant decline |
| mild-bullish | +1% | 2% | Slight uptrend |
| strong-bullish | +2% | 3% | Strong rally |

---

## Support & Feedback

- **Documentation**: `/docs/user-guide/`
- **Examples**: `/examples/optimizer_runs/`
- **Issue Tracker**: GitHub Issues
- **Community**: Discord/Slack

---

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**Status**: MVP Release
