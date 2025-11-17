# Metrics Interpretation Guide

## Understanding Compare Output

### Example Output
```
MetricsReport(
    mean_pnl=-4.27,
    median_pnl=-151.52,
    max_drawdown=-0.7287,
    sharpe=-0.0015,
    sortino=-0.0023,
    var=-0.6224,
    cvar=-0.7924,
    var_method='historical',
    lookback_window=None,
    covariance_estimator='sample',
    bankruptcy_rate=0.0,
    early_exercise_events=0
)
```

---

## Key Metrics Explained

### **mean_pnl** (Mean Profit & Loss)
**Value**: -4.27
**Interpretation**: Average P&L across all Monte Carlo paths is **-$4.27**

- **Positive**: Strategy made money on average
- **Negative**: Strategy lost money on average
- **Magnitude**: Indicates expected profit/loss per simulation run

**Your Result**: Small negative value suggests strategy is roughly break-even with slight losses.

---

### **median_pnl** (Median Profit & Loss)
**Value**: -151.52
**Interpretation**: The middle value when all path P&Ls are sorted is **-$151.52**

**Mean vs Median Analysis**:
- Mean (-4.27) > Median (-151.52) → **Right-skewed distribution**
- Most paths lose money, but a few big winners pull the average up
- **50% of paths lost more than $151.52**

**Implication**: Strategy has:
- Frequent small/medium losses
- Occasional large wins
- Not consistent (high variance)

---

### **max_drawdown** (Maximum Drawdown)
**Value**: -0.7287 = **-72.87%**
**Interpretation**: Largest peak-to-trough decline across all paths

**Formula**: `max_drawdown = min((portfolio_value - peak_value) / peak_value)`

**Your Result**: In the worst case, portfolio value dropped **72.87%** from its peak.

**Risk Assessment**:
- **< -20%**: Moderate risk
- **-20% to -40%**: High risk
- **> -40%**: Very high risk ⚠️
- **> -70%**: Extreme risk 🚨 ← **You are here**

**Implication**: This strategy is **extremely risky** with potential for severe losses.

---

### **sharpe** (Sharpe Ratio)
**Value**: -0.0015
**Interpretation**: Risk-adjusted return is **-0.0015**

**Formula**: `sharpe = (mean_return - risk_free_rate) / std_dev_returns`

**Typical Ranges**:
- **> 1.0**: Excellent (strong returns for risk taken)
- **0.5 - 1.0**: Good
- **0 - 0.5**: Acceptable
- **< 0**: Poor (losses or returns don't justify risk) ← **You are here**

**Your Result**: Negative Sharpe means:
- Losing money on average
- Poor risk-adjusted performance
- Risk-free investment (e.g., T-bills) would be better

---

### **sortino** (Sortino Ratio)
**Value**: -0.0023
**Interpretation**: Downside risk-adjusted return is **-0.0023**

**Difference from Sharpe**:
- Sharpe penalizes all volatility (up and down)
- Sortino only penalizes downside volatility (more realistic)

**Your Result**:
- More negative than Sharpe (-0.0023 vs -0.0015)
- Downside risk is even worse than overall risk
- Strategy has asymmetric risk (downside worse than upside)

---

### **var** (Value at Risk @ 95%)
**Value**: -0.6224 = **-62.24%**
**Interpretation**: With 95% confidence, losses won't exceed **62.24%**

**Or equivalently**: There's a **5% chance** of losing more than 62.24%

**Risk Categories**:
- **< -10%**: Low risk
- **-10% to -30%**: Moderate risk
- **-30% to -50%**: High risk
- **> -50%**: Extreme risk 🚨 ← **You are here**

**Your Result**: **5 out of 100 runs** will lose more than 62.24% of capital.

---

### **cvar** (Conditional Value at Risk)
**Value**: -0.7924 = **-79.24%**
**Interpretation**: If losses exceed VaR threshold, **expected loss is 79.24%**

**Also called**: Expected Shortfall (ES), Tail VaR

**What it means**:
- VaR tells you the threshold (-62.24%)
- CVaR tells you the **average loss beyond that threshold** (-79.24%)

**Your Result**: In the worst 5% of scenarios, you lose **79.24% on average**.

**Risk Assessment**: This is **catastrophic risk** territory.

---

### **bankruptcy_rate**
**Value**: 0.0 = **0%**
**Interpretation**: No paths went to zero or negative capital

**Positive Sign**: Despite high drawdowns, strategy avoided complete ruin.

---

### **early_exercise_events**
**Value**: 0
**Interpretation**: No American options were exercised early

**Note**: Relevant only for American-style options with early exercise features.

---

## Overall Assessment: Your TSLA Strategy

### ✅ What Went Well
- **No bankruptcies**: Strategy preserved capital (didn't go to zero)
- **Mean ≈ 0**: On average, roughly break-even (not terrible)

### ⚠️ Major Concerns

1. **Extreme Drawdown Risk** (-72.87%)
   - Portfolio can lose 73% of value in worst paths
   - This is **unacceptable risk** for most investors

2. **High Tail Risk** (VaR -62%, CVaR -79%)
   - 5% chance of losing 62%+
   - When it's bad, it's **really bad** (avg -79%)

3. **Inconsistent Performance** (Median << Mean)
   - Most paths lose money (median -$151)
   - A few big winners skew average upward
   - Not reliable for consistent returns

4. **Negative Risk-Adjusted Returns** (Sharpe/Sortino < 0)
   - Returns don't justify the risk taken
   - Better to hold cash or risk-free assets

### 🔍 Why These Results?

**Possible Reasons**:
1. **Position Sizing Too Aggressive**
   - Targeting $500-$1,000 daily P&L might be too large
   - Drawdowns suggest over-leveraged positions

2. **Strategy Not Calibrated for TSLA Volatility**
   - TSLA is extremely volatile
   - SMA trend + ATM call momentum may whipsaw in choppy markets

3. **No Stop Losses / Risk Management**
   - Current strategies lack protective stops
   - Losses can compound without circuit breakers

4. **Market Conditions in Simulation**
   - Laplace distribution might not match TSLA's actual behavior
   - Synthetic data may have adverse conditions

### 📊 Recommended Next Steps

1. **Reduce Position Sizing**
   ```bash
   # Lower target profit to reduce risk
   --stock-params '{"target_profit_usd": 250, "max_position_usd": 20000}'
   ```

2. **Add Stop Losses**
   - Implement max drawdown stops in strategies
   - Exit positions when losses exceed threshold

3. **Test on Different Symbols**
   - TSLA is uniquely volatile
   - Try SPY (S&P 500 ETF) for more stable testing

4. **Compare Strategies**
   ```bash
   # Test RSI reversion vs trend following
   --strategy stock_rsi_reversion
   ```

5. **Analyze Signals** (see next section)
   - Look at entry/exit patterns
   - Check if strategy is over-trading

---

## Quick Reference Card

| Metric | Your Value | Status | Meaning |
|--------|------------|--------|---------|
| **Mean P&L** | -$4.27 | ⚠️ Slightly negative | Near break-even |
| **Median P&L** | -$151.52 | 🚨 Negative | Most paths lose |
| **Max Drawdown** | -72.87% | 🚨 Extreme | Catastrophic risk |
| **Sharpe** | -0.0015 | 🚨 Negative | Poor risk/reward |
| **Sortino** | -0.0023 | 🚨 Negative | Worse downside |
| **VaR (95%)** | -62.24% | 🚨 Extreme | 5% chance >62% loss |
| **CVaR** | -79.24% | 🚨 Catastrophic | Tail losses ~79% |
| **Bankruptcy** | 0.0% | ✅ Good | No total losses |

**Overall Grade**: 🚨 **D- (High Risk, Poor Returns)**

**Recommendation**: **Do not trade this strategy with real money** until risk is reduced significantly.
