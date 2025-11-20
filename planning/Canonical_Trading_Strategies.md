Below are **canonical, widely-used stock and option strategies** that are simple, rule-driven, well-understood in the literature, and suitable for an MVP qse/Monte-Carlo system.

They require NO proprietary datasets and translate cleanly into your architecture.

I’m giving you **4 stock strategies** and **4 options strategies**, each with:

* Entry
* Exit
* Position size
* Required indicators
* Why it is canonical
* How to encode it in your system (signals + parameters)

These are “real” strategies, not academic toys.

---

# **I. Canonical Stock Strategies**

---

# **1. 20/50 SMA Trend-Following (classic crossover)**

**Category:** Momentum / trend following
**Works well for:** Liquid stocks and ETFs (SPY, QQQ, AAPL)

### **Entry**

* Go **long** when SMA20 crosses **above** SMA50 (bullish momentum).

### **Exit**

* Close long when SMA20 crosses **below** SMA50.

### **Position Sizing**

* Fixed dollar weight (e.g., $10,000 per trade)
* Or fixed % of equity (e.g., 10%)

### **Stops / Targets (optional)**

* Stop loss: 5–8%
* No explicit take-profit (momentum strategies ride trends)

### **Why this strategy is canonical**

* Used in CTAs, retail, and quant funds for 40+ years
* Robust across regimes
* Mechanically straightforward

### **Implementation Notes**

Requires only SMA20 and SMA50, both available in `pandas_ta`.

---

# **2. RSI Mean Reversion (Wilder’s RSI 30/70)**

**Category:** Short-term mean reversion
**Works well for:** High-volume equities with stable volatility

### **Entry**

* Go **long** when RSI(14) < 30 (oversold).

### **Exit**

* Close long when RSI(14) > 50
  (avoid waiting for 70; better risk-adjusted behavior)

### **Position Sizing**

* Fixed number of shares
* Or volatility-weighted position (ATR-based)

### **Stops**

* Stop loss: 3–5%
* No take profit needed (RSI exit handles it)

### **Why this strategy is canonical**

* One of the simplest and best-documented tactical reversion systems
* Still used as a baseline in low-latency and high-level quant shops

---

# **3. Bollinger Band Reversion (2σ)**

**Category:** Volatility-based mean reversion
**Works well for:** Stocks with noisy, sideways action

### **Entry**

* Go **long** when close price < lower Bollinger band (20, 2σ).

### **Exit**

* Sell when close > mid-band (20 SMA).
* Optional: full exit at upper band.

### **Position Sizing**

* 1 × ATR-tranche
* Or fixed fractional sizing

### **Stops**

* Hard stop 2× ATR below entry
* Take profit at upper band optional

### **Why canonical**

* John Bollinger’s original formulation
* Easy to parameterize for grid search

---

# **4. Breakout Strategy (Donchian 20-day breakout)**

**Category:** Trend-following breakout
**Works well for:** Stocks with structural momentum trends

### **Entry**

* Go long when price closes **above the highest high of the last 20 days**.

### **Exit**

* Exit when price closes **below the lowest low of the last 10 days**.

### **Position Sizing**

* ATR-based position sizing recommended
  (classic Turtle Trading rule)

### **Why canonical**

* Foundation of Turtle Trading rules
* Still effective on many trending assets

---

# **II. Canonical Options Strategies**

All are **long-only** strategies appropriate for retail + quant evaluation.
None require advanced market-making greeks infrastructure.

---

# **5. ATM Long Call Momentum (paired with SMA trend)**

**Category:** Directional momentum via calls
**Works well when:** Stock is trending and IV is not inflated

### **Entry**

* When SMA20 > SMA50 **and** price > SMA20
* Buy **ATM call**
* DTE = 30–45 days

### **Exit**

* Profit target: +50%
* Stop loss: -30% premium
* Time exit: 7 DTE (to avoid rapid theta decay)

### **Strike Selection**

* Nearest ATM strike (within ±1%)

### **IV Filter**

* Require IV percentile (IV Rank) < 60
  (avoid buying calls when IV is too expensive)

### **Why canonical**

* Simple, understandable, widely taught momentum call strategy
* Good baseline versus stock momentum

---

# **6. Long Put + RSI Oversold (protective/contrarian)**

**Category:** Tactical downside protection / mean reversion
**Works well when:** Stock dips sharply but may have follow-through

### **Entry**

* When RSI(14) < 25
* Buy **ATM put**
* DTE = 14–21 days

### **Exit**

* Profit: +40–60%
* Time: exit at 5 DTE
* Stop-loss: -25%

### **Strike**

* ATM (±1%)

### **Why canonical**

* This is the classic “catch falling knives safely” pattern with options,
  avoids equity exposure.

---

# **7. Long Straddle on News Spike (IV expansion strategy)**

**Category:** Volatility trading
**Works well when:** Expected move ≠ actual move (IV crush/expansion)

### **Entry**

* Only when:

  * price gap > 3%
  * volume spike > 2× avg
  * earnings/news event imminent
* Buy **ATM straddle** (call + put)
* DTE = nearest weekly (7–10)

### **Exit**

* Exit both legs when:

  * combined premium rises 30–50%
  * OR IV collapses and premium loses 20%

### **Why canonical**

* Classic earnings play
* Teaches IV crush effects in your simulator

---

# **8. Long Calendar Spread (low IV entry)**

**Category:** Theta/volatility structure trade
**Works well when:** IV is low but expected to rise

### **Entry**

* When IV Rank < 30
* Buy 60 DTE ATM call
* Sell 30 DTE ATM call

### **Exit**

* Exit entire spread when:

  * profit > 25%
  * OR time-based exit at 10 days before short leg expiry

### **Why canonical**

* Widely used “income + vol structure” strategy
* Clean greeks exposure (theta positive, vega positive)
