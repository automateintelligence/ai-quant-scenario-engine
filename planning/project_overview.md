# Project Overview

# Project Story
I am focused on single stock and option trading algo optimization. 

## CONSTITUTION: Modular designs that I can run on my virtual private server (VPS) with CPU only, no GPU right now.  Maximize parallelization and vector maths.  I am open to rules-based strategies; real-time LLM-enabled market, media, and social media analysis; and machine learning based models. Containerized services, like VectorBT, are acceptable options if they are worth the added complexity. I currently have paid API access to multiple LLM foundation models: OpenAI, You.com, and Anthropic. If we are successful, I will probably progress to GPU-based models and API services like Amazon Bedrock or Google Cloud.  For data and trading services, I already have access to the Schwab API. I can sign up for the Merrill Lynch API if that provides capabilities that Schwab does not.   I have stock and option trading level III (up to spreads, but not short uncovered) accounts at Merrill and Schwab.

- I envision a backtesting module, a strategy module, and a strategy optimization module.  I am very familiar with a variety of optimization methods that I would like to employ to efficiently optimize strategy parameters and test them with massive parallelization.  I am comfortable writing all my own modules, but if I can leverage existing open-source libraries, I prefer to do that first, but it must be in a way that makes sense and is relatively easy to use as I just described.
- Now that you have done all this research, please recommend a system that I can build to an MVP quickly to start evaluating single stock and option strategies.  My first goal is to compare risk and reward surfaces for stock trades versus option trades on the same underlying stock.  User-initiated CLI or GUI trade execution is a nice-to-have, but not a must for MVP.  I can trade on the Thinkorswim platform or on the brokerage websites if I must.  Then I can gradually increase complexity and features as we move forward.
Great, this is a nice, clean set of goals. I’ll aim for something you can actually start coding this week on a CPU-only VPS, but that won’t box you in later for portfolio-level work or GPUs.

1) I can use the Schwab API for data residual too, but I need to test it out to see of it is better than yfinance or PyData.  
2) I want to use existing library function calls for calculating distribution fits and a large suite of financial and momentum indicators.  
3) When there are other financial indicators, like Fed Rate, that I wanted to include, I assume that I can use event driven library or simply interpolate it into same time series time steps. 
4) There are more accurate option pricing models than Black-Scholes.  I would like to use one that is appropriate for this application. 

## User Stories
4) Stock Screening - As a user, quickly evaluate if a large number (100s) of stocks are a good candidate for trading strategy (specified by strategy module) before trading begins the next day.  
5) 

### User Story X - Parameter grid exploration (Priority: P2)

Author defines a parameter grid (e.g., thresholds, DTE, strike offset) and runs batch evaluations in parallel, receiving aggregate metrics (mean, CVaR, drawdown) per configuration.

**Why this priority**: Enables strategy optimization and surface mapping beyond a single run.

**Independent Test**: Supply a small grid (≥3 configs) and confirm per-config metrics are produced and ranked by an objective function.

**Acceptance Scenarios**:

1. **Given** a grid of strategy parameters, **When** the batch job executes, **Then** results include metrics per config and an objective score suitable for ranking.  
2. **Given** N paths × M configs, **When** parallel execution runs, **Then** the job completes within configured resource limits and without race conditions or corrupted outputs.

---

## 1. Recommended stack & philosophy

Given your constraints and preferences:

* **Language / environment**

  * Python 3.11+ on your VPS
  * Core libs: `numpy`, `pandas`, `scipy`, `numba`, `statsmodels` or `arch`, `quantstats`, `matplotlib`/`plotly`
* **Backtesting core**

  * Custom **vectorized Monte Carlo engine** (your own) rather than a full event engine like Zipline/Lean for MVP.
  * The event engines shine with historical order book behavior, corporate actions, etc., but for **risk/reward surface comparison via synthetic paths**, a custom vectorized core will be faster, simpler, and more transparent.
* **Analytics**

  * **QuantStats** (modern PyFolio-like tear sheets) for equity curves and summary stats.
* **Optional later**

  * **vectorbt** when you start doing huge parameter sweeps or multi-asset portfolios.
  * A more formal event-driven engine (QSTrader / Zipline Reloaded / Lean) when you want to replay real fill behavior and portfolio interactions.

MVP goal: a **pure-Python, CPU-only**, highly vectorized core that does:

> “Given an underlying and a set of strategy parameters, generate N Monte Carlo price paths and compute the P&L distribution for:
> (a) stock trades, and (b) option trades on that same stock.”

---

## 2. System architecture (MVP, product 3)

Think in three layers, aligned with what you described:

1. **Backtesting / Simulation Module**
2. **Strategy Module**
3. **Strategy Optimization Module**

We’ll make them cleanly separated and functional.

### 2.1 Backtesting / Simulation Module

**Responsibility:** Generate price paths, apply execution logic, compute P&L, and hand off results for reporting.

**Core primitives**

1. **Data loader** (historical base)

   * Inputs: symbol, date range, bar interval (daily for MVP).
   * Output: `pd.DataFrame` with `['open', 'high', 'low', 'close', 'volume']`.
   * For MVP, use something simple (e.g. `yfinance` or CSVs you export from Schwab). You don’t need Schwab’s API in loop yet.

2. **Return & distribution fitting**

   * Compute log returns: `r_t = log(S_t / S_{t-1})`.
   * Fit candidate distributions:

     * Gaussian (Normal)
     * Laplace
     * Student’s t (for fat tails; this will be your likely default)
   * Use SciPy / `arch`:

     * `scipy.stats` for Normal/Laplace.
     * `arch` for T-distributed residuals or GARCH+T if you want volatility clustering.
   * Pick the distribution by AIC/BIC or simple KS test. Keep it pluggable:

     ```python
     class ReturnDistribution:
         def fit(self, returns: np.ndarray) -> None: ...
         def sample(self, n_paths: int, n_steps: int) -> np.ndarray: ...
     ```

3. **Monte Carlo path generator**

   * Once you’ve fit a distribution `D`, you generate:

     * `R` = `(n_paths, n_steps)` matrix of simulated returns, drawn from `D`.
     * Convert to prices:

       ```python
       S0 = last_observed_price
       log_paths = np.cumsum(R, axis=1) + np.log(S0)
       prices = np.exp(log_paths)   # shape: (n_paths, n_steps)
       ```
   * *This is where you exploit NumPy heavily and optionally `numba` JIT to accelerate.*

4. **Execution engine (market simulator)**

Define a minimal interface:

```python
class MarketSimulator:
    def simulate_stock(self, prices, signals, position_size, fees) -> np.ndarray: ...
    def simulate_option(self, prices, option_spec, signals, fees) -> np.ndarray: ...
```

* `prices`: `(n_paths, n_steps)`
* `signals`: `(n_paths, n_steps)` position function (e.g. +1 long, 0 flat, −1 short) produced by the **strategy module**.
* **Stock P&L** (simple version, no margin, no borrowing costs):

  * P&L per path = sum over t of `position_t * (S_{t+1} − S_t)` minus fees.
* **Option P&L (MVP)**:

  * Start simple:

    * Only European calls/puts with fixed expiry `T_exp` and strike `K`.
    * You can measure:

      * Mark to market using Black-Scholes with fixed IV, or
      * Just payoff at expiry (if you care about end-state risk/reward, not path).
  * For each path:

    * Underlying path `S_t`.
    * Option payoff at expiry `max(±(S_T − K), 0)` scaled by contracts.
  * If you want mark-to-market risk surface, approximate daily value via Black-Scholes:

    * Input: `S_t`, `K`, `T_remaining`, `sigma` (choose from implied or from your distribution model), `r`.

The backtester then returns for each **strategy configuration**:

* Equity curve per path.
* Summary P&L distribution: mean, std, skew, kurtosis, VaR, CVaR, max drawdown, etc.

---

### 2.2 Strategy Module

**Responsibility:** Given a price path (or historical series), produce timestamped **signals** (and optional option specifications) independent of how the market is simulated.

Define a strategy interface that is agnostic to stock vs option:

```python
class Strategy:
    def generate_signals(self, prices: np.ndarray, params: dict) -> dict:
        """
        prices: (n_paths, n_steps) – simulated or historical
        returns: {
            "stock": signals_stock (n_paths, n_steps),
            "options": {
                "signals": signals_option (n_paths, n_steps),
                "spec": option_spec  # e.g. strike, expiry, type
            }
        }
        """
```

For the **first MVP**, you can pick something very simple and parametric, e.g.:

* For stock:

  * Rule: “Go long if price crosses below X-day moving average after a +Y% move yesterday, exit after Z days or when P% profit.”
* For options:

  * Rule: “Instead of buying the stock, buy an ATM call with expiry N days out when the same signal triggers; sell at the same exit condition or on expiry.”

Behind the scenes:

1. Take **historical** series.
2. Fit distribution.
3. Generate `N` future paths.
4. Run both “stock version” and “option version” of the strategy on the same set of paths.

This is perfect for your risk/reward surface comparison.

---

### 2.3 Strategy Optimization Module

**Responsibility:** Explore parameter spaces efficiently, using parallelization and vectorization, to estimate optimal parameter sets under your Monte Carlo engine.

For MVP:

* Accepts:

  * Strategy class
  * Parameter grid or domain:

    * e.g.:

      * moving average length: 5–20
      * take-profit: 3–15 %
      * stop-loss: 2–10 %
      * option strike offset: −5 % to +5 %
      * option days-to-expiry: 7–45
* For each parameter set:

  1. Run backtester (stock vs option).
  2. Collect summary metrics from QuantStats:

     * CAGR
     * Sharpe, Sortino
     * Max drawdown
     * CVaR (e.g. 5 %)
  3. Compute a scalar objective:

     ```text
     objective = w1 * CAGR - w2 * CVaR - w3 * MaxDrawdown
     ```
* Use:

  * For grid search: `itertools.product` + `joblib.Parallel` or Python’s `multiprocessing`.
  * For smarter search: `scipy.optimize`, `nevergrad`, `optuna`, or your own methods (CMA-ES, etc.).

Key design: **pure functions.**
`simulate(params) -> metrics` should be stateless and easy to parallelize.

---

### 2.4 CandidateSelection module

Decides which symbols/dates are “interesting” today. Uses features that capture transient/rare conditions. 
```python
class CandidateSelector(ABC):
    def score(self, features: pd.DataFrame) -> pd.Series:
        """Return a score or probability per timestamp."""
    def select(self, features: pd.DataFrame, threshold: float) -> pd.Series:
        """Return boolean Series: True when this timestamp is a candidate."""
```

You’re asking exactly the right question: “Is my strategy good *in general*?” is very different from “Is my strategy good *conditional on the weird thing that made this stock a candidate today*?”

You need to design for that explicitly. Think in terms of **two different models** and **conditional backtests**:

* A **Candidate Selector**: “Should I even look at this symbol right now?”
* An **Execution Strategy**: “Given I’m in this condition, how should I trade it?”

And then test the **execution strategy only on times when the selector would have fired**.

Below is a design that fits into your current architecture.

---

## 2.4.1. Separate the two problems in the architecture

Extend your system to have two distinct modules:

1. **CandidateSelection module**

   * Decides which symbols/dates are “interesting” today.
   * Uses features that capture transient/rare conditions.
2. **ExecutionStrategy module** (what we already designed)

   * Given “we are in a candidate state,” decides how to trade and manages risk.

In code terms, roughly:

```python
class CandidateSelector(ABC):
    def score(self, features: pd.DataFrame) -> pd.Series:
        """Return a score or probability per timestamp."""
    def select(self, features: pd.DataFrame, threshold: float) -> pd.Series:
        """Return boolean Series: True when this timestamp is a candidate."""
```

Then your strategy/backtest operates **only on timestamps where `select` is True**.

---

## 2.4.2. Define “candidate state” explicitly

You want to capture “uncommon transient conditions” that cause a stock to appear on your top-mover lists. For example:

* Large overnight or intraday gap (e.g. |return| > 3–5 σ of recent volatility)
* Volume spike (volume > X × rolling average)
* Volatility spike (realized or implied)
* News / sentiment / event flags (earnings, guidance, downgrades)
* Options activity (IV rank, skew, unusual volume)

Turn this into a **state vector** at time t:

[
\phi_t = [\text{gap_pct}_t, \text{vol_spike}_t, \text{rsi}_t, \text{iv_rank}_t, \dots]
]

And a **candidate rule**:

* Hard rule (deterministic):

  * “Candidate if: |1-day return| ≥ 5 percent and volume ≥ 3× 20-day average and price ≥ $X.”
* Or model-based:

  * Train a classifier (tree/GBM) to output “likely mean-reverting opportunity,” based on historical episodes and their outcomes.

Either way, the important part: **candidate selection is a function of state at t only** (no future leakage).

---

## 2.4.3. Build a library of historical “candidate episodes”

Once you define `select()`, run it over your historical feature set:

* For each symbol and date t where `is_candidate[t] == True`, define an **episode**:

  * Episode ID: (symbol, t0)
  * Episode window: `[t0, t0 + H]` where H = holding horizon (e.g. 5 trading days)
* Store:

  * Starting state features at t0 (`φ_t0`)
  * Subsequent price path S(t0 → t0+H)
  * Subsequent indicators if needed

You now have a **library of “what actually happened after we would’ve picked this as a candidate”**.

This lets you run **conditional backtests**:

> “How does my strategy perform *given* that this stock just looked like this (big move, spike, etc.)?”

---

## 2.4.4. Conditional backtesting: evaluate strategy only on candidate episodes

Instead of traditional backtesting on every bar:

1. For each candidate episode `(symbol, t0)`:

   * Feed the strategy the state at t0 and onward.
   * Simulate execution from t0 to t0+H for **stock** and **option** variants.
2. Collect P&L per episode:

   * Episode-level P&L, max drawdown, time-in-trade, etc.
3. Aggregate only over these episodes:

   * Conditional expected return, CVaR, hit rate, etc.

You might keep two sets of metrics:

* **Unconditional performance**: strategy on all days (sanity check).
* **Conditional performance**: strategy only when candidate conditions are met (what you actually care about operationally).

This directly answers:

> “If I trade this strategy only when the stock looks like it does today, what does history say about the distribution of outcomes?”

---

## 2.4.5. Conditional Monte Carlo instead of naive Monte Carlo

For the Monte Carlo part, you want to simulate paths that are **consistent with being in a rare, transient state**, not generic average days.

Two ways to do it:

### 2.4.5.1 Episode-based resampling (“analogue” Monte Carlo)

* From your candidate library, collect all episodes where the state is “similar” to today’s:

  * Use k-nearest neighbors in feature space (φ distance)
  * Or simple filters: “all big +8–12 percent gap ups with 3× volume”
* For each such episode:

  * Extract the forward returns path (e.g. 5–10 days)
* Build Monte Carlo by **resampling these forward paths with replacement**:

  * That gives you a bootstrap distribution of “what typically happens after states like this.”

Pros:

* Very realistic; no parametric model.
* Naturally respects asymmetries (post-earnings drift, post-news behavior).

### 2.4.5.2 State-conditioned distribution modeling

If you still want parametric / semi-parametric models:

* Fit **separate return distributions** or GARCH models on:

  * Data conditional on “candidate states,” or
  * Immediately after large moves (e.g. days t+1 to t+5 following big gaps)
* Or build a **regime classifier** (e.g. HMM, clustering) and:

  * Condition Monte Carlo parameters on the inferred regime (spike regime, crash regime, calm regime).
* Then Monte Carlo draws are from **P(returns | candidate_state)** rather than P(returns).

Either way, your Monte Carlo engine becomes:

```python
simulate_conditional_paths(
    current_state: FeatureVector,
    horizon: int,
    n_paths: int
) -> np.ndarray  # (n_paths, horizon)
```

And your backtester runs strategies over those conditional paths for stock vs options.

---

## 2.4.6. How this fits into your existing architecture

Add a new module and integrate with the others:

### 2.4.6.1 Candidate selection module

```python
class CandidateSelector(ABC):
    def score(self, features: pd.DataFrame) -> pd.Series: ...
    def select(self, features: pd.DataFrame, threshold: float) -> pd.Series: ...
```

Concrete implementation example:

* `TopMoverSelector`:

  * Uses daily % returns, volume spike, volatility, etc.
* `NewsSpikeSelector`:

  * Uses external_sentiment_score, gap, etc.

### 2.4.6.2 Episode builder

```python
def build_candidate_episodes(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    selector: CandidateSelector,
    horizon: int
) -> list[Episode]:
    """
    Episode: dataclass with (symbol, t0, window_prices, window_features, state_vector)
    """
```

This lives at the boundary between **data layer** and **strategy/backtest**.

### 2.4.6.3 Conditional backtest runner

```python
def run_conditional_backtest(
    episodes: list[Episode],
    strategy: Strategy,
    strategy_params: dict,
    simulator: MarketSimulator
) -> pd.DataFrame:
    """
    Returns per-episode metrics: [episode_id, pnl, max_dd, ...]
    """
```

### 2.4.6.4 Conditional Monte Carlo module

Either:

* `EpisodeResampler` that builds synthetic scenarios from episodes, or
* `StateConditionedDistribution` that adds `fit_for_state(state_vector)`.

This plugs into your existing `ReturnDistribution` / `generate_price_paths` machinery, but your “fit” is done on **episode subsets**, not the entire series.

---

## 3. How to compare stock vs options risk/reward surfaces (MVP workflow)

Here’s a concrete end-to-end workflow that you can implement now.

### Step 1: Pick a symbol and baseline window

* Example: AAPL, 3 years of daily data.
* Load OHLC, compute log returns.

### Step 2: Fit risk model

* Fit a Student’s T distribution (or GARCH+T if you want volatility clustering) to returns.
* Freeze the fitted parameters in a `ReturnDistribution` object.

### Step 3: Generate Monte Carlo paths

* Choose:

  * `n_paths` = 1,000–10,000 (CPU dependent).
  * `n_steps` = 20 – representing 20 trading days into the future (for short-term swing) or 60 for a few months.
* Use your distribution to generate `(n_paths, n_steps)` price paths.

### Step 4: Define paired strategies

For instance:

* **Stock strategy S_stock**:

  * Entry: if price drops at least 5 % in 1 day, buy 100 shares.
  * Exit: target +7 % from entry or stop −4 % or max holding 10 days.

* **Option strategy S_opt**:

  * Same entry trigger.
  * Instead of 100 shares:

    * Buy N ATM calls (strike = current price, expiry ~30 days).
  * Exit: same conditions (profit/stop/time) applied to option mark-to-market value.

The signals are analogous; only the instrument payoff/greeks differ.

### Step 5: Run simulations

For each Monte Carlo path:

1. Run `S_stock` → equity curve, final P&L.
2. Run `S_opt` → equity curve, final P&L.

Collect:

* Distribution of P&L for stock and for options.
* Key stats:

  * Mean P&L, median, percentile distribution (5th, 25th, 75th, 95th).
  * Max loss, worst X % outcomes.
  * Sharpe / Sortino if you treat each path as a pseudo-return sequence.
  * CVaR at 5 % or 1 %.
* Visuals:

  * Histogram of outcomes: stock vs options.
  * Cumulative distribution functions (CDFs).
  * Scatter: expected return vs CVaR for many parameter sets.

QuantStats can help generate tear sheets for each side, but for comparative visuals you’ll likely do a **custom set of plots** (matplotlib/plotly).

### Step 6: Build “risk/reward surface”

Now you can treat parameters as axes and compute metrics:

* Axes:

  * Option days-to-expiry
  * Strike moneyness (K/S)
  * Position size
* For each grid point, run the Monte Carlo backtest and compute:

  * Expected P&L
  * CVaR
  * Probability of loss
* Plot surfaces or contour maps:

  * E.g. 3D: `E[P&L]` vs `K/S` vs `DTE` for options vs constant stock strategy.

This directly answers: *does trading options on this underlier give me a better risk-adjusted profile than the comparable stock strategy?*

---

## 4. Where existing libraries fit

You **don’t** need to bolt on a full backtesting framework to build the MVP, but some libraries are high leverage:

* **QuantStats**

  * Use for summarizing each strategy’s returns/equity curve:

    ```python
    import quantstats as qs
    qs.reports.html(returns_series, benchmark="SPY", output="report.html")
    ```
  * Great for sanity checking your Monte Carlo results and giving you “PyFolio-style” tear sheets.

* **Backtesting.py (optional)**

  * If you want to sanity check your strategy logic on **historical data only** (no Monte Carlo) with minimal code, Backtesting.py is useful.
  * But for the **simulated paths**, your own engine is simpler.

* **vectorbt (later)**

  * Once you start exploring thousands of parameter combinations across many assets (for products 1 & 2), vectorbt will shine.
  * For now, it’s optional complexity.

* **arch / statsmodels**

  * For fitting heavy-tailed and/or time-varying volatility models.

---

## 5. Path to your other two products

The design above generalizes nicely:

1. **Portfolio asset allocation optimizer (efficient frontier + entropy)**

   * Replace “single price path” with a **matrix of asset returns** (real or simulated).
   * Use the same Monte Carlo machinery to simulate portfolio returns given weights.
   * Build an optimizer:

     * Efficient frontier: mean–variance optimization (`cvxpy`, `PyPortfolioOpt`).
     * Entropy maximization: add Shannon entropy of weights as a regularizer or constraint.
   * Reuse the **backtesting module** to stress-test candidate weight vectors under simulated market conditions.

2. **Long-term portfolio trading strategy optimizer (safer)**

   * Same architecture, but:

     * Strategies operate at lower frequency.
     * Simulations span years rather than weeks/months.
   * Optimization module can use your preferred methods (CMA-ES, evolutionary algorithms, etc.) with the backtester as objective.

---

## 6. Concrete next steps (what I’d do first)

1. **Set up repo and core dependencies** on your VPS.
2. Implement:

   * `ReturnDistribution` abstraction (fit + sample).
   * `generate_paths(returns, n_paths, n_steps)`.
   * `MarketSimulator.simulate_stock` and `simulate_option` (European call/put, expiry at last step).
3. Implement one **very simple parametric strategy** that has both stock and option variants.
4. Wire up:

   * Monte Carlo run for a fixed underlying.
   * Summary metrics and histograms for stock vs option.
5. Only after that:

   * Add a **parameter grid** and simple parallelization using `multiprocessing` or `joblib`.
   * Add QuantStats for richer reporting.

If you want, I can draft a **skeleton code layout** (folders, main classes, and function signatures) for this MVP so you can paste it into your editor and start filling in the details.

## Specification Outline

Got it. Let’s design this like you’re going to actually build it and then feed it into Spec-Driven Development.

I’ll:

* Respect your constraints (CPU VPS, Schwab API, existing libs for fits & indicators, better-than-BS pricer).
* Focus on **single-stock + options MVP** that later generalizes to portfolio products.
* Give you: architecture, main classes, function signatures, dependencies, and functional / non-functional requirements.

I’ll keep it code-oriented so it’s easy to translate into SRS / Specify specs.

---

## 1. High-level architecture

Three main domains, plus a shared “market model” layer:

1. **Data & Features Layer**

   * Data ingestion (Schwab, yfinance/CSV fallback)
   * Timescale alignment & interpolation (e.g., Fed rate, macro series)
   * Indicator & feature calculation using existing libraries

2. **Market Model & Simulation Layer**

   * Return distribution fitting & sampling
   * Price path generation (Monte Carlo)
   * Option pricing (Black-Scholes + “better” model abstraction)
   * Macro / exogenous factor handling

3. **Strategy & Backtesting Layer**

   * Strategy definitions (stock & options variants)
   * Signal generation (vectorized)
   * Market simulator (apply signals to price paths)
   * Metrics & reporting (includes QuantStats integration)

4. **Optimization Layer**

   * Parameter search (grid / random / CMA-ES / Optuna)
   * Objective evaluation (risk–reward surfaces)
   * Scenario & regime controls

We’ll make each layer modular and interface-driven.

---

## 2. Primary dependencies (Python)

Core scientific stack:

* `numpy` – vector / matrix math
* `pandas` – time series & tabular data
* `scipy` – distributions & optimization
* `numba` – optional JIT for hot loops
* `statsmodels` and/or `arch` – time-series & heavy-tail / GARCH models

Market & indicators:

* `pandas-ta` or `ta` – wide suite of technical indicators (momentum, trend, volatility)
* (Optional) `yfinance` – quick historical data if Schwab isn’t in the loop yet

Options & analytics:

* `quantstats` – portfolio/strategy performance metrics and tear sheets
* `py_vollib` or `QuantLib` (optional) – more advanced option pricing than raw Black–Scholes, if desired

Optimization:

* `optuna` or `nevergrad` (optional) – search over hyperparameters
* or plain `scipy.optimize` + `multiprocessing` / `joblib`

LLM / external analysis (later):

* Your existing LLM clients (OpenAI, Anthropic, etc.) can be wrapped as **signal generators** or **feature enrichers**, but we won’t hard-bake those into the MVP.

---

## 3. Core module and class layout

Assume a Python package structure like:

```text
quant_sys/
  data/
    data_source.py
    schwab_source.py
    yfinance_source.py
    feature_engineering.py
  models/
    distributions.py
    macro_factors.py
    option_pricing.py
    path_generator.py
  strategy/
    base.py
    stock_strategies.py
    option_strategies.py
  backtest/
    simulator.py
    metrics.py
    reporting.py
  optimize/
    optimizer.py
  config/
    settings.py
  cli/
    main.py
```

Below, I’ll outline key classes, methods, and signatures.

---

## 4. Data & Feature Layer

### 4.1 Data sources

```python
# quant_sys/data/data_source.py
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

class DataSource(ABC):
    """Abstract base for price and fundamental/macro data providers."""

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Returns OHLCV time series indexed by datetime with columns:
        ['open', 'high', 'low', 'close', 'volume'].
        """

    @abstractmethod
    def get_macro_series(
        self,
        series_name: str,
        start: datetime,
        end: datetime,
        frequency: str = "1d"
    ) -> pd.Series:
        """
        Returns a macro time series (e.g., Fed Funds rate) indexed by datetime.
        Implementation may fetch from FRED/Schwab/other source or local cache.
        """
```

Concrete implementations:

```python
# quant_sys/data/schwab_source.py
class SchwabDataSource(DataSource):
    def __init__(self, api_client):
        self.api_client = api_client

    def get_ohlcv(...): ...
    def get_macro_series(...): ...
```

```python
# quant_sys/data/yfinance_source.py
import yfinance as yf

class YFinanceDataSource(DataSource):
    def get_ohlcv(...): ...
    def get_macro_series(...): ...
```

You can set a **config** that chooses which data source to use (Schwab vs yfinance) and later benchmark them.

### 4.2 Feature & indicator calculation

Use `pandas-ta` or `ta` to avoid re-inventing indicators:

```python
# quant_sys/data/feature_engineering.py
import pandas as pd
import pandas_ta as ta

def add_technical_indicators(
    ohlcv: pd.DataFrame,
    indicators: dict
) -> pd.DataFrame:
    """
    Given OHLCV df and a spec of indicators, returns df with additional columns.

    indicators example:
    {
        "rsi": {"length": 14},
        "sma_fast": {"kind": "sma", "length": 10},
        "sma_slow": {"kind": "sma", "length": 50},
        "macd": {"fast": 12, "slow": 26, "signal": 9}
    }
    """
    df = ohlcv.copy()
    # Examples:
    df["rsi_14"] = ta.rsi(df["close"], length=indicators["rsi"]["length"])
    df["sma_fast"] = ta.sma(df["close"], length=indicators["sma_fast"]["length"])
    df["sma_slow"] = ta.sma(df["close"], length=indicators["sma_slow"]["length"])
    # etc.
    return df
```

### 4.3 Macro features and interpolation

You’re correct: you can either go event-driven or align to your main bar frequency.

```python
# quant_sys/models/macro_factors.py
import pandas as pd

def align_macro_to_prices(
    macro_series: pd.Series,
    price_index: pd.DatetimeIndex,
    method: str = "ffill"
) -> pd.Series:
    """
    Aligns a lower-frequency macro series (e.g., monthly Fed rate)
    to a higher-frequency price index (e.g., daily) via reindex + interpolation.

    method: 'ffill', 'bfill', or 'interpolate'
    """
    macro = macro_series.reindex(price_index, method="ffill" if method == "ffill" else None)
    if method == "interpolate":
        macro = macro.interpolate()
    return macro
```

Later, you can build a small **event abstraction** if you want, but for now, aligning to daily bars is clean and vectorizable.

---

## 5. Market Model & Simulation Layer

### 5.1 Distribution fitting

Use SciPy and ARCH to fit distributions and optionally GARCH / stochastic volatility models.

```python
# quant_sys/models/distributions.py
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from arch.univariate import ARCHModel

class ReturnDistribution(ABC):
    @abstractmethod
    def fit(self, returns: np.ndarray) -> None:
        """Fit distribution parameters from historical returns."""
    @abstractmethod
    def sample(self, n_paths: int, n_steps: int) -> np.ndarray:
        """Sample simulated returns array of shape (n_paths, n_steps)."""

class NormalDistribution(ReturnDistribution):
    def fit(self, returns: np.ndarray) -> None:
        self.mu, self.sigma = stats.norm.fit(returns)

    def sample(self, n_paths: int, n_steps: int) -> np.ndarray:
        return stats.norm.rvs(
            self.mu, self.sigma, size=(n_paths, n_steps)
        )

class StudentTDistribution(ReturnDistribution):
    def fit(self, returns: np.ndarray) -> None:
        self.df, self.loc, self.scale = stats.t.fit(returns)

    def sample(self, n_paths: int, n_steps: int) -> np.ndarray:
        return stats.t.rvs(
            self.df, loc=self.loc, scale=self.scale, size=(n_paths, n_steps)
        )

class GARCHStudentTDistribution(ReturnDistribution):
    """
    Model volatility clustering and fat tails via GARCH with t-distributed residuals.
    """
    def fit(self, returns: np.ndarray) -> None:
        am = ARCHModel(
            returns * 100, vol="GARCH", p=1, o=0, q=1, dist="StudentsT"
        )
        self.result = am.fit(disp="off")

    def sample(self, n_paths: int, n_steps: int) -> np.ndarray:
        # Use arch's simulation or custom logic to generate returns
        # out = self.result.simulate(...)
        # return out["data"].values.reshape(n_paths, n_steps) / 100
        ...
```

You can select which class to use via config.

### 5.2 Path generator

```python
# quant_sys/models/path_generator.py
import numpy as np

def generate_price_paths(
    s0: float,
    distribution: ReturnDistribution,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    Returns array of shape (n_paths, n_steps) with simulated prices.
    """
    r = distribution.sample(n_paths, n_steps)  # log returns
    log_s = np.log(s0) + np.cumsum(r, axis=1)
    return np.exp(log_s)
```

### 5.3 Option pricing models (beyond Black–Scholes)

Approach:

* Define an **OptionPricer** interface.
* Implement:

  * Black–Scholes (but using per-strike implied vol, not constant)
  * A more realistic model (e.g. local vol or stochastic vol via QuantLib, or “market-implied” pricing where you just use observed IV surface and BS)

For your CPU-only MVP, use BS with **per-strike IV** as a strong baseline. That already fixes the biggest Black–Scholes sin (constant vol). Later you can plug in Heston via QuantLib if needed.

```python
# quant_sys/models/option_pricing.py
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

@dataclass
class OptionSpec:
    kind: str        # 'call' or 'put'
    strike: float
    maturity_days: int
    implied_vol: float  # or func of moneyness / maturity
    risk_free_rate: float = 0.02

class OptionPricer(ABC):
    @abstractmethod
    def price(
        self,
        spot: np.ndarray,
        spec: OptionSpec,
        ttm: np.ndarray
    ) -> np.ndarray:
        """
        Price option(s) given spot array and time-to-maturity (in years).
        spot: np.ndarray of shape (n_paths, n_steps) or (n_steps,)
        ttm:  np.ndarray of same shape as spot or (n_steps,)
        """

class BlackScholesPricer(OptionPricer):
    def price(self, spot, spec: OptionSpec, ttm):
        # Vectorized BS formula with per-step ttm
        # Use numpy for d1, d2, Nd1, Nd2
        ...
        return prices

# Optional: Heston/Local Vol pricer using QuantLib
class HestonPricer(OptionPricer):
    def __init__(self, calibrated_params):
        self.params = calibrated_params

    def price(self, spot, spec: OptionSpec, ttm):
        # Call into QuantLib or your own Heston implementation
        ...
```

For **this application** (short-horizon, high-vol single names; MC paths you already control):

* Use **BS with implied vol surface** as pricer for mark-to-market.
* Use **intrinsic payoff at expiry** for Monte Carlo terminal P&L if you don’t care about intermediate MTM.

---

## 6. Strategy & Backtesting Layer

### 6.1 Strategy interface

```python
# quant_sys/strategy/base.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any

class Strategy(ABC):
    """
    Strategy operates on aligned feature & price arrays and returns signals.
    """

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        prices: df with columns ['open','high','low','close','volume']
        features: df with indicators, macro, etc.
        returns:
          {
            "stock": np.ndarray of shape (n_steps,), # position in shares
            "option": {
                "position": np.ndarray of shape (n_steps,),  # number of contracts
                "spec": OptionSpec
            }
          }
        """
```

For Monte Carlo, you can generalize to `(n_paths, n_steps)` by broadcasting the same signal rules across paths.

Example simple strategy class:

```python
# quant_sys/strategy/stock_strategies.py
class MeanReversionStockStrategy(Strategy):
    def generate_signals(self, prices, features, params):
        close = prices["close"].values
        sma = features["sma_fast"].values
        threshold = params.get("threshold", 0.03)
        position = np.zeros_like(close, dtype=float)

        # example: buy when close < sma*(1 - threshold), flat otherwise
        buy_mask = close < sma * (1 - threshold)
        position[buy_mask] = params.get("position_size", 100.0)
        return {"stock": position}
```

Option counterpart:

```python
# quant_sys/strategy/option_strategies.py
from quant_sys.models.option_pricing import OptionSpec

class CallOverlayStrategy(Strategy):
    def generate_signals(self, prices, features, params):
        close = prices["close"].values
        # Option spec based on params:
        spec = OptionSpec(
            kind="call",
            strike=params["strike_factor"] * close[0],
            maturity_days=params["maturity_days"],
            implied_vol=params["implied_vol"]
        )
        position = np.zeros_like(close, dtype=float)
        # same trigger as stock strategy:
        buy_mask = close < features["sma_fast"] * (1 - params["threshold"])
        position[buy_mask] = params.get("contracts", 1.0)
        return {"option": {"position": position, "spec": spec}}
```

### 6.2 Market simulator

Handles both **historical** and **simulated** price arrays.

```python
# quant_sys/backtest/simulator.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from quant_sys.models.option_pricing import OptionPricer, OptionSpec

class MarketSimulator:
    def __init__(
        self,
        option_pricer: OptionPricer,
        commission_per_share: float = 0.0,
        commission_per_contract: float = 0.0,
        slippage_bps: float = 0.0
    ):
        self.op = option_pricer
        self.comm_share = commission_per_share
        self.comm_contract = commission_per_contract
        self.slippage_bps = slippage_bps

    def simulate_stock(
        self,
        prices: pd.Series,
        position: np.ndarray
    ) -> pd.Series:
        """
        prices: close series indexed by datetime
        position: shares held at each time step (same length)
        Returns: equity curve (Series of portfolio value changes or cumulative P&L).
        """
        px = prices.values
        # Simple: P&L = position[t-1] * (px[t] - px[t-1])
        pnl = np.zeros_like(px, dtype=float)
        pnl[1:] = position[:-1] * (px[1:] - px[:-1])
        # TODO: subtract commissions according to trades:
        # trades = np.diff(position)
        # cost = np.abs(trades) * self.comm_share
        return pd.Series(np.cumsum(pnl), index=prices.index)

    def simulate_option_path(
        self,
        spot: pd.Series,
        position: np.ndarray,
        spec: OptionSpec
    ) -> pd.Series:
        """
        spot: underlying spot prices
        position: contracts held per step
        """
        dates = spot.index
        n = len(dates)
        # time to maturity (years) at each step:
        ttm_days = np.maximum(
            spec.maturity_days - np.arange(n),
            0
        )
        ttm = ttm_days / 252.0
        spot_arr = spot.values
        # compute option price at each step:
        opt_prices = self.op.price(
            spot_arr, spec, ttm
        )
        # mark-to-market P&L on position:
        pnl = np.zeros_like(opt_prices)
        pnl[1:] = position[:-1] * (opt_prices[1:] - opt_prices[:-1])
        return pd.Series(np.cumsum(pnl), index=dates)

    def simulate(
        self,
        prices: pd.DataFrame,
        strategy_output: Dict[str, Any]
    ) -> Dict[str, pd.Series]:
        """
        Orchestrates stock & option simulation depending on what the strategy returned.
        """
        results = {}
        if "stock" in strategy_output:
            results["stock_equity"] = self.simulate_stock(
                prices["close"], strategy_output["stock"]
            )
        if "option" in strategy_output:
            so = strategy_output["option"]
            results["option_equity"] = self.simulate_option_path(
                prices["close"], so["position"], so["spec"]
            )
        return results
```

For Monte Carlo paths `(n_paths, n_steps)`, you can vectorize `simulate_stock` and `simulate_option_path` using `numpy`/`numba`, returning a `(n_paths, n_steps)` P&L matrix.

### 6.3 Metrics & reporting

```python
# quant_sys/backtest/metrics.py
import pandas as pd
import quantstats as qs

def summarize_equity_curve(
    equity: pd.Series,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Returns key stats (CAGR, Sharpe, Max Drawdown, CVaR, etc.)
    """
    returns = equity.pct_change().dropna()
    stats = {
        "cagr": qs.stats.cagr(returns),
        "sharpe": qs.stats.sharpe(returns, rf=risk_free_rate),
        "sortino": qs.stats.sortino(returns, rf=risk_free_rate),
        "max_drawdown": qs.stats.max_drawdown(returns),
        "cvar_5": qs.stats.cvar(returns, 0.05),
    }
    return stats

def generate_tearsheet(
    returns: pd.Series,
    benchmark: str | pd.Series,
    output_path: str
) -> None:
    """
    Generates QuantStats HTML tear sheet.
    """
    qs.reports.html(
        returns,
        benchmark=benchmark,
        output=output_path
    )
```

---

## 7. Optimization Layer

```python
# quant_sys/optimize/optimizer.py
from typing import Dict, Any, Callable, List
import itertools
import numpy as np

def grid_search(
    param_grid: Dict[str, List[Any]],
    simulate_fn: Callable[[Dict[str, Any]], Dict[str, float]]
) -> List[Dict[str, Any]]:
    """
    param_grid: {"threshold": [0.01, 0.02], "maturity_days":[14, 30], ...}
    simulate_fn: function that takes params dict and returns metrics dict:
      {"objective": float, "sharpe": float, "max_dd": float, ...}
    Returns: list of results [{ "params":..., "metrics":... }, ...] sorted by objective.
    """
    keys = list(param_grid.keys())
    all_results = []
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        metrics = simulate_fn(params)
        all_results.append({"params": params, "metrics": metrics})
    all_results.sort(key=lambda r: r["metrics"]["objective"], reverse=True)
    return all_results
```

Your `simulate_fn` will:

1. Build features from data.
2. Run strategy → signals.
3. Simulate equity (stock vs option).
4. Compute metrics and a scalar objective (risk–reward surface point).

For Monte Carlo, `simulate_fn` will **internally call** the path generator and simulator across simulated paths, then aggregate results (e.g., mean equity curve or distribution of final P&L).

---

## 8. Functional requirements (for SDD / Specify)

You can turn these directly into specification items.

**FR-1: Data ingestion**

* FR-1.1: The system shall fetch OHLCV time series for a specified symbol, date range, and interval from a configured `DataSource`.
* FR-1.2: The system shall support at least two data sources: Schwab API and yfinance (or CSV).
* FR-1.3: The system shall fetch macroeconomic series (e.g., Fed Funds rate) and align them to the OHLCV index via forward-fill or interpolation.

**FR-2: Feature generation**

* FR-2.1: The system shall compute technical indicators using existing libraries (e.g., `pandas-ta`) without reimplementing standard formulas.
* FR-2.2: The system shall allow configuration of which indicators to compute and their parameters.

**FR-3: Distribution modeling and simulation**

* FR-3.1: The system shall compute log returns from historical closing prices.
* FR-3.2: The system shall fit a configurable return distribution model (Normal, Student’s t, GARCH+Student’s t).
* FR-3.3: The system shall generate `N` simulated price paths of length `T` based on the fitted distribution.
* FR-3.4: The system shall expose a pluggable `ReturnDistribution` interface supporting additional models.

**FR-4: Option pricing**

* FR-4.1: The system shall define an `OptionSpec` object encapsulating option type, strike, maturity, implied volatility, and risk-free rate.
* FR-4.2: The system shall implement a `BlackScholesPricer` that prices calls and puts vectorially, using per-strike implied volatility.
* FR-4.3: The system shall support swapping in a more advanced `OptionPricer` (e.g., Heston / local volatility via QuantLib) without changing strategy or simulator code.
* FR-4.4: The system shall support using either terminal intrinsic payoff or daily mark-to-market pricing for options during simulation.

**FR-5: Strategy definition**

* FR-5.1: The system shall define a `Strategy` base class with a `generate_signals` method that outputs stock and/or option position arrays given price and feature data.
* FR-5.2: The system shall provide at least one reference strategy that has both a stock and an options implementation using the same entry/exit conditions.

**FR-6: Backtesting & Monte Carlo**

* FR-6.1: The system shall simulate strategy performance on historical price series, producing an equity curve.
* FR-6.2: The system shall simulate strategy performance on Monte Carlo price paths, producing distributions of equity curves and final P&L.
* FR-6.3: The backtesting module shall be independent of strategy implementation, consuming only signals and `OptionSpec`.
* FR-6.4: The backtesting module shall support both stock P&L and option P&L computations.

**FR-7: Metrics & reporting**

* FR-7.1: The system shall compute core performance metrics for each equity curve: CAGR, Sharpe ratio, Sortino ratio, max drawdown, and CVaR.
* FR-7.2: The system shall integrate with QuantStats to generate HTML tear sheets for selected backtests.
* FR-7.3: The system shall export summarized results (including parameters and metrics) in a machine-readable format (e.g., JSON, CSV) for further analysis.

**FR-8: Optimization**

* FR-8.1: The system shall provide a generic optimization interface that accepts a parameter grid and a simulation function.
* FR-8.2: The system shall compute a scalar objective function from strategy performance metrics (e.g., a weighted combination of expected return, CVaR, and max drawdown).
* FR-8.3: The system shall return top-ranked parameter sets and associated metrics, suitable for building risk–reward surfaces (e.g., for stock vs options comparisons).

---

## 9. Non-functional requirements

**NFR-1: Performance**

* NFR-1.1: The system shall leverage NumPy vectorization as the default implementation pattern; Python loops in core simulation code should be avoided or JIT-compiled with Numba.
* NFR-1.2: The system shall be capable of running at least 1,000 Monte Carlo paths of 60 trading days for a single strategy configuration in under X seconds on a CPU-only VPS (you can pick your target once you test).
* NFR-1.3: The system shall support parallel evaluation of parameter configurations using `multiprocessing` or equivalent.

**NFR-2: Modularity**

* NFR-2.1: Data sources, distribution models, option pricers, strategies, and optimizers shall be implemented via abstractions/interfaces so they can be replaced or extended without modifying unrelated modules.
* NFR-2.2: The backtesting engine shall not depend on a specific broker or data provider.

**NFR-3: Maintainability & Spec-Driven Development**

* NFR-3.1: All public classes and methods shall have docstrings specifying purpose, inputs, outputs, and side effects, suitable for automated doc generation.
* NFR-3.2: The project shall be structured into logically separated modules (data, models, strategy, backtest, optimize) to support Spec-Driven Development using GitHub Specify.
* NFR-3.3: Configuration (e.g., data source selection, distribution model, option pricer implementation) shall be externalized (e.g., `config/settings.py` or `.yaml`) rather than hard-coded.

**NFR-4: Testability**

* NFR-4.1: Core components (distribution fitting, path generation, option pricing, P&L calculations) shall be unit testable with deterministic seeds.
* NFR-4.2: The system shall allow injecting mock data sources for tests to avoid live broker calls.

**NFR-5: Deployment**

* NFR-5.1: The system shall be deployable as a containerized service on a CPU-only VPS.
* NFR-5.2: The system shall expose at least a CLI entrypoint (later possibly a REST or GUI interface) for running simulations and optimizations.

## Workflow Skeleton

Great. Here’s a full end-to-end skeleton you can drop into a repo and start filling in. It is intentionally minimal but wired all the way through: load data → compute features → fit distribution → generate paths → run stock & option strategies → simulate → compute metrics.

I’ll assume a package name `quant_sys`.

---

## 1. Directory layout

```text
quant_sys/
  __init__.py

  config/
    __init__.py
    settings.py

  data/
    __init__.py
    data_source.py
    schwab_source.py
    yfinance_source.py
    feature_engineering.py

  models/
    __init__.py
    distributions.py
    macro_factors.py
    path_generator.py
    option_pricing.py

  strategy/
    __init__.py
    base.py
    stock_strategies.py
    option_strategies.py

  backtest/
    __init__.py
    simulator.py
    metrics.py

  optimize/
    __init__.py
    optimizer.py

  cli/
    __init__.py
    main.py
```

You can adapt names, but this is concrete enough to start.

---

## 2. `config/settings.py`

```python
# quant_sys/config/settings.py

from dataclasses import dataclass

@dataclass
class DataConfig:
    source: str = "yfinance"  # or "schwab"
    default_interval: str = "1d"

@dataclass
class SimulationConfig:
    distribution: str = "student_t"  # "normal", "garch_t"
    n_paths: int = 1000
    n_steps: int = 60

@dataclass
class OptionConfig:
    pricer: str = "black_scholes"
    risk_free_rate: float = 0.02

@dataclass
class Settings:
    data: DataConfig = DataConfig()
    simulation: SimulationConfig = SimulationConfig()
    options: OptionConfig = OptionConfig()

settings = Settings()
```

---

## 3. Data layer

### 3.1 `data/data_source.py`

```python
# quant_sys/data/data_source.py

from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

class DataSource(ABC):
    """Abstract base for price and macro data providers."""

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Returns OHLCV DataFrame indexed by datetime with columns:
        ['open', 'high', 'low', 'close', 'volume'].
        """
        raise NotImplementedError

    @abstractmethod
    def get_macro_series(
        self,
        series_name: str,
        start: datetime,
        end: datetime,
        frequency: str = "1d",
    ) -> pd.Series:
        """
        Returns macro series (e.g. Fed Funds rate) indexed by datetime.
        Implementation may fetch from external service or local cache.
        """
        raise NotImplementedError
```

### 3.2 `data/yfinance_source.py`

```python
# quant_sys/data/yfinance_source.py

from datetime import datetime
import pandas as pd
import yfinance as yf

from .data_source import DataSource

class YFinanceDataSource(DataSource):
    """DataSource implementation using yfinance as a quick baseline."""

    def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, interval=interval)
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df = df[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index)
        return df

    def get_macro_series(
        self,
        series_name: str,
        start: datetime,
        end: datetime,
        frequency: str = "1d",
    ) -> pd.Series:
        """
        Stub for macro series; later you can use FRED or other sources.
        For now, returns an empty Series.
        """
        return pd.Series(dtype=float)
```

### 3.3 `data/schwab_source.py` (stub)

```python
# quant_sys/data/schwab_source.py

from datetime import datetime
import pandas as pd

from .data_source import DataSource

class SchwabDataSource(DataSource):
    """Placeholder for Schwab API data source."""

    def __init__(self, api_client) -> None:
        self.api_client = api_client

    def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        # TODO: implement Schwab API calls
        raise NotImplementedError("SchwabDataSource.get_ohlcv not implemented")

    def get_macro_series(
        self,
        series_name: str,
        start: datetime,
        end: datetime,
        frequency: str = "1d",
    ) -> pd.Series:
        # TODO: implement when you have a macro source
        return pd.Series(dtype=float)
```

### 3.4 `data/feature_engineering.py`

```python
# quant_sys/data/feature_engineering.py

import pandas as pd
import pandas_ta as ta
from typing import Dict, Any

def add_technical_indicators(
    ohlcv: pd.DataFrame,
    indicators: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV DataFrame.

    indicators example:
    {
      "rsi": {"length": 14},
      "sma_fast": {"kind": "sma", "length": 10},
      "sma_slow": {"kind": "sma", "length": 50},
    }
    """
    df = ohlcv.copy()

    for name, spec in indicators.items():
        kind = spec.get("kind", name)
        if kind == "rsi":
            df[f"rsi_{spec['length']}"] = ta.rsi(df["close"], length=spec["length"])
        elif kind == "sma":
            df[f"sma_{spec['length']}"] = ta.sma(df["close"], length=spec["length"])
        # Extend with other indicator kinds as needed

    return df
```

---

## 4. Market model layer

### 4.1 `models/macro_factors.py`

```python
# quant_sys/models/macro_factors.py

import pandas as pd

def align_macro_to_prices(
    macro_series: pd.Series,
    price_index: pd.DatetimeIndex,
    method: str = "ffill",
) -> pd.Series:
    """
    Align macro series (e.g., Fed rate) to price index via reindex + ffill / interpolate.
    """
    if macro_series.empty:
        return pd.Series(index=price_index, dtype=float)

    macro = macro_series.reindex(price_index)

    if method == "ffill":
        macro = macro.ffill()
    elif method == "bfill":
        macro = macro.bfill()
    elif method == "interpolate":
        macro = macro.interpolate()

    return macro
```

### 4.2 `models/distributions.py`

```python
# quant_sys/models/distributions.py

from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from arch.univariate import ARCHModel

class ReturnDistribution(ABC):
    @abstractmethod
    def fit(self, returns: np.ndarray) -> None:
        """Fit distribution parameters from historical returns."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, n_paths: int, n_steps: int) -> np.ndarray:
        """Sample log returns array of shape (n_paths, n_steps)."""
        raise NotImplementedError


class NormalDistribution(ReturnDistribution):
    def fit(self, returns: np.ndarray) -> None:
        self.mu, self.sigma = stats.norm.fit(returns)

    def sample(self, n_paths: int, n_steps: int) -> np.ndarray:
        return stats.norm.rvs(self.mu, self.sigma, size=(n_paths, n_steps))


class StudentTDistribution(ReturnDistribution):
    def fit(self, returns: np.ndarray) -> None:
        self.df, self.loc, self.scale = stats.t.fit(returns)

    def sample(self, n_paths: int, n_steps: int) -> np.ndarray:
        return stats.t.rvs(self.df, loc=self.loc, scale=self.scale, size=(n_paths, n_steps))


class GARCHStudentTDistribution(ReturnDistribution):
    """
    GARCH(1,1) with Student's t residuals.
    """

    def fit(self, returns: np.ndarray) -> None:
        am = ARCHModel(returns * 100, vol="GARCH", p=1, o=0, q=1, dist="StudentsT")
        self.result = am.fit(disp="off")

    def sample(self, n_paths: int, n_steps: int) -> np.ndarray:
        # NOTE: this is a stub for now.
        # You can use self.result.forecast or simulate to generate paths.
        raise NotImplementedError("GARCHStudentTDistribution.sample not implemented yet")
```

### 4.3 `models/path_generator.py`

```python
# quant_sys/models/path_generator.py

import numpy as np
from .distributions import ReturnDistribution

def generate_price_paths(
    s0: float,
    distribution: ReturnDistribution,
    n_paths: int,
    n_steps: int,
) -> np.ndarray:
    """
    Generate price paths from a fitted return distribution.
    Returns array of shape (n_paths, n_steps).
    """
    r = distribution.sample(n_paths, n_steps)
    log_s = np.log(s0) + np.cumsum(r, axis=1)
    return np.exp(log_s)
```

### 4.4 `models/option_pricing.py`

```python
# quant_sys/models/option_pricing.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

@dataclass
class OptionSpec:
    kind: str            # 'call' or 'put'
    strike: float
    maturity_days: int
    implied_vol: float
    risk_free_rate: float = 0.02


class OptionPricer(ABC):
    @abstractmethod
    def price(
        self,
        spot: np.ndarray,
        spec: OptionSpec,
        ttm: np.ndarray,
    ) -> np.ndarray:
        """Return option price array with same shape as spot."""
        raise NotImplementedError


class BlackScholesPricer(OptionPricer):
    """
    Black-Scholes pricer using per-option implied vol.
    """

    def price(
        self,
        spot: np.ndarray,
        spec: OptionSpec,
        ttm: np.ndarray,
    ) -> np.ndarray:
        # Avoid division by zero
        ttm = np.maximum(ttm, 1e-8)

        S = spot.astype(float)
        K = spec.strike
        r = spec.risk_free_rate
        sigma = spec.implied_vol

        sqrt_t = np.sqrt(ttm)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * ttm) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        if spec.kind == "call":
            prices = S * norm.cdf(d1) - K * np.exp(-r * ttm) * norm.cdf(d2)
        else:
            prices = K * np.exp(-r * ttm) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return prices
```

---

## 5. Strategy layer

### 5.1 `strategy/base.py`

```python
# quant_sys/strategy/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class Strategy(ABC):
    """
    Strategy operates on aligned price & feature data and returns position signals.
    """

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Should return a dict like:
        {
          "stock": position_array,  # optional
          "option": {               # optional
            "position": position_array,
            "spec": OptionSpec,
          }
        }
        """
        raise NotImplementedError
```

### 5.2 `strategy/stock_strategies.py`

```python
# quant_sys/strategy/stock_strategies.py

from typing import Dict, Any
import numpy as np
import pandas as pd

from .base import Strategy

class MeanReversionStockStrategy(Strategy):
    """
    Example: buy when price below SMA by threshold, flat otherwise.
    """

    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        close = prices["close"].values
        sma = features["sma_fast"].values
        threshold = params.get("threshold", 0.03)
        size = params.get("position_size", 100.0)

        position = np.zeros_like(close, dtype=float)
        buy_mask = close < sma * (1 - threshold)
        position[buy_mask] = size

        return {"stock": position}
```

### 5.3 `strategy/option_strategies.py`

```python
# quant_sys/strategy/option_strategies.py

from typing import Dict, Any
import numpy as np
import pandas as pd

from .base import Strategy
from quant_sys.models.option_pricing import OptionSpec

class MeanReversionCallStrategy(Strategy):
    """
    Same entry condition as stock strategy, but uses ATM calls instead.
    """

    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        close = prices["close"].values
        sma = features["sma_fast"].values

        threshold = params.get("threshold", 0.03)
        contracts = params.get("contracts", 1.0)
        maturity_days = params.get("maturity_days", 30)
        iv = params.get("implied_vol", 0.4)

        # Simple: strike = first close (ATM at entry)
        strike = close[0]

        spec = OptionSpec(
            kind="call",
            strike=strike,
            maturity_days=maturity_days,
            implied_vol=iv,
        )

        position = np.zeros_like(close, dtype=float)
        buy_mask = close < sma * (1 - threshold)
        position[buy_mask] = contracts

        return {"option": {"position": position, "spec": spec}}
```

---

## 6. Backtesting layer

### 6.1 `backtest/simulator.py`

```python
# quant_sys/backtest/simulator.py

from typing import Dict, Any
import numpy as np
import pandas as pd

from quant_sys.models.option_pricing import OptionPricer, OptionSpec

class MarketSimulator:
    """
    Simulates stock and option P&L given prices and position signals.
    """

    def __init__(
        self,
        option_pricer: OptionPricer,
        commission_per_share: float = 0.0,
        commission_per_contract: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> None:
        self.op = option_pricer
        self.comm_share = commission_per_share
        self.comm_contract = commission_per_contract
        self.slippage_bps = slippage_bps

    def simulate_stock(
        self,
        prices: pd.Series,
        position: np.ndarray,
    ) -> pd.Series:
        px = prices.values.astype(float)
        pnl = np.zeros_like(px, dtype=float)
        pnl[1:] = position[:-1] * (px[1:] - px[:-1])

        # TODO: subtract commissions/slippage based on trades (diff of position)
        equity = np.cumsum(pnl)
        return pd.Series(equity, index=prices.index)

    def simulate_option_path(
        self,
        spot: pd.Series,
        position: np.ndarray,
        spec: OptionSpec,
    ) -> pd.Series:
        dates = spot.index
        px = spot.values.astype(float)
        n = len(px)

        ttm_days = np.maximum(spec.maturity_days - np.arange(n), 0)
        ttm_years = ttm_days / 252.0

        opt_prices = self.op.price(px, spec, ttm_years)
        pnl = np.zeros_like(opt_prices, dtype=float)
        pnl[1:] = position[:-1] * (opt_prices[1:] - opt_prices[:-1])

        # TODO: commissions / slippage for contracts
        equity = np.cumsum(pnl)
        return pd.Series(equity, index=dates)

    def simulate(
        self,
        prices: pd.DataFrame,
        strategy_output: Dict[str, Any],
    ) -> Dict[str, pd.Series]:
        results: Dict[str, pd.Series] = {}
        if "stock" in strategy_output:
            results["stock_equity"] = self.simulate_stock(
                prices["close"], strategy_output["stock"]
            )
        if "option" in strategy_output:
            so = strategy_output["option"]
            results["option_equity"] = self.simulate_option_path(
                prices["close"], so["position"], so["spec"]
            )
        return results
```

### 6.2 `backtest/metrics.py`

```python
# quant_sys/backtest/metrics.py

from typing import Dict
import pandas as pd
import quantstats as qs

def summarize_equity_curve(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    returns = equity.pct_change().dropna()
    stats = {
        "cagr": qs.stats.cagr(returns),
        "sharpe": qs.stats.sharpe(returns, rf=risk_free_rate),
        "sortino": qs.stats.sortino(returns, rf=risk_free_rate),
        "max_drawdown": qs.stats.max_drawdown(returns),
        "cvar_5": qs.stats.cvar(returns, 0.05),
    }
    return stats

def generate_tearsheet(
    returns: pd.Series,
    benchmark: str | pd.Series,
    output_path: str,
) -> None:
    qs.reports.html(returns, benchmark=benchmark, output=output_path)
```

---

## 7. Optimization (stub)

```python
# quant_sys/optimize/optimizer.py

from typing import Dict, Any, Callable, List
import itertools

def grid_search(
    param_grid: Dict[str, List[Any]],
    simulate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    param_grid: {"threshold":[0.02,0.03], "maturity_days":[14,30], ...}
    simulate_fn: takes params dict, returns {"metrics":{...}, "objective":float}
    """
    keys = list(param_grid.keys())
    results: List[Dict[str, Any]] = []

    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        out = simulate_fn(params)
        results.append({"params": params, **out})

    results.sort(key=lambda r: r["objective"], reverse=True)
    return results
```

---

## 8. CLI wiring (`cli/main.py`)

This is the “end-to-end” script: load data, compute features, fit distribution, generate paths, run two strategies, simulate, compare.

```python
# quant_sys/cli/main.py

import argparse
from datetime import datetime
import numpy as np
import pandas as pd

from quant_sys.config.settings import settings
from quant_sys.data.yfinance_source import YFinanceDataSource
from quant_sys.data.feature_engineering import add_technical_indicators
from quant_sys.models.distributions import (
    NormalDistribution,
    StudentTDistribution,
)
from quant_sys.models.path_generator import generate_price_paths
from quant_sys.models.option_pricing import BlackScholesPricer
from quant_sys.strategy.stock_strategies import MeanReversionStockStrategy
from quant_sys.strategy.option_strategies import MeanReversionCallStrategy
from quant_sys.backtest.simulator import MarketSimulator
from quant_sys.backtest.metrics import summarize_equity_curve

def get_distribution_model(name: str):
    if name == "normal":
        return NormalDistribution()
    elif name == "student_t":
        return StudentTDistribution()
    # extend with GARCHStudentTDistribution etc.
    raise ValueError(f"Unknown distribution model: {name}")

def main():
    parser = argparse.ArgumentParser(
        description="MVP: compare stock vs option strategies on simulated paths."
    )
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    args = parser.parse_args()

    symbol = args.symbol
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    # 1. Load historical data
    ds = YFinanceDataSource()  # later: choose based on settings.data.source
    ohlcv = ds.get_ohlcv(symbol, start, end, settings.data.default_interval)

    # 2. Add indicators
    indicators = {
        "sma_fast": {"kind": "sma", "length": 20},
    }
    features = add_technical_indicators(ohlcv, indicators)

    # 3. Fit return distribution
    log_returns = np.log(ohlcv["close"]).diff().dropna().values
    dist_model = get_distribution_model(settings.simulation.distribution)
    dist_model.fit(log_returns)

    # 4. Generate Monte Carlo paths
    s0 = float(ohlcv["close"].iloc[-1])
    n_paths = settings.simulation.n_paths
    n_steps = settings.simulation.n_steps

    price_paths = generate_price_paths(s0, dist_model, n_paths, n_steps)
    # For MVP, just use one representative path for strategy logic,
    # or aggregate across paths later.
    # Here we just treat one path as if it were "future prices".
    path_index = pd.date_range(
        end + pd.Timedelta(days=1),
        periods=n_steps,
        freq="B",
    )
    path_df = pd.DataFrame(
        {
            "open": price_paths[0],
            "high": price_paths[0],
            "low": price_paths[0],
            "close": price_paths[0],
            "volume": 0.0,
        },
        index=path_index,
    )
    # For features on path, reuse last indicators or recompute from path_df
    path_features = add_technical_indicators(path_df, indicators)

    # 5. Instantiate strategies
    stock_strategy = MeanReversionStockStrategy()
    option_strategy = MeanReversionCallStrategy()

    stock_params = {
        "threshold": 0.03,
        "position_size": 100.0,
    }
    option_params = {
        "threshold": 0.03,
        "contracts": 1.0,
        "maturity_days": 30,
        "implied_vol": 0.5,
    }

    stock_signals = stock_strategy.generate_signals(path_df, path_features, stock_params)
    option_signals = option_strategy.generate_signals(path_df, path_features, option_params)

    # 6. Simulate equity curves
    pricer = BlackScholesPricer()
    simulator = MarketSimulator(pricer)

    stock_results = simulator.simulate(path_df, stock_signals)
    option_results = simulator.simulate(path_df, option_signals)

    stock_equity = stock_results["stock_equity"]
    option_equity = option_results["option_equity"]

    # 7. Compute metrics
    stock_stats = summarize_equity_curve(stock_equity)
    option_stats = summarize_equity_curve(option_equity)

    print(f"\n=== {symbol} STOCK strategy stats ===")
    for k, v in stock_stats.items():
        print(f"{k}: {v}")

    print(f"\n=== {symbol} OPTION strategy stats ===")
    for k, v in option_stats.items():
        print(f"{k}: {v}")

    # Later: extend to full Monte Carlo distribution (loop over paths, aggregate).

if __name__ == "__main__":
    main()
```

---

This gives you:

* A coherent package structure.
* All key classes and function signatures.
* A CLI that runs a minimal single-path comparison for stock vs call strategy, which you can extend to full Monte Carlo and parameter sweeps.

