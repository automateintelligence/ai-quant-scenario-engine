# Data Model

Parent: `plan.md` (spec-driven). Used by contracts and quickstart examples.

## Core Entities

### Regime
- **Fields**: `label` (enum: neutral, strong-neutral, low-volatility, volatility-dir-uncertain, mild-bearish, strong-bearish, mild-bullish, strong-bullish), `mode` (table|calibrated|explicit), `mean_daily_return` (float), `daily_vol` (float), `skew` (float), `kurtosis_excess` (float), `source` (config|calibrated|override).
- **Validation**: Unknown labels error; explicit overrides take precedence; multi-day horizons compound daily params.

### ReturnDistribution
- **Interface**: Shared with 001-mvp-pipeline. Supports two workflows:
  - **009 workflow** (regime-driven): `generate_paths(s0, trade_horizon, bars_per_day, regime_params, seed) -> np.ndarray[n_paths, n_steps]`
  - **001 workflow** (historical fit): `fit(returns) -> None`, `sample(n_paths, n_steps) -> np.ndarray` (not used in 009)
- **Implementations**: garch_t, student_t, laplacian, bootstrap (regime-conditioned), ml_conditional (future).
- **Validation**: Seeds required for reproducibility; cap `n_paths` to `max_paths` (default 20k); record variance stats for CI.

### OptionPricer
- **Interface**: `price_option(underlying, strike, maturity, rate, dividend, iv_state, type)`; `greeks(...)`.
- **Implementations**: bjerksund_stensland (default), black_scholes (fallback), heston (advanced), slv/svi (future).
- **Validation**: On failure fallback to BS; if BS fails skip candidate and log diagnostics (FR-021/FR-022); supports IV modes constant|sticky-delta|custom fn.

### IVSurface
- **Fields**: grid of `(log_moneyness, tau, iv)`; interpolation method bilinear; metadata `as_of`, `source`.
- **Validation**: Must have ≥2 strikes × 2 expiries; missing IV points skipped with warning; provide `iv_at(S,K,T)`.

### CandidateStructure
- **Fields**: `structure_type` (vertical, iron_condor, straddle, strangle, butterfly future), `legs[]` (side, type, strike, expiry, qty, fill_price_source), `capital_used`, `max_loss`, `expiry`, `width`, `liquidity_ok` (bool), `filters_passed` (list).
- **Validation**: Generated only from Stage 0-2 filtered strikes/expiries; enforce width limits; ensure leg expiries consistent.

### Metrics
- **Fields**: `E_PnL`, `CI_epnl`, `POP_0`, `POP_target`, `ROC`, `MaxLoss_trade`, `VaR_5`, `CVaR_5`, `Delta`, `Theta`, `Gamma`, `Vega`, `score`, `score_decomposition`, `path_count`, `variance_flags`.
- **Validation**: CI requires sample variance; POP_target default profit_target=$500; score uses config weights; path_count recorded for adaptive steps.

### StrategyScorer
- **Interface**: `score(candidate, metrics, config) -> float`.
- **Implementations**: intraday-spreads (default), directional-bullish/bearish (future), volatility-play, gamma-scalping; plugin loader discovers scorers in `scorers/`.
- **Validation**: Weights loaded from config; normalization documented; plugins must register name and version.

### Position (for monitoring)
- **Fields**: `legs[]` (strike, type, side, qty, entry_price), `entry_time`, `trade_horizon`, `regime`, `config_snapshot`, `H_remaining`, `alerts` (profit_target, stop_loss), `last_mark_time`, `last_metrics`.
- **Validation**: `H_remaining = max(1, trade_horizon - days_elapsed)`; config snapshot required for replay; alerts optional but validated if provided.

### RunArtifacts
- **Fields**: `top_10` (list[CandidateStructure+Metrics]), `top_100_cache`, `diagnostics` (stage counts, rejection breakdown), `orders_json` (broker-ready), `logs_path`, `ci_reports` (E[PnL]/POP intervals), `runtime_s`.
- **Validation**: Diagnostics must include filter rejection breakdown and pricer failures; runtime recorded for SC-001/SC-012 checks.

## Relationships
- Regime → ReturnDistribution → paths → OptionPricer valuations → Metrics → StrategyScorer → Ranking.
- CandidateStructure generated from option chain + filters; Metrics attach to CandidateStructure; RunArtifacts cache ordered outputs.
- Position uses CandidateStructure + updated market data for monitoring; reuses OptionPricer and ReturnDistribution for remaining horizon paths.

## State & Lifecycle
1. Load option chain (Schwab API) and config/regime.
2. Stage 0/1 filter expiries and strikes; Stage 2 generate structures.
3. Stage 3 analytic prefilter computes approximate metrics and applies hard constraints, keeping top K per structure.
4. Stage 4 MC scoring computes full metrics, confidence intervals, and scores.
5. Rank, emit Top-10 + cache Top-100 + diagnostics + orders JSON.
6. Monitoring flow loads Position JSON, fetches live data, recomputes remaining-horizon metrics/alerts, and logs/outputs events.
