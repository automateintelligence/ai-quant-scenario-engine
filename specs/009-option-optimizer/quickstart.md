# Quickstart: Option Strategy Optimizer (009)

## Prereqs
- Python 3.11
- Install deps (inside repo): `pip install -e .[dev]` (pytest, ruff included)
- Config file: `config.yml` with regimes, pricing, distributions, filters, scoring sections shared with US1.

## Optimize a strategy (primary CLI)
```bash
qse optimize-strategy --ticker NVDA \
  --regime strong-bullish \
  --trade-horizon 3 \
  --config config.yml \
  --data-source schwab \
  --override "mc.num_paths=5000" \
  --override "filters.max_capital=15000"
```
Expected: Top-10 trades with full metrics, score decomposition, diagnostics, and cached Top-100 + orders JSON. A full sweep over broad chains can run up to ~1 hour; for a fast retest, reuse cached Top-10 structures:
```bash
qse optimize-strategy --ticker NVDA \
  --regime strong-bullish \
  --trade-horizon 1 \
  --config config.yml \
  --retest top10.json
```
Retests should complete in <30 seconds.

## Interactive tuning (optional)
```bash
qse interactive
> set ticker NVDA
> set regime strong-bullish
> set trade-horizon 1
> set data-source schwab  # falls back to yfinance with warning if unavailable
> override scoring.w_theta 0.2
> run optimizer
> export order.json
```
Use for exploration; automation should stay with single-command mode.

## Monitor an open position
```bash
qse monitor --position my_trade.json --interval 300
```
Expected: periodic repricing, updated POP/E[PnL]/tail metrics for remaining horizon, alerts on profit_target/stop_loss, optional broker-ready exit order suggestion.

## Testing and validation
- Run unit/integration tests: `cd src && pytest`
- Lint: `cd src && ruff check .`
- Reference test (once available): run synthetic chain test case to ensure scorer stability and regression safety.

## Artifacts
- Outputs: Top-10 to stdout/JSON, Top-100 cache, `orders/` JSON, diagnostics log with stage counts and rejection breakdown, CI for E[PnL]/POP.
