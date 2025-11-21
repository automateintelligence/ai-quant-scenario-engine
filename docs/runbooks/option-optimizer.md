# Option Strategy Optimizer Runbook

## Runtime Modes
- **Full sweep (mode a)**: Broad candidate search may run up to 1 hour (FR-061). Use during pre/post-market with after-hours data.
- **Retest (mode b)**: Re-evaluate cached Top-10 with refreshed market data in <30 seconds (FR-061). Use the `--retest top10.json` flow.
- **Batch retest**: 10 cached Top-10 lists in ~5 minutes (<30s per ticker) for SC-012. Schedule full sweeps separately.

## Data Sources
- Default: `--data-source schwab` (Trader API). Emits warning and falls back to yfinance when unavailable.
- Capture provider provenance in logs and results; avoid parallel hammering of Schwab (see docs/data-sources/parallel-runbook.md).

## Commands
- Full sweep example:
```bash
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 3 \
  --config config.yml --data-source schwab --override "mc.num_paths=5000"
```
- Retest cached list:
```bash
qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1 \
  --config config.yml --data-source schwab --retest top10.json
```
- Monitor position:
```bash
qse monitor --position my_trade.json --interval 300 --data-source schwab
```

## Performance Tips
- Use default 5k paths; adaptive CI will raise paths only when CI is wide (FR-032/FR-033).
- Limit expiries/structures via config for faster sweeps; retest mode skips Stage 0–3.
- Parallelize [P] tasks judiciously; avoid overlapping file edits.

## Verification
- Tests: `pytest -q tests/unit/optimizers` (targeted); full suite may take longer.
- Lint: `ruff check .`
- Artifacts: save Top-10/Top-100, diagnostics (stage counts, rejection breakdown), and orders JSON.
