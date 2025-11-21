# Quant Scenario Engine (MVP Pipeline)

CPU-only quant scenario engine for backtesting and Monte Carlo workflows across stock and option strategies. Implements heavy-tailed return models, deterministic seeding, resource-aware storage policy, and Typer-based CLI.

**Currently Implemented Commands:** `fetch`, `compare`
**Planned Commands:** `grid`, `screen`, `conditional`, `replay`

## Environment
- Python 3.10+ on Linux (8 vCPU / 24 GB RAM target)
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install runtime deps: `pip install -e . --quiet && echo "Installation complete"`
- Dev tools: `pip install -r requirements-dev.txt`

## Quickstart
1) **Data prep (Parquet)**
   ```bash
   python -m qse.cli.main fetch --symbol AAPL --start 2018-01-01 --end 2024-12-31 --interval 1d --target data/
   ```
   Layout: `data/historical/interval=1d/symbol=AAPL/_v1/*.parquet`, features under `data/features/interval=1d/symbol=AAPL/`.

2) **Run baseline stock vs option MC comparison**
   ```bash
   python -m qse.cli.compare \
     --symbol AAPL --paths 1000 --steps 60 \
     --distribution laplace --strategy stock_basic \
     --option-strategy call_basic --iv 0.25 --seed 42
   ```
   Compares holding stock vs buying call option. Outputs under `runs/<run_id>/` (metrics.json/csv, run_meta.json, logs/).

---

### Planned Features (Not Yet Implemented)

<details>
<summary>Click to expand planned command examples</summary>

3) **Parameter grid exploration**
   ```bash
   python -m qse.cli.main grid --config configs/grid_aapl.yaml
   ```

4) **Candidate screening + conditional runs**
   ```bash
   python -m qse.cli.main screen --universe configs/universe.yaml --selector configs/selector_gap.yaml --lookback 5y --top 20
   python -m qse.cli.main conditional --symbol AAPL --selector configs/selector_gap.yaml --paths 1000 --steps 60 --seed 99
   ```

5) **Replay**
   ```bash
   python -m qse.cli.main compare --replay runs/<run_id>/run_meta.json
   ```

</details>

## Configuration files (US7)
You can drive all commands with a YAML/JSON config and overrides.

**Precedence:** CLI flags > environment vars (`QSE_` prefix) > YAML/JSON > built-ins. Overrides are logged.

**Example YAML (`configs/run_aapl.yaml`):**
```yaml
symbol: AAPL
data_source: yfinance        # or schwab_stub
distribution: student_t      # laplace | student_t | normal | garch_t
pricer: black_scholes        # black_scholes | py_vollib | quantlib(stub)
mc:
  paths: 1500
  steps: 90
  seed: 123
```

Use with any CLI that supports `--config` (compare, grid, screen, conditional, replay):
```bash
python -m qse.cli.main compare --config configs/run_aapl.yaml --paths 2000  # CLI overrides YAML
```
Environment overrides (examples):
```bash
export QSE_SYMBOL=MSFT
export QSE_MC_PATHS=500
python -m qse.cli.main compare --config configs/run_aapl.yaml
```

## Testing
- Run all tests with coverage: `pytest tests/ --cov=backtesting --cov-report=term-missing`
- Unit tests: `pytest tests/unit -v`
- Integration tests: `pytest tests/integration -v`
- Contract tests: `pytest tests/contract -v`

## Performance guardrails
- MC stays in-memory when estimated footprint <25% RAM; memmap/npz fallback ≥25%; abort ≥50%.
- Max workers defaults to 6 on 8-core VPS; configurable via CLI/config.
- First run may be slower due to numba/arch compilation.

## Assumptions
- Parquet data pre-downloaded; drift triggers new partition suffix.
- Deterministic seeds captured in artifacts alongside package versions and system info.
