# Quickstart

Parent: `plan.md` (derived from `spec.md` per constitution). References: `research.md` decisions, `data-model.md` entities, `contracts/` schemas for CLI/config validation.

1) **Environment**
- Python 3.11 on Linux VPS (8 vCPU, 24 GB RAM).
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install runtime deps (draft): `pip install numpy pandas scipy numba statsmodels arch pandas-ta quantstats plotly typer[all] yfinance py_vollib`.
- First-run build notes: ensure Python dev headers (`python3-dev`) and build tools (`build-essential`) are installed so `numba/arch` compile cleanly.

2) **Data prep & Parquet partitioning**
- Generate partitions: `python -m quant-scenario-engine.cli.fetch --symbol AAPL --start 2018-01-01 --end 2024-12-31 --interval 1d --target data/`
- Layout: `data/historical/interval=1d/symbol=AAPL/_v1/*.parquet` (version suffix increments when source changes).
- Feature files live separately under `data/features/interval=1d/symbol=AAPL/`.

3) **Run baseline stock vs option MC comparison**
- `python -m quant-scenario-engine.cli.compare --symbol AAPL --paths 1000 --steps 60 --distribution laplace --strategy stock_basic --option-strategy call_basic --iv 0.25 --seed 42`
- Outputs (per run_id):
  - `runs/<run_id>/metrics.json` and `metrics.csv`
  - `runs/<run_id>/run_meta.json`
  - `runs/<run_id>/logs/` and optional plots

4) **Parameter grid exploration (with example config)**
- Example `configs/grid_aapl.yaml`:
  ```yaml
  symbol: AAPL
  start: 2018-01-01
  end: 2024-12-31
  distribution: laplace
  paths: 1000
  steps: 60
  seed: 123
  max_workers: 6
  grid:
    - name: stock_basic
      kind: stock
      grid:
        threshold: [0.005, 0.01, 0.02]
    - name: call_basic
      kind: option
      grid:
        dte: [30, 60]
        strike_offset: ["atm", "+0.05"]
      shared:
        option_spec:
          option_type: call
          maturity_days: 60
          implied_vol: 0.25
          risk_free_rate: 0.05
          contracts: 1
  ```
- Run: `python -m quant-scenario-engine.cli.grid --config configs/grid_aapl.yaml`

5) **Candidate screening + conditional backtesting/MC**
- `python -m quant-scenario-engine.cli.screen --universe configs/universe.yaml --selector configs/selector_gap.yaml --lookback 5y --top 20`
- `python -m quant-scenario-engine.cli.conditional --symbol AAPL --selector configs/selector_gap.yaml --paths 1000 --steps 60 --seed 99`
- Screening outputs `runs/<run_id>/candidates.json`; conditional run backtests only on those episodes and reports stock vs option metrics.

6) **Replay a run**
- `python -m quant-scenario-engine.cli.compare --replay runs/<run_id>/run_meta.json`
- Refuses replay when data version changed unless `--force-replay`; drift status recorded in output.

7) **Resource safeguards**
- MC runs stay in-memory when estimated footprint <25% RAM; otherwise memmap/npz chosen automatically with warnings.
- Workers capped at 6 on 8-core VPS; adjustable via config but guarded by FR-018 estimator.
- First invocation of numba-jitted code may take longer due to compilation; subsequent runs are faster.

8) **Package assumptions**
- Commands assume `quant-scenario-engine` import path is installed (editable): `pip install -e .`.
- CLI names map to Typer entrypoints under `quant-scenario-engine/cli`.

9) **Common pitfalls**
- Missing build deps for `numba`/`arch`: install `python3-dev`, `build-essential`, `libopenblas-dev`.
- Parquet schema drift: bump `_v#` suffix when source changes and keep older files for reproducibility.
- Selector generating too few episodes: command will warn and fall back to unconditional MC unless `--no-fallback` is set.

10) **First-time setup helper (optional)**
- Script sketch:
  ```bash
  python3 -m venv .venv \
    && source .venv/bin/activate \
    && pip install -U pip \
    && pip install -r requirements-dev.txt \
    && python -m quant-scenario-engine.cli.fetch --symbol AAPL --start 2018-01-01 --end 2024-12-31 --interval 1d --target data/
  ```
