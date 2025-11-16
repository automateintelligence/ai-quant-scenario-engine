# Implementation Plan: Backtesting & Strategy Spec Authoring (Quant Scenario Engine)

**Branch**: `001-mvp-pipeline` | **Date**: 2025-11-16 | **Spec**: [/specs/001-mvp-pipeline/spec.md](/specs/001-mvp-pipeline/spec.md)
**Input**: Feature specification from `/specs/001-mvp-pipeline/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

MVP delivers a CPU-only Quant Scenario Engine that ingests historical OHLCV, fits heavy-tailed return models (Laplacian default; alternatives like Student-T), generates Monte Carlo price paths, and compares paired stock vs option strategies via a CLI. It must support parameter grids, conditional candidate episodes, and modular component swaps (data source, distribution, option pricer) while producing reproducible artifacts (run metadata, metrics, optional plots) on the `001-mvp-pipeline` branch.

## Technical Context

**Language/Version**: Python 3.11 (CPU-only VPS)  
**Primary Dependencies**: numpy, pandas, scipy, numba, statsmodels/arch (fat-tail fits), pandas-ta (indicators), quantstats/plotly for reporting, yfinance (default data), Schwab API stub, CLI via typer; advanced option pricer backend **NEEDS CLARIFICATION (QuantLib vs custom/Heston)**  
**Storage**: Parquet for canonical OHLCV/features; `.npz` or `numpy.memmap` for MC when exceeding RAM thresholds; run metadata as JSON; configs via YAML/CLI env  
**Testing**: pytest + hypothesis-style property tests for distributions; contract tests for CLI/API surfaces; coverage target ≥80% per constitution  
**Target Platform**: Linux VPS (8 vCPU, 24 GB RAM), no GPU  
**Project Type**: Single-package research/CLI toolchain  
**Performance Goals**: Baseline CLI (1k paths × 60 steps) ≤10s; grid up to ~50 configs ≤15m; MC memory <25% RAM in-memory or memmap fallback; reproducible seeded runs  
**Constraints**: CPU-only; avoid >50% RAM for MC arrays; robustness to missing data; deterministic seeds; must fail fast on invalid configs; option pricer accuracy vs speed trade-off **NEEDS CLARIFICATION**  
**Scale/Scope**: Single user; screen ≥100 symbols daily; 5–20 live symbols; up to thousands of MC paths per run; batch grids up to ~50 configs

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Specification-driven**: Plan/research/data-model/contracts/quickstart must align to spec.md; no implementation outside documented scope (Constitution II, XVIII).
- **Contracts & reproducibility**: Define explicit schemas for CLIs/configs/artifacts and guarantee seeded reruns + run metadata (Constitution VIII, XIX).
- **Testing discipline**: Commit to boundary tests (CLI + data adapters + pricers), contract tests, and ≥80% coverage; fail fast on violations (Constitution XV, V).
- **Simplicity & separation**: Prefer minimal vectorized pipeline; avoid speculative subsystems (e.g., live trading stack) until justified; keep component swaps via config (Constitution III, XVI).

## Project Structure

### Documentation (this feature)

```text
specs/001-mvp-pipeline/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
backtesting/
├── data/               # data source adapters (yfinance, Schwab stub, loaders)
├── features/           # indicator + macro enrichment
├── distributions/      # fit/sample abstractions for Normal/Laplace/Student-T/GARCH
├── mc/                 # Monte Carlo generators + memmap/npz persistence helpers
├── pricing/            # option pricers (Black-Scholes default; alt pricers TBD)
├── strategies/         # stock + option strategies, param schemas
├── simulation/         # core simulator, grid runner, conditional episode logic
└── cli/                # typer entrypoints for compare, grid, screening

tests/
├── unit/
├── integration/        # CLI + data/pricer wiring + persistence paths
└── contract/           # CLI/config/schema/contracts
```

**Structure Decision**: Single-package layout rooted in `backtesting/` with sibling `tests/`; keeps CLI + engine co-located for fast iteration while preserving clear submodules for data, MC, pricers, strategies, and simulations.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
