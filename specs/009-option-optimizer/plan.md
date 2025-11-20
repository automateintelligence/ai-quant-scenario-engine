# Option Strategy Optimizer (009-option-optimizer) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Branch**: `009-option-optimizer` | **Date**: 2025-11-20 | **Spec**: specs/009-option-optimizer/spec.md
**Input**: Feature specification from `/specs/009-option-optimizer/spec.md`

## Summary
Option strategy optimizer that ingests option chains and a qualitative regime, filters thousands of structures through staged screening (expiry/strike/liquidity → analytic prefilter → full Monte Carlo) and returns Top-10 ranked trades with diagnostics, live monitoring, and pluggable pricing/scoring. Architecture: config-driven regime→distribution mapping feeding PathGenerator + OptionPricer interface (Bjerksund default with BS fallback), staged candidate generator, composite StrategyScorer plugins, and CLI/daemon flows for optimize-strategy and monitor.

## Technical Context
**Language/Version**: Python 3.11
**Primary Dependencies**: numpy/scipy, pandas, typing; arch or similar for GARCH (optional); click/typer-style CLI; logging for diagnostics; potential py_vollib/QuantLib adapters future.
**Storage**: In-memory plus JSON/npz artifacts; no database.
**Testing**: pytest, ruff (per repo guidelines).
**Target Platform**: Linux CPU-only VPS.
**Project Type**: Single-project library/CLI.
**Performance Goals**: <30s runtime for Top-10 at market open with 5k paths baseline (FR-061); reduce 1k–10k candidates to 50–200 survivors; path cap 20k for adaptive CI tightening.
**Constraints**: CPU-only; max capital 15k; MaxLoss/capital ≤5%; configs shared with US1; deterministic seeds; pay-spread + $0.65/contract costs.
**Scale/Scope**: Single underlying per optimize run; batch mode sequential 10 tickers in <5 minutes (SC-012); cache Top-100 trades.

## Constitution Check
- Specification-driven workflow: spec.md present; plan/tasks to align (Constitution II, XIX).
- Quality gates: tests required (pytest/ruff), CI must fail on violations; confidence intervals + diagnostics (Section II.V, XIII, XV).
- Security/Privacy: CLI/local data only; no PII; zero external auth (Section II.VI, VII) — NA but documented.
- Observability: Structured diagnostics for stage counts, pricer fallbacks, MC variance, alerts (Section II.XIII, IX).
- Deployment safety: not deploying services; CLI artifacts + config versioning only.
- Gate status: PASS (no violations requiring complexity tracking).

## Project Structure

### Documentation (this feature)

```text
specs/009-option-optimizer/
├── plan.md              # This file (/speckit.plan output)
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/
└── qse/
    ├── cli/
    ├── distributions/
    ├── pricing/
    ├── selectors/
    ├── scorers/
    ├── optimizers/
    └── monitoring/

tests/
└── qse/
    ├── unit/
    └── integration/
```

**Structure Decision**: Single Python project with `src/qse` modules for pricing, distributions, candidate generation, scorers, CLI commands (`optimize-strategy`, `monitor`). Tests mirrored under `tests/qse` with unit + integration buckets.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
