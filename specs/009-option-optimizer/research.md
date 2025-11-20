# Research Findings (Phase 0 Summary)

Parent: `plan.md` (per spec.md and constitution). Children: informs `data-model.md`, `contracts/`, `quickstart.md`.

Each decision captures rationale and rejected alternatives for traceability.

## Decision Log

1) **Regime-to-distribution mapping modes (FR-013..FR-016)**  
**Decision:** Support three modes: `table` (config lookup of mean_daily_return, daily_vol, skew, kurtosis_excess), `calibrated` (bootstrap historical bars matching regime features), and `explicit` (CLI overrides). Multi-day horizons compound daily params stepwise.  
**Rationale:** Table mode enables deterministic CLI runs; calibrated/explicit preserve expert overrides without changing code.  
**Alternatives considered:** Single mode (table only) — rejected; blocks calibration and explicit expert inputs.

2) **Pricing stack and fallback (FR-017..FR-023)**  
**Decision:** Default pricer Bjerksund-Stensland; automatic fallback to Black-Scholes on convergence error with warning; skip candidate if both fail; shared `OptionPricer` interface for US1/US9; IV evolution modes: constant, sticky-delta, custom callable.  
**Rationale:** Matches acceptance scenarios, preserves run continuity, and keeps pricing consistent across products.  
**Alternatives considered:** Halt on pricer failure (breaks run); BS-only (loses American exercise realism); ad-hoc IV handling (inconsistent).

3) **Candidate filtering and staging (FR-006..FR-012, FR-010)**  
**Decision:** Stage 0 choose 3–5 expiries in [7,45] DTE; Stage 1 strikes filtered by regime moneyness band and liquidity (vol/OI/bid-ask%); Stage 2 generate verticals width 1–3, iron condors, straddles/strangles; Stage 3 analytic prefilter with hard constraints (capital ≤15k, MaxLoss/capital ≤5%, E[PnL]≥$500, POP≥60%, POP_target≥30%) and keep top K per structure (20–50); Stage 4 full MC scoring on 50–200 survivors.  
**Rationale:** Bounded search space for <30s runtime while preserving structure variety.  
**Alternatives considered:** Brute-force full MC on all candidates (too slow); heuristic single-stage filtering (risk of missing good trades).

4) **Transaction costs and net-P&L handling (FR-042..FR-047)**  
**Decision:** Pay-the-spread fills (short legs at bid, long at ask) plus $0.65/contract commission; include expected exit spread in E[PnL]; apply all filters/scoring on net P&L after costs; allow `--commission 0` override for testing with spread costs retained unless also overridden.  
**Rationale:** Aligns with acceptance scenarios; prevents optimistic profitability.  
**Alternatives considered:** Mid-price fills (unrealistic); ignoring exit costs (overstates PnL).

5) **Output footprint and diagnostics (FR-048..FR-055, FR-075)**  
**Decision:** Return Top-10 to primary output; cache Top-100 with broker-ready JSON orders; include leg-by-leg breakdown, full metrics, score decomposition, and diagnostic summaries when empty or when pricer/MC fallbacks occur; always report 95% CIs for E[PnL]/POP; adaptive paths to max 20k when CI width exceeds threshold.  
**Rationale:** Meets success criteria for transparency and uncertainty quantification without exceeding runtime budgets.  
**Alternatives considered:** Point estimates only (too opaque); full-candidate dump (noisy, slower for users).
