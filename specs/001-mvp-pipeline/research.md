# Research Findings (Phase 0 Summary)

Each decision below links to the corresponding requirements and captures performance/operational implications. Rejection criteria are explicit for traceability.

## Decision Log

1) **Option pricer backend for MVP and extension (FR-016)**  
**Decision:** Use closed-form Black–Scholes (scipy/numpy) for European calls/puts with fixed IV; define `OptionPricer` interface and ship optional `py_vollib` adapter; reserve QuantLib/Heston for a later adapter behind the same interface.  
**Rationale:** BS analytic keeps CPU-only VPS fast and avoids heavy builds; interface prevents lock-in and satisfies FR-016 configurability.  
**Alternatives considered:** QuantLib/Heston now (heavy compile, slows onboarding); custom Heston (complex, low ROI for MVP); BS-only without interface (blocks swap-ability).  
**Performance note:** BS pricing is O(n_steps); negligible vs MC generation. `py_vollib` adds minor overhead but stays CPU-friendly.

2) **Return distribution fitting approach (FR-002, FR-013)**  
**Decision:** Default Laplace fit via `scipy.stats.laplace`; enable Student-T via `scipy.stats.t`; optional GARCH-T via `arch` guarded by `use_garch` flag. Persist params + seeds in run_meta.  
**Rationale:** Laplace covers fat tails with minimal tuning; Student-T adds heavier tails for stress; GARCH-T available but off-by-default to preserve ≤10s baseline.  
**Alternatives considered:** Normal-only (insufficient tails), always-on GARCH (too slow for 1k×60), mixture models (complex, unjustified for MVP).  
**Performance note:** Laplace/Student-T fit O(n); GARCH fit slower (warn user when enabled; document expected latency per symbol ≈ seconds).

3) **Conditional candidate episode generation & conditional MC (FR-CAND-001..006)**  
**Decision:** Deterministic `CandidateSelector` DSL (gap/volume/volatility rules) emitting candidate episodes. Conditional MC: non-parametric bootstrapping from matched episodes + optional parametric refit (Laplace/Student-T); fallback to unconditional MC when pool < N_min.  
**Rationale:** Rule-based selectors are explainable and fast over ≥100 symbols; bootstrapping delivers scenario realism without heavy modeling; parametric refit keeps speed.  
**Alternatives considered:** kNN/state-embedding ML (more complexity/slower), pure unconditional MC (ignores conditioning goal), copulas (overkill for MVP).  
**Performance note:** Selector scan complexity O(symbols × bars); conditional bootstrap cost proportional to candidate pool size.

4) **Historical data sourcing & partitioning (FR-001, FR-017, DM-series)**  
**Decision:** yfinance default loader; Schwab adapter stub with identical interface; Parquet partitioned by `symbol=` and `interval=` with source version suffixes when upstream changes.  
**Rationale:** Fast to prototype; stable contract for future Schwab integration; partitioning accelerates slicing and supports DNFR storage expectations.  
**Alternatives considered:** CSV-only (slow, weak schema), Schwab-first (blocked by credential testing; slows MVP), centralized DB (unneeded for single-user VPS).  
**Performance note:** Parquet IO faster than CSV; partition pruning limits memory.

5) **Monte Carlo performance & storage policy (FR-013, FR-018, DM-008..DM-014)**  
**Decision:** In-memory MC when estimated footprint <25% RAM; chunked + `numpy.memmap` when ≥25%; persist `.npz` only for reuse/replay. Use vectorized NumPy + optional `numba` JIT; cap workers to ≤6 on 8 vCPU.  
**Rationale:** Prevents OOM, meets runtime/throughput goals, aligns with DM-series. Worker cap matches FR-018 limits.  
**Alternatives considered:** Always persist MC (disk-heavy), Dask/Ray (overhead for MVP), GPU (not available on VPS).

## MVP Defaults vs Advanced Toggles

| Area | Default (MVP) | Advanced toggle |
|------|----------------|-----------------|
| Distribution | Laplace | Student-T; GARCH-T via `use_garch` flag |
| Option pricer | Black–Scholes | `py_vollib`; QuantLib/Heston adapter later |
| Data source | yfinance | Schwab adapter promoted via config |
| MC storage | In-memory <25% RAM | memmap/npz when over threshold or replay requested |
| Selector | Built-in gap/volume rule | Custom DSL configs |
| Workers | max_workers=6 | Lower/raise via config within FR-018 guardrails |
