/sc:improve --type quality --type performance "1) I am interested in making sure the extraction and embedding APIs work, so that they could potentially be MCP tools for the LLM part of the RAG pipeline.  So, we are close to MVP for that in my mind when we fix these file format issues.
2) I am mostly concerned with a robust document ingestion pipeline that runs on the KS-001 server using the ingestion service.  My understanding is that system does not use the API services at all, it will call the embedding and extraction services locally. So, getting the APIs working is good but we also have the parallel and perhaps more important goal of making the extraction and embedding service robust and fast. What can we do to make the current system more robust and faster? Am I correct in assuming that running extraction and embedding locally rather than through the API will be significantly faster? We want the src/api/ingestion.py service to be able to run batch processing on a que of files after they have been saved to the Cloudflare R2 object store.
3) There are *a lot* more than 6 failed and 4 unsupported documents from this test.  We removed the fallback extraction of the base64 binary, so we just raise an exception now.  But we still need to address why we are getting many, many "Extraction attempt 3/3 failed" for various document types. Could it be that we are exceeding any arbitrary user limits in our settings or .env?" 
  
/compact "Please remove unrelated discussion, earlier brainstorming, and obsolete ideas,
but keep all specifications, constraints, and interface definitions required for correct
implementation."

/speckit.tasks "Follow the workflow in '.codex/speckit.tasks.md'.  Develop the implementation as laid out in 'specs/009-option-optimizer/spec.md', plan.md, research.md, data-model.md.  Also review 'planning/Option_Spead_Candiate_Optimizer.md' for detailed instructions.  I want to be certain that we are designing to the real API interfaces that we will be using to retrive data.  Please examine at 'planning/yfinance_specs.md' and 'docs/*.pdf' and fetch any necessary support documentation.  Then find and review '*.py' in the repo where we use yfinance().  1) Determine and document any necessary API contracts.  2) Build the clients to retrieve the necessary data for feature branch '009-option-optimizer' as outlined in 'planning/Option_Spead_Candiate_Optimizer.md'.  We want to primarily use the Schwab API, but fallback with warnings to yfinance for anything that Schwab does not have or if there is an unexpected error.  Ensure that the logic, conventions, patterns, and interfaces stay in alignment with feature branch 'specs/001-mvp-pipeline/*.md'"

/speckit.implement "Be sure to follow the workflow in '.codex/speckit.implement.md'.  Implement incomplete tasks in priority order. Run the tests after each stage of development. Be sure to commit with a detailed message after each phase is completed.  Be sure to mark completed tasks in tasks.md."

/speckit.implement "Be sure to follow the workflow in '.codex/speckit.implement.md'.  Implement incomplete Phase 3 tasks in priority order. When you are finished a) Ask yourself, "Does this code meet the intent of 'planning/Option_Spead_Candiate_Optimizer.md'?"  b) Run the tests after each stage of development. c) Be sure to mark completed tasks in tasks.md.  d) Be sure to commit using your task summary as the message after each phase is completed."

/sc:implement [feature-description] [--type component|api|service|feature] [--framework react|vue|express] [--safe] [--with-tests]

/sc:analyze [target] [--focus quality|security|performance|architecture] [--depth quick|deep] [--format text|json|report]

/sc.improve [target] [--type quality|performance|maintainability|style] [--safe] [--interactive]

/sc.document  --type user-guide|api|technical --format markdown|html

/sc.spawn [complex-task] [--strategy sequential|parallel|adaptive] [--depth normal|deep]

/sc.recommend

/sc:implement 
"I agree with your implementation plan. We want to do full implementation for each phase.  Review planning/Option_Spead_Candiate_Optimizer.md' to be sure we are meeting the intent of the design for the option spread optimizer. a) Write tests for end-to-end validation against acceptance scenarios (Top-10 computation, runtime targets, diagnostics completeness). b) Run the tests after each stage of development. c) Be sure to mark completed tasks in tasks.md. d) Be sure to commit with a detailed message after each phase is completed.

Keep in mind our over-arching workflow for this feature branch: StrategyOptimizer at the CLI: setup config .yml files. Pick one stock and price action regime.  That's pretty much it.  it should be automatic from there - use schwab api to collect price history, fundamental, news, etc for the stock, model its pricing from the price history using distribution fit testing, auto filter and select option spread candidates based on the regime and stock pricing, generate option stochastics and greeks and pricing models, use vector math libraries to perform selection/optimization of spread strategies based on our custom scoring system (some logic needs to be in the scoring that adapts to regime), measure trading horizon and expiration pricing for each spread candidate, down-select to final top100 and top10 candidates, store top 100 and return top10 with performance diagnostics and optimization analysis summary (we can use LLM APIs for this if we want to!), provide plots for profit probability (I have examples of what this should look like.  For MVP we can just use images, but I will want to use Graphana, Streamlit, mplfinance, etc. for a UI that updates real time with streaming data from the Schwab API.)

# Phase 1: Implement StrategyOptimizer Pipeline 
1) The models in qse/distributions/, like student_t.py MUST all be used by StrategyOptimizer for underlying stock pricing and monte carlo simulation building.
1. qse/optimizers/strategy_optimizer.py (200-300 lines)
- Stage 0: Expiry selection (select 3-5 expiries with DTE ∈ [7,45])
- Stage 1: Strike filtering (moneyness + liquidity filters)
- Stage 2: Candidate generation (verticals, Iron Condors, straddles, strangles)
- Stage 3: Analytic prefilter (hard constraints, top-K selection)
- Stage 4: Full MC scoring (GARCH-t paths, Bjerksund-Stensland pricing)
2. qse/optimizers/candidate_generator.py (150-200 lines)
- Vertical spread generation with width limits
- Iron Condor generation
- Straddle/strangle generation
3. qse/optimizers/mc_engine.py (100-150 lines)
- Regime-driven path generation
- MC scoring loop with repricing
- Confidence interval computation
4. tests/e2e/test_optimizer_acceptance.py (200-300 lines)
- US1 scenarios 1-4 from spec.md
- US2 scenarios 1-5 from spec.md
- Runtime verification (<30s target)
- Diagnostics completeness validation
	
# Phase 2: Run Tests & Validate
- Execute E2E tests after each stage
- Verify against acceptance criteria from spec.md
- Ensure runtime targets met
Write integration tests exercising end-to-end grid CLI yet; run_meta/artifact integration for grid outputs is minimal (JSON only, no run_meta fields recorded). Distribution fitting in CLI uses a synthetic laplace fit (placeholder) rather than sourcing returns—good enough for scaffolding but not production-ready with data loading. Budget enforcement is warning-based (no hard timeout stop). If you need spec-level replay/provenance for grids, additional run_meta wiring is still pending.

# Phase 3: Documentation & Commits
- Update tasks.md marking T014-T018 complete
- Commit with detailed messages per phase
- Write a detailed user guide for how to use StrategyOptimizer at the CLI for the complete workflow: How and where to setup config .yml files. Pick one stock and price action regime.  That's pretty much it.  it should be automatic from there - use schwab api to collect price history, fundamental, news, etc for the stock, model its pricing from the price history using distribution fit testing, auto filter and select option spread candidates based on the regime and stock pricing, generate option stochastics and greeks and pricing models, use vector math libraries to perform selection/optimization of spread strategies based on our custom scoring system (some logic needs to be in the scoring that adapts to regime), measure trading horizon and expiration pricing for each spread candidate, down-select to final top100 and top10 candidates, store top 100 and return top10 with performance diagnostics and optimization analysis summary (we can use LLM APIs for this if we want to!), provide plots for profit probability (I have examples of what this should look like.  For MVP we can just use images, but I will want to use Graphana, Streamlit, mplfinance, etc. for a UI that updates real time with streaming data from the Schwab API.)

"