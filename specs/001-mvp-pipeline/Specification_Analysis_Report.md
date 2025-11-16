Specification Analysis Report
ID	Category	Severity	Location(s)	Summary	Recommendation

C1	Coverage Gap	HIGH	spec.md:141 (FR-015); tasks.md (no mapping)	No task covers documenting/validating performance targets (FR-015).	Add a task to define/validate performance budgets (e.g., benchmarks + doc) and map it to FR-015/SC-001..003.
C2	Coverage Gap	MEDIUM	spec.md:158-214 (DM-001..DM-014); tasks.md:T019,T030,T031	DM-series largely uncovered: only directory layout/storage threshold tasks exist; no tasks for resolution tiering, Parquet compression/versioning, or historical vs feature separation.	Add tasks to enforce DM-001..007 (resolution tiers, partitions, compression/versioning) and DM-012..014 policies (persistence conditions, memmap thresholds) in data ingest/storage tooling.
T1	Terminology Drift	MEDIUM	quickstart.md:66-68; plan.md:49-63; tasks.md:29-35	Module naming drift: quickstart/plan refer to installing backtesting package, tasks use quant-scenario-engine namespace—risk of import/CLI confusion.	Standardize package name and CLI module path across docs/tasks; add a note or task to rename/mirror accordingly.
A1	Ambiguity	LOW	tasks.md:8-20	Tests noted as “not explicitly requested” yet constitution mandates ≥80% coverage; task list includes some test-related items but not organized by FR/SC.	Add explicit test tasks per user story/NFR (contract/integration/property tests) to satisfy constitution Principle XV and SC-coverage.
Coverage Summary Table (FR/FR-CAND/DM)
Requirement Key	Has Task?	Task IDs	Notes
FR-001	Yes	T021	
FR-002	Yes	T025,T026,T093	
FR-003	Yes	T038,T039	
FR-004	Yes	T038	
FR-005	Yes	T041,T045	
FR-006	Yes	T049,T099	
FR-007	Yes	T074	
FR-008	Yes	T018,T042,T043,T069,T120	
FR-009	Yes	T024,T034,T092	
FR-010	Yes	T023	
FR-011	Yes	T022	
FR-012	Yes	T032	
FR-013	Yes	T017,T030,T031	
FR-014	Yes	T102,T103	
FR-015	No	—	Missing task for performance targets
FR-016	Yes	T011,T033,T034,T094,T095	
FR-017	Yes	T022	
FR-018	Yes	T017,T075,T078	
FR-019	Yes	T018,T044,T109,T111-T117	
FR-CAND-001	Yes	T052	
FR-CAND-002	Yes	T054,T055,T063	
FR-CAND-003	Yes	T063-T067	
FR-CAND-004	Yes	T080-T087	
FR-CAND-005	Yes	T080-T082	
FR-CAND-006	Yes	T052,T070	
DM-001	No	—	Resolution tiering not tasked
DM-002	No	—	
DM-003	No	—	
DM-004	No	—	Parquet compression/versioning missing
DM-005	No	—	Raw vs feature separation enforcement missing
DM-006	Partial	T019	Path setup only
DM-007	No	—	
DM-008	Partial	T030	In-memory threshold only
DM-009	No	—	Repro guidance not tasked
DM-010	Yes	T031	npz persistence
DM-011	Yes	T031	memmap fallback
DM-012	No	—	Persist format policy beyond memmap not tasked
DM-013	No	—	Canonical Parquet retention/versioning missing
DM-014	No	—	MC treated as ephemeral not tasked
Constitution Alignment Issues: None direct, but missing performance and DM tasks jeopardize compliance with specification-driven and observability requirements.

Unmapped Tasks: (Representative) T001-T007 (setup scaffolding) not tied to specific requirements; consider tagging to applicable NFR/plan prerequisites.

Metrics:

Total Requirements (FR+FR-CAND+DM): 39
Total Tasks: 117 (per tasks.md)
Coverage % (requirements with ≥1 task): ~67% (26/39)
Ambiguity Count: 1
Duplication Count: 0
Critical Issues Count: 1
Next Actions
Address HIGH items before implementation: align directory/package naming (I1/T1) and add tasks for FR-015 and DM coverage (C1/C2).
After adjustments, rerun /speckit.analyze to verify coverage.