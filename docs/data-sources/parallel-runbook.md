# Parallel Execution Runbook for [P] Tasks

Purpose: Coordinate parallel agent/cloud execution for tasks marked [P] in `specs/009-option-optimizer/tasks.md`, focusing on data provider setup, contract analysis, and test suites.

## Scope of Parallelization
- Phase 1-2 data tasks: T002, T004, T005, T006, T007, T009, T010.
- Phase 4+: [P]-marked tasks can run concurrently once dependencies in tasks.md are satisfied.

## Execution Guidelines
1) Use separate shells/tmux panes per task group. Keep logs in `logs/parallel/<task-id>.log`.
2) For HTTP-dependent tasks (Schwab docs parsing, fallback implementations), avoid overwhelming external APIs; respect rate limits noted in Schwab contracts.
3) For Python test jobs, pin `PYTHONPATH=.` and use `pytest -q` for targeted modules to reduce contention.
4) Coordinate writes: do not edit the same file concurrently. If overlap is unavoidable, serialize commits.
5) Cloud/off-host runners: mirror repo and Python 3.11 env; return patches via PR or `git format-patch` to avoid merge conflicts.

## Minimal Command Patterns
- Targeted tests: `pytest tests/unit/data/test_yfinance.py -q`
- Contract tests (when added): `pytest tests/contract/test_optimize_contract.py -q`
- Lint (serialize if touching same code): `ruff check .`

## Handoff
- Record completion in `specs/009-option-optimizer/tasks.md` by marking `[X]`.
- If contention or blockers occur, log in `logs/parallel/blockers.md` with task ID and owner.
