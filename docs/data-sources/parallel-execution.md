# Parallel Execution Guide

This guide consolidates the runbook, tmux quickstart, and operator skill notes for running tasks marked `[P]` in `specs/009-option-optimizer/tasks.md`.

## Scope
- Primary focus: phase 1–2 data tasks (T002, T004, T005, T006, T007, T009, T010).
- Any later-phase task marked `[P]` can be parallelized once its dependencies are complete.

## Core Guidelines
1. Use separate shells or tmux panes per task group. Keep logs in `logs/parallel/<task-id>.log`.
2. Respect HTTP rate limits for external APIs (e.g., Schwab docs parsing) to avoid throttling.
3. For Python tasks, set `PYTHONPATH=.` and prefer targeted runs such as `pytest -q` to limit contention.
4. Avoid editing the same file concurrently; serialize commits if overlaps are unavoidable.
5. Record completion in `specs/009-option-optimizer/tasks.md` (`[X]`) and log blockers in `logs/parallel/blockers.md`.

## Quickstart: tmux panes per task
Use the tmux launcher to create one pane per task ID. Each pane is pre-labeled with logging guidance and a recommended command.

```bash
scripts/parallel_tmux.sh T004 T005 T006
# or choose a custom session name
scripts/parallel_tmux.sh -s opt-parallel T009 T012
```

### Script behavior
- Requires `tmux` installed locally.
- Creates a session named `qse_parallel` by default.
- Tiled panes share a single window; each pane shows the log path and recommended command but does not auto-run it.
- Logs append to `logs/parallel/<task>.log`; panes export `PYTHONPATH=.` for Python invocations.

## Operator workflow (skill)
1. Identify candidates: scan `specs/009-option-optimizer/tasks.md` for unchecked `[P]` entries and confirm dependencies are satisfied.
2. Group tasks to avoid file contention; separate HTTP-heavy tasks (e.g., T004) from CPU-bound tasks (e.g., T002).
3. Launch tmux panes with the task IDs using the quickstart commands above.
4. In each pane, run the recommended command (or adjust as needed) and monitor logs in `logs/parallel/<task-id>.log`.
5. When tasks finish, mark them in the task list and capture any blockers in `logs/parallel/blockers.md`.

## Reference commands
- Targeted tests: `pytest tests/unit/data/test_yfinance.py -q`
- Contract tests (when available): `pytest tests/contract/test_optimize_contract.py -q`
- Lint (serialize if touching the same code): `ruff check .`
