# Parallel Execution Helpers

This note complements `parallel-runbook.md` by providing a quick way to spin up tmux panes for [P] tasks in `specs/009-option-optimizer/tasks.md`.

## Quickstart: tmux panes per task
Use `scripts/parallel_tmux.sh` to create a tmux session with one pane per task ID. Each pane prints log guidance (write to `logs/parallel/<task>.log`) and keeps the runbook visible in the prompt header.

```bash
scripts/parallel_tmux.sh T004 T005 T006
# or choose a custom session name
scripts/parallel_tmux.sh -s opt-parallel T009 T012
```

### Script notes
- Requires `tmux` installed locally.
- Creates a session named `qse_parallel` by default.
- Splits a single window into tiled panes, one per task ID.
- Leaves panes ready for you to run targeted commands (e.g., `pytest -q ...`) while keeping logs separate.

## Manual checklist (matches `parallel-runbook.md`)
If you prefer not to use tmux, follow the same checklist manually:
1. Open separate shells (or tabs) per task ID listed in the runbook scope.
2. Set `PYTHONPATH=.` when running targeted tests.
3. Keep logs in `logs/parallel/<task-id>.log` and avoid editing the same file concurrently.
4. Respect HTTP rate limits when hitting external APIs.
5. Record completion in `specs/009-option-optimizer/tasks.md`; log blockers in `logs/parallel/blockers.md`.
