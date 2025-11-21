# Skill: Parallel Task Orchestration

**Trigger:** When the user asks to "execute parallel tasks" or "run [P] tasks".

**Context:** Refer to `parallel-runbook.md` for guidelines on logging and rate limits.
Refer to `scripts/parallel-launch.sh` for the execution mechanism.

**Workflow:**

1.  **Identify Candidates:**
    * Scan `specs/009-option-optimizer/tasks.md` for tasks marked `[P]` that are not yet checked `[X]`.
    * Check dependencies (ensure preceding tasks are complete).

2.  **Group Construction:**
    * Group tasks that touch different files to avoid lock contention (Guideline #4).
    * Separate HTTP-heavy tasks (T004) from CPU-heavy tasks (T002).

3.  **Execution Command:**
    * Generate the command to spawn the environment.
    * *Format:* `./scripts/parallel-launch.sh <ID_1> <ID_2> <ID_3>`

4.  **Monitoring & Handoff:**
    * Remind the user to check `logs/parallel/<ID>.log`.
    * Upon success, instruct user to mark `[X]` in the task list.

**Example Output:**
"I have identified 3 parallel-safe tasks (T002, T006, T010). 
Run the following to spawn the tmux environment with logging pre-configured:
`./scripts/parallel-launch.sh T002 T006 T010`"
