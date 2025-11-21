#!/bin/bash
# scripts/parallel-launch.sh
# Usage: ./scripts/parallel-launch.sh T002 T004 T007

SESSION_NAME="optimizer-parallel"
LOG_DIR="logs/parallel"

# 1. Ensure Log Directory Exists
mkdir -p "$LOG_DIR"

# 2. Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed."
    exit 1
fi

# 3. Create or Attach to Session
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION_NAME -n "control"
    tmux send-keys -t $SESSION_NAME:control "echo 'Parallel Session Started. Switch windows to monitor tasks.'" C-m
fi

# 4. Task Definitions (Simple Registry based on Runbook)
get_task_command() {
    case $1 in
        T002) echo "pytest tests/unit/data/test_yfinance.py -q" ;;
        T004) echo "# Schwab Docs Parsing (HTTP dependent)" ;;
        T005) echo "# Fallback Implementation" ;;
        T006) echo "# Contract Analysis" ;;
        *)    echo "pytest -q # Generic test runner" ;;
    esac
}

# 5. Spawn Panes for each Task ID
for TASK_ID in "$@"; do
    LOG_FILE="${LOG_DIR}/${TASK_ID}.log"
    CMD=$(get_task_command $TASK_ID)
    
    # Create a new window for the task
    tmux new-window -t $SESSION_NAME -n "$TASK_ID"
    
    # Setup Environment (Guideline #3 & #1)
    tmux send-keys -t $SESSION_NAME:$TASK_ID "export PYTHONPATH=." C-m
    tmux send-keys -t $SESSION_NAME:$TASK_ID "echo '--- Starting $TASK_ID ---' | tee -a $LOG_FILE" C-m
    
    # Pre-type the command but don't execute (User/AI review) or Execute automatically
    # Here we let it wait for user confirmation to avoid accidental API limits
    tmux send-keys -t $SESSION_NAME:$TASK_ID "$CMD 2>&1 | tee -a $LOG_FILE"
done

# 6. Attach
tmux select-window -t $SESSION_NAME:1
tmux attach-session -t $SESSION_NAME
