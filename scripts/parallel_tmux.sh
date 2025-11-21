#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SESSION_NAME="qse_parallel"

print_usage() {
  cat <<USAGE
Usage: ${0##*/} [-s session-name] TASK_ID [TASK_ID...]

Spawn a tmux session with one pane per task ID to coordinate work from
specs/009-option-optimizer/tasks.md. Each pane is pre-labeled and prints
log guidance from docs/data-sources/parallel-runbook.md.

Options:
  -s SESSION   Override the tmux session name (default: ${DEFAULT_SESSION_NAME}).
  -h           Show this help text.

Examples:
  ${0##*/} T004 T005 T006
  ${0##*/} -s opt-parallel T009 T012
USAGE
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but not installed. Install tmux and retry." >&2
    exit 1
  fi
}

create_session() {
  local session_name=$1
  local first_task=$2

  tmux new-session -d -s "$session_name" -n parallel "\
    printf 'Task %s\n' '$first_task'; \
    printf 'Log to logs/parallel/%s.log and follow docs/data-sources/parallel-runbook.md\n' '$first_task'; \
    exec bash"
}

add_pane() {
  local session_name=$1
  local task_id=$2

  tmux split-window -t "${session_name}:parallel" -v "\
    printf 'Task %s\n' '$task_id'; \
    printf 'Log to logs/parallel/%s.log and follow docs/data-sources/parallel-runbook.md\n' '$task_id'; \
    exec bash"
}

main() {
  local session_name=$DEFAULT_SESSION_NAME
  local task_ids=()

  while getopts ":s:h" opt; do
    case $opt in
      s) session_name=$OPTARG ;;
      h)
        print_usage
        exit 0
        ;;
      \?)
        echo "Unknown option: -$OPTARG" >&2
        print_usage
        exit 1
        ;;
    esac
  done
  shift $((OPTIND - 1))

  if [ "$#" -eq 0 ]; then
    echo "At least one TASK_ID is required." >&2
    print_usage
    exit 1
  fi

  require_tmux

  while [ "$#" -gt 0 ]; do
    task_ids+=("$1")
    shift
  done

  create_session "$session_name" "${task_ids[0]}"

  local idx=1
  local total=${#task_ids[@]}
  while [ $idx -lt $total ]; do
    add_pane "$session_name" "${task_ids[$idx]}"
    idx=$((idx + 1))
  done

  tmux select-layout -t "${session_name}:parallel" tiled
  tmux select-pane -t "${session_name}:parallel" -L >/dev/null 2>&1 || true
  tmux attach-session -t "$session_name"
}

main "$@"
