#!/usr/bin/env bash
# Returns project-specific API key based on current directory
# Keys are stored securely in ~/.anthropic/

PROJECT_DIR="$(pwd)"

case "$PROJECT_DIR" in
  */website|*/website-worktrees/*|*/backend|*/backend-worktrees/*)
    # KnowledgeSight Gateway project
    if [ -f ~/.anthropic/website-api-key.txt ]; then
      API_KEY=$(cat ~/.anthropic/website-api-key.txt)
      printf "ANTHROPIC_API_KEY=%s" "$API_KEY"
    else
      echo "Error: ~/.anthropic/website-api-key.txt not found" >&2
      exit 1
    fi
    ;;
  *)
    # Default key for other projects
    if [ -f ~/.anthropic/default-api-key.txt ]; then
      API_KEY=$(cat ~/.anthropic/default-api-key.txt)
      printf "ANTHROPIC_API_KEY=%s" "$API_KEY"
    else
      echo "Error: ~/.anthropic/default-api-key.txt not found" >&2
      exit 1
    fi
    ;;
esac
