#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/python3" src/services/search_service/search_worker.py