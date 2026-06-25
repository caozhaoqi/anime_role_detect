#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="$PROJECT_DIR"

# 优先使用系统 Python（Docker 容器），fallback 到虚拟环境（本地开发）
if command -v python3 &>/dev/null; then
    exec python3 src/services/search_service/search_worker.py
elif [ -f "$PROJECT_DIR/.venv/bin/python3" ]; then
    exec "$PROJECT_DIR/.venv/bin/python3" src/services/search_service/search_worker.py
else
    echo "ERROR: python3 not found" >&2
    exit 1
fi