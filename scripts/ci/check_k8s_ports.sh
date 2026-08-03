#!/usr/bin/env bash
# check_k8s_ports.sh — check_k8s_ports.py 的薄包装入口
#
# 兼容入口：优先使用仓库本地 .venv 的 python3（本地开发），
# 否则回退到 PATH 中的 python3（CI runner，已 pip install pyyaml）。
# 用法: ./check_k8s_ports.sh [--k8s-dir <dir>]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="python3"
if [ -x "${SCRIPT_DIR}/../../.venv/bin/python3" ]; then
  PYTHON_BIN="${SCRIPT_DIR}/../../.venv/bin/python3"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/check_k8s_ports.py" "$@"
