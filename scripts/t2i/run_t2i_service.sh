#!/usr/bin/env bash
# 启动 t2i_service（角色图像生成微服务）
# 注意：必须跑在 t2i-mac venv（含 torch 2.3.1 + diffusers 0.30.1），不能用主 .venv
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)"

# P3 修复（2026-08-20）：统一用 python3（与 supervisord.conf 一致）
VENV_PY="t2i-mac/bin/python3"
if [ ! -x "$VENV_PY" ]; then
  echo "✗ 未找到 t2i-mac venv，请先创建："
  echo "    python3 -m venv t2i-mac"
  echo "    t2i-mac/bin/pip install -r scripts/t2i/requirements-t2i-mac.txt"
  exit 1
fi

PORT="${1:-8100}"
echo "▶ 启动 t2i_service @ :$PORT (venv=t2i-mac)"
exec "$VENV_PY" src/services/t2i_service/app.py --port "$PORT"
