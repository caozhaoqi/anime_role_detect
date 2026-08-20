#!/bin/bash
# t2i-service 启动包装脚本（supervisord 调用）
# INTERNAL_SERVICE_TOKEN 未设置时默认为空串 → 服务中间件退化为"不校验"（本机开发）；
# 生产环境在启动 supervisord 前 export INTERNAL_SERVICE_TOKEN=<值> 即启用内部鉴权。
# 修复背景：supervisord 的 %(ENV_x)s 展开要求变量必须存在，未设置会直接启动失败
# （2026-08-20）。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="$PROJECT_DIR"
: "${INTERNAL_SERVICE_TOKEN:=}"
export INTERNAL_SERVICE_TOKEN

exec "$PROJECT_DIR/t2i-mac/bin/python3" "$PROJECT_DIR/src/services/t2i_service/app.py" --port 8100
