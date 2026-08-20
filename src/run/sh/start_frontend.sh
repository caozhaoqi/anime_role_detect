#!/bin/bash
# frontend 启动包装脚本（supervisord 调用）
# 自动探测 node/npm 并前置到 PATH：兼容 nvm、~/.local/node、homebrew、系统包等安装位置。
# 修复背景：supervisord.conf frontend 段曾硬编码 PATH=/opt/homebrew/...（mac 路径），
# 在 Ubuntu 服务器上会覆盖掉 node 所在路径导致 exit 127（2026-08-20）。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_DIR/src/frontend" || exit 1

# 前置常见 node 安装路径（已存在则跳过，避免重复）
for p in \
    "$HOME/.nvm/versions/node/"*/bin \
    "$HOME/.local/node/bin" \
    /opt/homebrew/bin \
    /usr/local/bin \
    /usr/bin; do
    if [ -d "$p" ]; then
        case ":$PATH:" in
            *":$p:"*) ;;
            *) export PATH="$p:$PATH" ;;
        esac
    fi
done

command -v node >/dev/null 2>&1 || { echo "ERROR: node not found in PATH" >&2; exit 127; }
command -v npm >/dev/null 2>&1 || { echo "ERROR: npm not found in PATH" >&2; exit 127; }

exec bash -c "npm run build && npm start"
