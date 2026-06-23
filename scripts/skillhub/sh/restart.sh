#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

echo "🚀 ARD Skill Hub 重启脚本"
echo "=========================="

echo ""
echo "📁 项目目录: $PROJECT_ROOT"

echo ""
echo "⏹️  停止旧服务..."
pkill -f "uvicorn ardc.api.main:app" 2>/dev/null || true
sleep 2

echo ""
echo "🔍 检查端口占用..."
if lsof -i:8000 | grep -q LISTEN; then
    echo "⚠️  端口 8000 仍被占用，强制终止..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

echo ""
echo "🔧 切换到项目目录..."
cd "$PROJECT_ROOT"

echo ""
echo "🔧 激活虚拟环境..."
if [ -f "../../.venv/bin/activate" ]; then
    echo "  使用: ../../.venv"
    source ../../.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "  使用: .venv"
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "  使用: venv"
    source venv/bin/activate
else
    echo "⚠️  未找到虚拟环境，使用系统 Python"
fi

echo ""
echo "🔧 检查 uvicorn 是否安装..."
if ! command -v uvicorn &> /dev/null; then
    echo "❌ uvicorn 未安装，请先安装依赖: pip install uvicorn"
    exit 1
fi

echo ""
echo "🚀 启动服务..."
nohup uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4 > nohup.out 2>&1 &
sleep 3

echo ""
echo "🔍 验证服务..."
if curl -s http://localhost:8000/api/health | grep -q "healthy"; then
    echo "✅ 服务健康检查通过!"
    echo ""
    echo "📋 服务信息:"
    echo "   端口: 8000"
    echo "   进程ID: $(ps aux | grep "uvicorn ardc.api.main" | grep -v grep | awk '{print $2}')"
    echo "   日志: $PROJECT_ROOT/nohup.out"
else
    echo "❌ 服务启动失败!"
    echo "   请查看日志: tail -20 $PROJECT_ROOT/nohup.out"
    exit 1
fi

echo ""
echo "=========================="
echo "重启完成"