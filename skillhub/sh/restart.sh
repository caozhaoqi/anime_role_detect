#!/bin/bash
# ARD Skill Hub 重启脚本
# 停止旧服务并启动新服务

echo "🚀 ARD Skill Hub 重启脚本"
echo "=========================="

# 停止所有 uvicorn 进程
echo ""
echo "⏹️  停止旧服务..."
pkill -f "uvicorn ardc.api.main:app" || true
sleep 2

# 检查是否还有进程占用端口
echo ""
echo "🔍 检查端口占用..."
if lsof -i:8000 | grep -q LISTEN; then
    echo "⚠️  端口 8000 仍被占用，强制终止..."
    lsof -ti:8000 | xargs kill -9
    sleep 1
fi

# 进入项目目录
cd ~/czq/anime_role_detect/skillhub

# 激活虚拟环境
echo ""
echo "🔧 激活虚拟环境..."
source .venv/bin/activate

# 启动新服务
echo ""
echo "🚀 启动服务..."
nohup uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4 > /dev/null 2>&1 &
sleep 3

# 检查服务状态
echo ""
echo "✅ 服务已启动!"
echo ""

# 验证服务
echo "🔍 验证服务..."
if curl -s http://localhost:8000/api/health | grep -q "healthy"; then
    echo "✅ 服务健康检查通过!"
    echo ""
    echo "📋 服务信息:"
    echo "   端口: 8000"
    echo "   进程ID: $(ps aux | grep "uvicorn ardc.api.main" | grep -v grep | awk '{print $2}')"
    echo "   日志: nohup.out"
else
    echo "❌ 服务启动失败!"
    echo "   请查看日志: tail -20 nohup.out"
fi

echo ""
echo "=========================="
echo "重启完成"