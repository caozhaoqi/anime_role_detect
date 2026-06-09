#!/bin/bash
# ARD Skill Hub 服务部署脚本
# 配置 systemd 服务管理

echo "🚀 ARD Skill Hub 服务部署脚本"
echo "=============================="

# 停止当前运行的服务
echo ""
echo "⏹️  停止现有服务..."
pkill -f "uvicorn ardc.api.main:app" 2>/dev/null || true
sleep 2

# 复制服务配置文件
echo ""
echo "📦 安装服务配置..."
cp conf/ardc-api.service /etc/systemd/system/

# 重新加载 systemd
echo ""
echo "🔄 重新加载 systemd..."
systemctl daemon-reload

# 启动服务
echo ""
echo "🚀 启动服务..."
systemctl start ardc-api

# 设置开机自启
echo ""
echo "🔧 设置开机自启..."
systemctl enable ardc-api

# 检查服务状态
echo ""
echo "✅ 服务部署完成!"
echo ""
echo "🔍 服务状态:"
systemctl status ardc-api --no-pager

# 等待服务启动
sleep 3

# 验证服务
echo ""
echo "🔍 验证服务..."
if curl -s http://localhost:8000/api/health | grep -q "healthy"; then
    echo "✅ 服务健康检查通过!"
else
    echo "❌ 服务启动失败!"
    echo "   查看日志: journalctl -u ardc-api -f"
fi

echo ""
echo "=============================="
echo "部署完成"
