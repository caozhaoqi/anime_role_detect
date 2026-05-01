#!/bin/bash
# ngrok 启动脚本
# 用于将本地服务暴露到公网

NGROK_PORT=5000
NGROK_CONFIG_DIR="/Users/caozhaoqi/Library/Application Support/ngrok/ngrok.yml"

echo "=========================================="
echo "  ngrok 内网穿透服务启动"
echo "=========================================="
echo "本地端口: $NGROK_PORT"
echo ""

# 检查 ngrok 是否安装
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok 未安装"
    echo "请先安装 ngrok: https://ngrok.com/download"
    exit 1
fi

# 检查 ngrok 配置目录
if [ ! -d "$NGROK_CONFIG_DIR" ]; then
    echo "创建 ngrok 配置目录..."
    mkdir -p "$NGROK_CONFIG_DIR"
fi

# 启动 ngrok
echo "🚀 启动 ngrok..."
ngrok http $NGROK_PORT --log=stdout

# 获取公网 URL（备用方法）
# ngrok http $NGROK_PORT 2>&1 | grep "Forwarding" &
# sleep 5
# curl http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url'