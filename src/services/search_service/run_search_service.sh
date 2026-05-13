#!/bin/bash
# 启动图像搜索与视频识别服务

cd "$(dirname "$0")"

# 设置环境变量
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES"
export PYTORCH_ENABLE_MPS_FALLBACK="1"

# 启动服务
echo "启动图像搜索与视频识别服务..."
python3 app.py --host 0.0.0.0 --port 8001

echo "服务已停止"
