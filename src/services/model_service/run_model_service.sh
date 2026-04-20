#!/bin/bash

# 模型服务启动脚本

echo "当前工作目录: $(pwd)"
echo "脚本目录: $(dirname "$0")"
echo "计算的路径: $(dirname "$0")/../../../.."

# 切换到项目根目录
cd "$(dirname "$0")/../../../.."

echo "切换后的工作目录: $(pwd)"

# 安装依赖
echo "安装依赖..."
python3 -m pip install -r requirements.txt

# 设置环境变量
export MODEL_SERVICE_URL="http://localhost:8001"
export GPU_ENABLED="true"

# 启动模型服务
echo "启动模型服务..."
python3 src/services/model_service/app.py
