#!/bin/bash

# 模型服务启动脚本

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 安装依赖
echo "安装依赖..."
python3 -m pip install -r ../../requirements.txt

# 设置环境变量
export MODEL_SERVICE_URL="http://localhost:8001"
export GPU_ENABLED="true"

# 启动模型服务
echo "启动模型服务..."
python3 app.py
