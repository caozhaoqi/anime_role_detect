#!/bin/bash

# 跨平台启动脚本
# 用于启动API服务，包含环境配置和诊断功能

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "项目根目录: $PROJECT_ROOT"
echo "切换到项目根目录..."
cd "$PROJECT_ROOT"

# 加载环境配置
if [ -f "$SCRIPT_DIR/setup_env.sh" ]; then
    echo "加载环境配置..."
    source "$SCRIPT_DIR/setup_env.sh"
else
    echo "警告: 环境配置脚本不存在，使用默认配置"
fi

# 创建必要的目录
mkdir -p logs
mkdir -p temp

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"

# 检查是否安装了必要的依赖
echo "检查依赖..."
python3 -c "import fastapi; import uvicorn; import torch; import loguru" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的依赖，请运行: pip install -r requirements.txt"
    exit 1
fi

# 检查设备类型
python3 -c "import torch; device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'); print(f'检测到设备: {device}')"

# 启动API服务
echo "启动API服务..."
echo "日志文件: logs/api.log"
echo "崩溃日志: logs/crash.log"
echo "诊断日志: logs/diagnostics.log"

# 使用uvicorn启动服务
python3 -m uvicorn src.backend.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info \
    --access-log \
    --timeout-keep-alive 60 \
    --limit-concurrency 100 \
    --backlog 2048

echo "API服务已停止"