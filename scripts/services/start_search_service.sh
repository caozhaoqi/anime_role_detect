#!/bin/bash
# 启动搜索服务的包装脚本 - 解决macOS上的Mutex锁问题

# 设置项目根目录
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

# 设置环境变量
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES"
export PYTORCH_ENABLE_MPS_FALLBACK="1"
export MPS_HIGH_WATERMARK_RATIO="0.0"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.0"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export VECLIB_MAXIMUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export KMP_DUPLICATE_LIB_OK="TRUE"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 禁用加速
export ACCELERATE_DISABLED="1"

# 使用Python 3的绝对路径（根据系统情况调整）
PYTHON_BIN="/Library/Developer/CommandLineTools/usr/bin/python3"

echo "启动搜索服务..."
echo "项目根目录: $PROJECT_ROOT"
echo "Python路径: $PYTHON_BIN"

# 启动服务
exec "$PYTHON_BIN" src/services/search_service/search_service_app.py