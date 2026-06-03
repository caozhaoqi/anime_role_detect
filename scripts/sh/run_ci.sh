#!/bin/bash
# 本地 CI/CD 测试脚本
# 模拟 GitHub Actions 工作流

set -e

echo "======================================"
echo "  本地 CI/CD 测试脚本"
echo "======================================"
echo ""

# 步骤1: 检查代码
echo "[步骤1/4] 检查代码"
echo "当前目录: $(pwd)"
echo "Git 状态:"
git status --porcelain | head -5
echo ""

# 步骤2: 设置 Python 环境
echo "[步骤2/4] 设置 Python 环境"
python3 --version
echo ""

# 步骤3: 依赖检查
echo "[步骤3/4] 依赖检查"
echo "检查核心依赖..."

# 检查关键包
check_package() {
    pkg=$1
    python3 -c "import $pkg; print(f'✓ {pkg}: {getattr($pkg, \"__version__\", \"OK\")}')" 2>/dev/null || echo "✗ $pkg: 未安装"
}

check_package torch
check_package fastapi
check_package onnxruntime
check_package celery
check_package redis
check_package prometheus_client
echo ""

# 步骤4: API 服务测试
echo "[步骤4/4] API 服务测试"

# 检查服务是否运行
if curl -s http://localhost:8000/api/v1/onnx/models > /dev/null 2>&1; then
    echo "✓ API 服务运行正常"
    
    # 测试 ONNX 模型列表
    echo ""
    echo "ONNX 可用模型:"
    curl -s http://localhost:8000/api/v1/onnx/models | python3 -m json.tool
    
    # 测试 Prometheus 监控
    echo ""
    echo "Prometheus 指标:"
    curl -s http://localhost:9090/metrics | grep requests_total | head -2
    
    # 测试健康检查
    echo ""
    echo "健康检查:"
    curl -s http://localhost:8000/api/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "健康检查端点未配置"
else
    echo "✗ API 服务未运行，请先启动服务"
    echo "  启动命令: python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"
fi

echo ""
echo "======================================"
echo "  CI/CD 测试完成"
echo "======================================"
