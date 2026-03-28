#!/bin/bash

# 跨平台环境配置脚本
# 用于设置Python崩溃追踪和内存管理环境变量

echo "配置跨平台环境变量..."

# 开启 Python 内置崩溃堆栈打印 (跨平台通用)
export PYTHONFAULTHANDLER=1

# 强制实时刷新日志，防止崩溃时缓冲区日志丢失
export PYTHONUNBUFFERED=1

# 针对 Mac MPS 的优化：当显存占用达到 80% 时强制回收
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8

# 设置日志级别
export LOG_LEVEL=INFO

# 设置模型缓存大小
export CACHE_SIZE=1000

# 设置API超时时间
export API_TIMEOUT=30

# 设置最大工作线程数
export MAX_WORKERS=4

# 设置批量处理的最大文件数
export MAX_BATCH_FILES=10

# 启用内存监控
export ENABLE_MEMORY_MONITOR=true

# 设置内存警告阈值（百分比）
export MEMORY_WARNING_THRESHOLD=85

# 设置内存紧急阈值（百分比）
export MEMORY_CRITICAL_THRESHOLD=95

# 设置GPU内存警告阈值（百分比）
export GPU_MEMORY_WARNING_THRESHOLD=85

# 设置GPU内存紧急阈值（百分比）
export GPU_MEMORY_CRITICAL_THRESHOLD=95

# 启用诊断日志
export ENABLE_DIAGNOSTICS=true

# 设置诊断日志文件路径
export DIAGNOSTICS_LOG_FILE=logs/diagnostics.log

# 设置崩溃日志文件路径
export CRASH_LOG_FILE=logs/crash.log

# 设置性能监控日志文件路径
export PERFORMANCE_LOG_FILE=logs/performance.log

echo "环境变量配置完成！"
echo "PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER"
echo "PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
echo "PYTORCH_MPS_HIGH_WATERMARK_RATIO=$PYTORCH_MPS_HIGH_WATERMARK_RATIO"
echo "LOG_LEVEL=$LOG_LEVEL"
echo "ENABLE_MEMORY_MONITOR=$ENABLE_MEMORY_MONITOR"
echo "ENABLE_DIAGNOSTICS=$ENABLE_DIAGNOSTICS"