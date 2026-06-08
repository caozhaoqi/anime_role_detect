#!/bin/bash
# 设置环境变量以避免macOS多线程问题
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 运行Python脚本
python3 "$@"