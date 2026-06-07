#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗模块最终验证测试
使用subprocess隔离torch导入，避免Mac上的mutex问题
"""

import subprocess
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
test_script = project_root / "tests" / "pipeline" / "test_cleaner_mock.py"

print("=" * 60)
print("🧪 运行数据清洗模块验证测试")
print("=" * 60)

# 使用subprocess运行，避免当前进程的torch导入问题
env = os.environ.copy()
env["PYTHONPATH"] = str(project_root)

# 设置环境变量禁用CUDA
env["CUDA_VISIBLE_DEVICES"] = ""
env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

result = subprocess.run(
    [sys.executable, str(test_script)],
    capture_output=True,
    text=True,
    env=env
)

print(result.stdout)
if result.stderr:
    # 过滤掉警告
    stderr_lines = [l for l in result.stderr.split('\n') 
                   if 'Warning' not in l and 'warning' not in l]
    if stderr_lines:
        print("STDERR:", '\n'.join(stderr_lines[:5]))

print("=" * 60)
print(f"退出码: {result.returncode}")
print("=" * 60)

if result.returncode == 0:
    print("✅ 所有验证测试通过!")
else:
    print("❌ 测试失败")
    sys.exit(1)
