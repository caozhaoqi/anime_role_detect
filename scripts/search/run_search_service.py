#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动搜索服务的包装脚本
使用subprocess确保环境变量正确设置
"""

import os
import sys
import subprocess
import time

def main():
    # 设置环境变量
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    env['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    env['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['VECLIB_MAXIMUM_THREADS'] = '1'
    env['NUMEXPR_NUM_THREADS'] = '1'
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 设置PYTHONPATH
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    
    # 构建命令
    cmd = [
        sys.executable,
        '-c',
        '''
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.services.search_service.app import app
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
'''
    ]
    
    print(f"启动搜索服务...")
    print(f"项目根目录: {project_root}")
    
    try:
        process = subprocess.Popen(cmd, env=env, cwd=project_root)
        print(f"服务已启动，进程ID: {process.pid}")
        
        # 等待服务启动
        time.sleep(5)
        
        # 检查服务状态
        import requests
        try:
            response = requests.get("http://localhost:8001/api/health")
            if response.status_code == 200:
                print(f"✅ 服务启动成功: {response.json()}")
            else:
                print(f"⚠️ 服务响应异常: {response.status_code}")
        except Exception as e:
            print(f"⚠️ 无法连接到服务: {e}")
        
        # 等待进程结束
        process.wait()
        
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动服务失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
