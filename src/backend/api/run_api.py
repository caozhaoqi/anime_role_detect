#!/usr/bin/env python3
"""API服务启动脚本"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.api.app import app
import uvicorn

if __name__ == "__main__":
    # 按照文档配置，在8000端口启动
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
