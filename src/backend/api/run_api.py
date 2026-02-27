#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API服务启动脚本
"""

import os
import sys
import argparse
import logging
import subprocess
import time
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger('run_api')


def setup_logging(verbose: bool = False):
    """
    设置日志
    
    Args:
        verbose: 是否启用详细日志
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def check_dependencies():
    """
    检查依赖项
    """
    required_packages = ['fastapi', 'uvicorn', 'python-multipart']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"缺少依赖项: {missing_packages}")
        logger.info("正在安装依赖项...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                *missing_packages
            ])
            logger.info("依赖项安装成功")
        except subprocess.CalledProcessError as e:
            logger.error(f"依赖项安装失败: {e}")
            return False
    
    return True


def start_api_service(host: str = '0.0.0.0', port: int = 8000, reload: bool = False):
    """
    启动API服务
    
    Args:
        host: 主机地址
        port: 端口号
        reload: 是否启用热重载
    """
    logger.info(f"启动API服务: http://{host}:{port}")
    
    # 构建命令
    cmd = [
        sys.executable,
        '-m', 'uvicorn',
        'src.api.app:app',
        '--host', host,
        '--port', str(port)
    ]
    
    if reload:
        cmd.append('--reload')
    
    # 启动服务
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("API服务已停止")
    except subprocess.CalledProcessError as e:
        logger.error(f"API服务启动失败: {e}")


def check_api_status(host: str = 'localhost', port: int = 8000, timeout: int = 30) -> bool:
    """
    检查API服务状态
    
    Args:
        host: 主机地址
        port: 端口号
        timeout: 超时时间（秒）
        
    Returns:
        bool: 服务是否正常运行
    """
    import requests
    
    url = f"http://{host}:{port}/api/health"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                logger.info(f"API服务状态正常: {response.json()}")
                return True
        except requests.RequestException:
            pass
        
        time.sleep(2)
    
    logger.error(f"API服务启动超时 ({timeout}秒)")
    return False


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='API服务启动脚本')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=8000, help='端口号')
    parser.add_argument('--reload', action='store_true', help='启用热重载')
    parser.add_argument('--check', action='store_true', help='检查服务状态')
    parser.add_argument('--verbose', action='store_true', help='启用详细日志')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 检查依赖项
    if not check_dependencies():
        logger.error("依赖项检查失败，退出")
        return 1
    
    if args.check:
        # 检查服务状态
        status = check_api_status(args.host, args.port)
        return 0 if status else 1
    else:
        # 启动服务
        start_api_service(args.host, args.port, args.reload)
        return 0


if __name__ == '__main__':
    sys.exit(main())
