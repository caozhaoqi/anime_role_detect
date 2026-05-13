#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫角色识别系统 - 稳健启动脚本
参考 hcm-core 设计模式，支持：
- 服务依赖管理
- 启动重试机制
- 资源监控
- 优雅的启动顺序控制
"""

import os
import sys
import time
import subprocess
import signal
import psutil

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 直接读取配置文件内容
import importlib.util
config_path = os.path.join(project_root, "src/run/services_config.py")
spec = importlib.util.spec_from_file_location("services_config", config_path)
services_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(services_config)
SERVICES = services_config.SERVICES
SERVICE_GROUPS = services_config.SERVICE_GROUPS

# 服务启动配置
START_DELAY = 3  # 服务间启动延迟（秒）
MAX_RETRY = 3    # 最大重试次数
RETRY_DELAY = 5  # 重试间隔（秒）

processes = {}

def log_info(message):
    """日志输出"""
    print(f"[\033[94mINFO\033[0m] {message}")

def log_success(message):
    """成功日志"""
    print(f"[\033[92mSUCCESS\033[0m] {message}")

def log_error(message):
    """错误日志"""
    print(f"[\033[91mERROR\033[0m] {message}")

def log_warning(message):
    """警告日志"""
    print(f"[\033[93mWARNING\033[0m] {message}")

def check_memory_available():
    """检查可用内存"""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    if available_gb < 2:
        log_warning(f"可用内存不足: {available_gb:.1f} GB")
    return available_gb

def wait_for_port(port, timeout=30):
    """等待端口就绪"""
    import socket
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def start_service(service_name):
    """启动单个服务"""
    service = SERVICES.get(service_name)
    if not service:
        log_error(f"服务 {service_name} 不存在")
        return False
    
    port = service["port"]
    script = service["script"]
    description = service["description"]
    
    # 检查端口是否被占用
    if wait_for_port(port):
        log_warning(f"端口 {port} 已被占用，跳过启动")
        return True
    
    log_info(f"🚀 启动 {service['name']}...")
    log_info(f"   描述: {description}")
    log_info(f"   端口: {port}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    
    # 构建命令
    script_path = os.path.join(project_root, "src", script)
    cmd = [sys.executable, script_path, "--port", str(port)]
    
    try:
        process = subprocess.Popen(cmd, cwd=project_root, env=env)
        processes[service_name] = process
        
        # 等待服务启动
        time.sleep(START_DELAY)
        
        # 检查是否启动成功
        if process.poll() is None:
            # 等待端口就绪
            if wait_for_port(port, timeout=15):
                log_success(f"✅ {service['name']} 启动成功")
                return True
            else:
                log_error(f"❌ {service['name']} 端口未就绪")
                process.terminate()
                return False
        else:
            log_error(f"❌ {service['name']} 启动失败，退出码: {process.returncode}")
            return False
            
    except Exception as e:
        log_error(f"❌ {service['name']} 启动异常: {str(e)}")
        return False

def start_services_by_group(group_name):
    """按分组启动服务（带依赖顺序）"""
    service_names = SERVICE_GROUPS.get(group_name, [])
    
    # 定义服务启动顺序（基于依赖关系）
    # 多媒体服务 → 模型服务 → 主API服务 → API网关
    dependency_order = {
        "multimedia_service": 1,
        "model_service": 2,
        "api_service": 3,
        "api_gateway": 4
    }
    
    # 按依赖顺序排序
    service_names.sort(key=lambda x: dependency_order.get(x, 99))
    
    log_info(f"\n📦 按依赖顺序启动服务: {service_names}")
    
    success_count = 0
    for service_name in service_names:
        service = SERVICES.get(service_name)
        if not service.get("enabled", True):
            log_info(f"⏭️ {service['name']} 已禁用，跳过")
            continue
        
        # 检查内存
        available_mem = check_memory_available()
        if available_mem < 1:
            log_warning(f"内存不足 ({available_mem:.1f}GB)，暂停启动...")
            time.sleep(10)
        
        # 启动服务（带重试）
        success = False
        for retry in range(MAX_RETRY):
            if start_service(service_name):
                success = True
                break
            if retry < MAX_RETRY - 1:
                log_warning(f"重试启动 {service['name']} ({retry + 1}/{MAX_RETRY})")
                time.sleep(RETRY_DELAY)
        
        if success:
            success_count += 1
            # 等待服务稳定
            time.sleep(2)
        else:
            log_error(f"❌ {service['name']} 启动失败")
    
    return success_count

def stop_all_services():
    """停止所有服务"""
    log_info("\n🛑 停止所有服务...")
    for service_name, process in processes.items():
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
                log_info(f"✅ 已停止 {service_name}")
            except Exception as e:
                log_error(f"❌ 停止 {service_name} 失败: {str(e)}")
                process.kill()

def signal_handler(sig, frame):
    """信号处理"""
    stop_all_services()
    sys.exit(0)

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="动漫角色识别系统 - 稳健启动脚本")
    parser.add_argument("-g", "--group", type=str, default="all", help="服务分组")
    args = parser.parse_args()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("🎬 动漫角色识别系统 - 稳健启动脚本")
    print("=" * 60)
    
    # 检查内存
    available_mem = check_memory_available()
    log_info(f"系统可用内存: {available_mem:.1f} GB")
    
    # 启动服务
    total_services = len([s for s in SERVICES.values() if s.get("enabled")])
    success_count = start_services_by_group(args.group)
    
    print("\n" + "=" * 60)
    log_success(f"🎉 服务启动完成！成功启动 {success_count}/{total_services} 个服务")
    print("=" * 60)
    
    # 显示服务列表
    print("\n📋 运行中服务:")
    for service_name, process in processes.items():
        if process.poll() is None:
            service = SERVICES.get(service_name)
            print(f"   ✅ {service['name']}: http://localhost:{service['port']}")
    
    print("\n按 Ctrl+C 停止所有服务...")
    
    # 保持主进程运行
    try:
        while True:
            time.sleep(1)
            # 检查进程状态
            for service_name, process in list(processes.items()):
                if process.poll() is not None:
                    log_error(f"⚠️ {service_name} 意外退出，退出码: {process.returncode}")
    except KeyboardInterrupt:
        stop_all_services()

if __name__ == "__main__":
    main()