# 【技术难点】分布式服务协调

> 在多服务部署环境中，如何实现服务发现、健康检查和故障恢复是一个关键挑战。

---

## 🔍 问题背景

系统包含多个独立运行的服务，需要解决以下问题：

1. **服务发现**：如何定位可用服务？
2. **健康检查**：如何检测服务状态？
3. **端口冲突**：如何避免端口占用？
4. **故障恢复**：服务不可用时如何处理？

---

## 💡 解决方案：服务注册中心

### 服务注册与健康检查

```python
import os
import requests
import time
from typing import Dict, Optional

class ServiceRegistry:
    def __init__(self):
        # 从环境变量读取服务配置，支持 Docker/K8s 动态部署
        self.services: Dict[str, dict] = {
            "gateway": {
                "host": os.getenv("GATEWAY_HOST", "localhost"),
                "port": int(os.getenv("GATEWAY_PORT", "8000")),
                "healthy": False
            },
            "backend": {
                "host": os.getenv("BACKEND_HOST", "localhost"),
                "port": int(os.getenv("BACKEND_PORT", "8001")),
                "healthy": False
            },
            "model": {
                "host": os.getenv("MODEL_HOST", "localhost"),
                "port": int(os.getenv("MODEL_PORT", "8888")),
                "healthy": False
            }
        }
    
    def check_health(self, service_name: str) -> bool:
        """检查单个服务健康状态"""
        service = self.services.get(service_name)
        if not service:
            return False
        
        try:
            url = f"http://{service['host']}:{service['port']}/api/health"
            response = requests.get(url, timeout=5)
            healthy = response.status_code == 200
            self.services[service_name]["healthy"] = healthy
            return healthy
        except Exception:
            self.services[service_name]["healthy"] = False
            return False
    
    def check_all(self) -> Dict[str, bool]:
        """检查所有服务"""
        results = {}
        for name in self.services:
            results[name] = self.check_health(name)
        return results
```

### 环境变量配置

在实际部署中（如 Docker 或 Kubernetes），服务的 IP 地址是动态分配的。通过环境变量配置，可以灵活地适应不同的部署环境。

#### .env 文件示例

```env
# API Gateway
GATEWAY_HOST=localhost
GATEWAY_PORT=8000

# Backend API
BACKEND_HOST=localhost
BACKEND_PORT=8001

# Model Service
MODEL_HOST=localhost
MODEL_PORT=8888

# 数据库配置
DB_HOST=postgres
DB_PORT=5432
DB_NAME=anime_db

# Redis 配置
REDIS_HOST=redis
REDIS_PORT=6379
```

#### Docker Compose 中的环境变量传递

```yaml
version: '3.8'
services:
  backend:
    environment:
      - GATEWAY_HOST=gateway
      - GATEWAY_PORT=8000
      - MODEL_HOST=model_service
      - MODEL_PORT=8888
    depends_on:
      - gateway
      - model_service
```

### 端口冲突处理

```python
import socket

def is_port_available(port: int) -> bool:
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def get_available_port(start_port: int = 8000) -> int:
    """获取可用端口"""
    port = start_port
    while port < 9000:
        if is_port_available(port):
            return port
        port += 1
    raise ValueError("No available port found")

# 使用示例
port = get_available_port(8000)
print(f"可用端口: {port}")
```

---

## 🚀 使用示例

```python
# 初始化服务注册中心
registry = ServiceRegistry()

# 检查所有服务状态
status = registry.check_all()
print(f"服务状态: {status}")
# {'gateway': True, 'backend': True, 'model': False}

# 获取可用服务
backend = registry.get_available_service("backend")
if backend:
    print(f"后端服务: http://{backend['host']}:{backend['port']}")
else:
    print("后端服务不可用")

# 处理故障转移
try:
    response = requests.get(f"http://{backend['host']}:{backend['port']}/api/health")
except requests.exceptions.ConnectionError:
    print("❌ 服务不可用，尝试备用服务...")
```

---

## ⚡ 健康检查定时任务

```python
import threading

class HealthChecker:
    def __init__(self, registry: ServiceRegistry, interval: int = 30):
        self.registry = registry
        self.interval = interval
        self.running = True
    
    def start(self):
        """启动健康检查线程"""
        def check_loop():
            while self.running:
                status = self.registry.check_all()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 服务状态: {status}")
                time.sleep(self.interval)
        
        thread = threading.Thread(target=check_loop, daemon=True)
        thread.start()
    
    def stop(self):
        """停止健康检查"""
        self.running = False

# 使用示例
checker = HealthChecker(registry, interval=30)
checker.start()
```

---

## 📝 关键要点

1. **服务注册**：维护服务列表和状态
2. **健康检查**：定期检测服务可用性
3. **端口管理**：自动检测可用端口
4. **故障转移**：服务不可用时的降级处理
5. **异步检测**：使用线程定期检查，不阻塞主流程

---

*下篇预告：图像预处理与特征提取*
