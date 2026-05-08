# 【技术难点】API Gateway 设计与实现

> API Gateway 作为系统的统一入口，需要处理请求路由、认证转发、跨服务通信等复杂任务。

---

## 🔍 问题背景

系统采用微服务架构，包含多个独立服务：

| 服务 | 端口 | 职责 |
|------|------|------|
| API Gateway | 8000 | 统一入口、请求路由 |
| Backend API | 8001 | 业务逻辑、用户认证 |
| Model Service | 8888 | 模型推理、特征提取 |
| Frontend | 3001 | 用户界面 |

**核心挑战**：如何实现统一入口，同时处理认证转发和服务间通信？

---

## 💡 解决方案

### 请求路由中间件（带容错机制）

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import time

app = FastAPI(title="Character Classification Gateway")

SERVICES = {
    "backend": "http://localhost:8001",
    "model": "http://localhost:8888"
}

# 断路器状态
circuit_breakers = {
    "backend": {"tripped": False, "last_failure": 0, "failure_count": 0},
    "model": {"tripped": False, "last_failure": 0, "failure_count": 0}
}

CIRCUIT_BREAKER_THRESHOLD = 5  # 连续失败次数阈值
CIRCUIT_BREAKER_TIMEOUT = 30   # 断路器恢复时间（秒）

def is_circuit_open(service_name: str) -> bool:
    """检查断路器是否打开"""
    cb = circuit_breakers[service_name]
    
    # 如果断路器未触发，返回关闭状态
    if not cb["tripped"]:
        return False
    
    # 检查是否超过恢复时间
    if time.time() - cb["last_failure"] > CIRCUIT_BREAKER_TIMEOUT:
        cb["tripped"] = False
        cb["failure_count"] = 0
        return False
    
    return True

def trip_circuit(service_name: str):
    """触发断路器"""
    cb = circuit_breakers[service_name]
    cb["failure_count"] += 1
    
    if cb["failure_count"] >= CIRCUIT_BREAKER_THRESHOLD:
        cb["tripped"] = True
        cb["last_failure"] = time.time()
        print(f"🔴 断路器触发: {service_name}")

@app.middleware("http")
async def proxy_middleware(request: Request, call_next):
    """请求代理中间件（带超时重试和断路器）"""
    # 健康检查直接返回
    if request.url.path == "/api/health":
        return JSONResponse(content={"status": "healthy", "service": "gateway"})
    
    path = request.url.path
    headers = dict(request.headers)
    headers.pop("host", None)
    
    # 根据路径前缀路由
    service_name = None
    if path.startswith("/api/auth") or path.startswith("/api/classify"):
        target_url = f"{SERVICES['backend']}{path}"
        service_name = "backend"
        timeout = 30  # 后端服务超时时间
    elif path.startswith("/api/model"):
        target_url = f"{SERVICES['model']}{path}"
        service_name = "model"
        timeout = 60  # 模型服务超时时间更长
    else:
        return JSONResponse(
            content={"code": 404, "message": "Not found", "data": None},
            status_code=404
        )
    
    # 检查断路器状态
    if is_circuit_open(service_name):
        return JSONResponse(
            content={
                "code": 503,
                "message": f"Service {service_name} is temporarily unavailable",
                "data": None
            },
            status_code=503
        )
    
    # 带重试的请求
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=await request.body(),
                    timeout=timeout
                )
                # 重置失败计数
                circuit_breakers[service_name]["failure_count"] = 0
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code
                )
        
        except httpx.TimeoutException:
            if attempt < max_retries:
                await asyncio.sleep(1)  # 等待1秒后重试
                continue
            trip_circuit(service_name)
            return JSONResponse(
                content={
                    "code": 504,
                    "message": f"Request to {service_name} timed out",
                    "data": None
                },
                status_code=504
            )
        
        except httpx.HTTPError as e:
            if attempt < max_retries:
                await asyncio.sleep(1)
                continue
            trip_circuit(service_name)
            return JSONResponse(
                content={
                    "code": 503,
                    "message": f"Service {service_name} unavailable: {str(e)}",
                    "data": None
                },
                status_code=503
            )
```

### 认证头转发

```python
def forward_auth_header(headers: dict) -> dict:
    """转发认证头到后端服务"""
    auth_headers = {}
    
    # 保留认证相关的头
    auth_keys = ["authorization", "token", "x-api-key"]
    for key, value in headers.items():
        if key.lower() in auth_keys:
            auth_headers[key] = value
    
    return auth_headers
```

---

## 🚀 使用示例

```bash
# 启动 Gateway
python3 -m uvicorn gateway:app --host 0.0.0.0 --port 8000

# 通过 Gateway 访问后端服务
curl http://localhost:8000/api/health
# {"status": "healthy", "service": "gateway"}

# 用户登录
curl -X POST -F "username=admin" -F "password=admin123" \
     http://localhost:8000/api/auth/login

# 图像分类
curl -X POST -F "file=@test.jpg" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/classify
```

---

## ⚡ 关键配置

```python
# 服务配置
SERVICE_CONFIG = {
    "backend": {
        "host": "localhost",
        "port": 8001,
        "timeout": 60
    },
    "model": {
        "host": "localhost", 
        "port": 8888,
        "timeout": 120  # 模型推理可能需要更长时间
    }
}

# 路由规则
ROUTE_RULES = {
    "/api/auth": "backend",
    "/api/classify": "backend",
    "/api/history": "backend",
    "/api/model": "model",
    "/api/feature": "model"
}
```

---

## 📝 关键要点

1. **统一入口**：所有请求通过 Gateway 进入
2. **请求路由**：根据路径前缀转发到对应服务
3. **认证转发**：透传认证头到后端服务
4. **系统代理绕过**：使用 `trust_env=False` 确保 localhost 通信
5. **超时控制**：根据服务特性设置不同超时时间

---

*下篇预告：分布式服务协调*
