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

### 请求路由中间件

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(title="Character Classification Gateway")

SERVICES = {
    "backend": "http://localhost:8001",
    "model": "http://localhost:8888"
}

@app.middleware("http")
async def proxy_middleware(request: Request, call_next):
    """请求代理中间件"""
    # 健康检查直接返回
    if request.url.path == "/api/health":
        return JSONResponse(content={"status": "healthy", "service": "gateway"})
    
    path = request.url.path
    headers = dict(request.headers)
    headers.pop("host", None)
    
    # 根据路径前缀路由
    if path.startswith("/api/auth") or path.startswith("/api/classify"):
        target_url = f"{SERVICES['backend']}{path}"
    elif path.startswith("/api/model"):
        target_url = f"{SERVICES['model']}{path}"
    else:
        raise HTTPException(status_code=404, detail="Not found")
    
    # 绕过系统代理，确保 localhost 通信
    async with httpx.AsyncClient(trust_env=False) as client:
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=await request.body(),
            timeout=60
        )
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code
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
