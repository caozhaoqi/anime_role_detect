#!/usr/bin/env python3
"""
API网关服务 - 聚合文档版
统一管理所有服务的访问，负责路由、认证、监控等功能
支持 Swagger UI 聚合所有微服务 API 文档
"""
import os
import sys
import traceback
import asyncio
import time
import hashlib
import math
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import httpx

# 路径配置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from src.core.config.service_config import get_service_config

config = get_service_config()

from src.core.logging import get_enhanced_logger as get_logger
from src.core.logging_setup import setup_logging
from src.services.api_gateway.routing import resolve_route

logger = get_logger("api_gateway")

# 2.5 结构化日志：统一 loguru 输出为 JSON 行（幂等，仅配置一次）
setup_logging("api-gateway")

# 微服务配置 - 使用配置文件中的端口
SERVICES = {
    "model": {
        "url": config.MODEL_SERVICE_URL,
        "prefix": "/api/model",
        "name": "模型服务 (Model Service)",
        "docs_path": "/openapi.json",
    },
    "api": {
        "url": config.CORE_API_URL,
        "prefix": "/api",
        "name": "业务API服务 (Core API)",
        "docs_path": "/api/openapi.json",
    },
    "multimedia": {
        "url": config.MULTIMEDIA_SERVICE_URL,
        "prefix": "/api",
        "name": "多媒体服务 (Multimedia)",
        "docs_path": "/openapi.json",
    },
    "search": {
        "url": config.SEARCH_SERVICE_URL,
        "prefix": "/api/search",
        "name": "搜索服务 (Search Service)",
        "docs_path": "/openapi.json",
    },
}

app = FastAPI(
    title="Anime Role Detect API Gateway",
    description="统一API网关 - 聚合微服务入口",
    version="1.0.0",
    docs_url=None,  # 禁用默认文档，使用自定义聚合版
    redoc_url=None,
)

# OpenTelemetry 链路追踪
try:
    from src.utils.monitoring.opentelemetry import instrument_app
    instrument_app(app, service_name="api-gateway")
except Exception as e:
    logger.warning(f"OpenTelemetry 初始化失败: {e}")

# 请求追踪中间件 - 自动记录所有请求
try:
    from src.utils.monitoring.tracing.tracer import Tracer
    from src.services.support.trace_storage_service import get_trace_storage_service
    
    tracer = Tracer(service_name="api-gateway")
    trace_storage = get_trace_storage_service()
    
    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID", "")
        if not trace_id:
            import uuid
            trace_id = str(uuid.uuid4()).replace("-", "")
        
        start_time = time.time()
        response = None
        status = "OK"
        
        try:
            response = await call_next(request)
            status = "OK" if response.status_code < 400 else "ERROR"
            return response
        except Exception as e:
            status = "ERROR"
            raise
        finally:
            if response:
                end_time = time.time()
                duration_ms = round((end_time - start_time) * 1000, 2)
                endpoint = f"{request.method} {request.url.path}"
                
                try:
                    from src.utils.monitoring.tracing.trace import Trace
                    from src.utils.monitoring.tracing.span import Span, SpanContext
                    
                    local_trace = Trace(trace_id=trace_id)
                    local_trace.start_time = start_time
                    local_trace.end_time = end_time
                    
                    context = SpanContext(trace_id=trace_id)
                    span = Span(name=endpoint, context=context, start_time=start_time)
                    span.end_time = end_time
                    span.attributes = {
                        "http.method": request.method,
                        "http.path": request.url.path,
                        "http.status_code": response.status_code,
                    }
                    local_trace.add_span(span)
                    local_trace.status = status
                    local_trace.duration_ms = duration_ms
                    
                    trace_storage.store_trace(local_trace)
                except Exception as trace_err:
                    logger.debug(f"存储追踪数据失败: {trace_err}")
    
    logger.info("请求追踪中间件已启用")
except Exception as e:
    logger.warning(f"请求追踪中间件初始化失败: {e}")

# ======================================================================
# 2.4 API Gateway 增强：入口级限流 + 上游轻量熔断（无新增第三方依赖）
# ----------------------------------------------------------------------
# 限流：基于内存令牌桶，按客户端标识（JWT Bearer token 哈希 或 客户端 IP）限流，
#       防止单一客户端刷爆网关。可通过环境变量 GATEWAY_RATE_LIMIT_PER_MIN 等调整。
# 熔断：仅在网关入口层对"上游不可达 / 5xx"做快速失败，与 circuit_breaker_service
#       （api-service 侧"服务间调用级熔断"，含 HALF_OPEN 与降级 fallback）明确分层，
#       不重复造轮子——网关不实现降级/半开，仅做内存单进程的固定窗口快速失败。
# ======================================================================

# --- 入口级限流（令牌桶） ---
_RATE_LIMIT_EXEMPT = {"/health", "/live", "/ready", "/api/health"}


def _rate_limit_client_key(request):
    # type: (Request) -> str
    """客户端标识：优先用 JWT Bearer token 哈希（近似 JWT sub），否则用客户端 IP。"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[len("Bearer "):].strip()
        if token:
            return "tok:" + hashlib.sha256(token.encode()).hexdigest()[:16]
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        ip = xff.split(",")[0].strip()
    elif request.client is not None:
        ip = request.client.host
    else:
        ip = "unknown"
    return "ip:" + ip


class _TokenBucketRateLimiter:
    """极简内存令牌桶限流（单进程；多 worker 时各进程独立计数）。"""

    def __init__(self, rate_per_min, burst=None):
        self.rate = float(rate_per_min)
        self.capacity = float(burst if burst is not None else rate_per_min)
        self.refill_per_sec = self.rate / 60.0
        self._buckets = {}

    def check(self, key):
        now = time.time()
        bucket = self._buckets.get(key, [self.capacity, now])
        tokens, last = bucket[0], bucket[1]
        elapsed = now - last
        tokens = min(self.capacity, tokens + elapsed * self.refill_per_sec)
        if tokens >= 1.0:
            tokens -= 1.0
            self._buckets[key] = [tokens, now]
            return True, 0.0
        retry_after = (1.0 - tokens) / self.refill_per_sec
        self._buckets[key] = [tokens, now]
        return False, retry_after


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request, call_next):
        if request.url.path in _RATE_LIMIT_EXEMPT:
            return await call_next(request)
        client_key = _rate_limit_client_key(request)
        allowed, retry_after = self.limiter.check(client_key)
        if allowed:
            return await call_next(request)
        retry_after_int = int(math.ceil(retry_after))
        logger.warning("限流触发 path=%s key=%s", request.url.path, client_key)
        resp = JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": "请求过于频繁，请稍后再试",
                "retry_after": retry_after_int,
            },
        )
        resp.headers["Retry-After"] = str(retry_after_int)
        return resp


_gateway_rate_limiter = _TokenBucketRateLimiter(
    rate_per_min=int(os.getenv("GATEWAY_RATE_LIMIT_PER_MIN", "60")),
    burst=int(os.getenv("GATEWAY_RATE_LIMIT_BURST", os.getenv("GATEWAY_RATE_LIMIT_PER_MIN", "60"))),
)
app.add_middleware(RateLimitMiddleware, limiter=_gateway_rate_limiter)


# --- 网关层轻量熔断（仅针对上游不可达 / 5xx，与 circuit_breaker_service 分层）---
_GATEWAY_CB_THRESHOLD = int(os.getenv("GATEWAY_CB_THRESHOLD", "5"))
_GATEWAY_CB_OPEN_SECONDS = int(os.getenv("GATEWAY_CB_OPEN_SECONDS", "30"))
_GATEWAY_BREAKERS = {}


def _breaker_is_open(service):
    # type: (str) -> bool
    b = _GATEWAY_BREAKERS.get(service)
    if b is None or b["open_until"] is None:
        return False
    if time.time() < b["open_until"]:
        return True
    # 开路到期自动恢复（固定窗口，无半开态）：重置后放行
    b["open_until"] = None
    b["failures"] = 0
    return False


def _breaker_record_success(service):
    # type: (str) -> None
    b = _GATEWAY_BREAKERS.setdefault(service, {"failures": 0, "open_until": None})
    b["failures"] = 0
    b["open_until"] = None


def _breaker_record_failure(service):
    # type: (str) -> None
    b = _GATEWAY_BREAKERS.setdefault(service, {"failures": 0, "open_until": None})
    b["failures"] += 1
    if b["failures"] >= _GATEWAY_CB_THRESHOLD and b["open_until"] is None:
        b["open_until"] = time.time() + _GATEWAY_CB_OPEN_SECONDS
        logger.warning("网关熔断开启[%s]，%ss 内对该上游快速失败", service, _GATEWAY_CB_OPEN_SECONDS)


# CORS 配置：生产环境应通过 CORS_ORIGINS 环境变量限定，默认仅允许本地开发
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
if _cors_origins_env:
    allowed_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Accept",
        "Origin",
        "X-CSRF-Token",
    ],
)

# 添加响应压缩中间件
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 挂载监控和日志路由
try:
    from src.services.api_gateway.routers.monitor import router as monitor_router
    from src.services.api_gateway.routers.logs import router as logs_router
    app.include_router(monitor_router)
    app.include_router(logs_router)
    logger.info("监控和日志路由挂载成功")
except Exception as e:
    logger.warning(f"监控路由挂载失败: {e}")

client = None


@app.on_event("startup")
async def startup_event():
    global client
    logger.info("启动API网关服务")
    client = httpx.AsyncClient(timeout=60.0, trust_env=False)
    logger.info("API网关服务启动完成")


@app.on_event("shutdown")
async def shutdown_event():
    global client
    if client:
        await client.aclose()
    logger.info("API网关服务已关闭")


GATEWAY_VERSION = "2.3.0"


async def _aggregate_downstream_health(http_client) -> dict:
    """探测各下游微服务的 /api/health，返回结构化状态（失败仅标记 down，不抛异常）。

    读取下游 JSON 中的顶层 status 字段（healthy/degraded/unhealthy）以正确传播降级状态：
    HTTP 非 200 或连接异常 -> down；200 且 status=healthy -> up；
    200 但 status=degraded/unhealthy -> degraded（网关自身随之降级）。
    """
    results = {}
    for svc_key, svc_cfg in SERVICES.items():
        url = svc_cfg["url"].rstrip("/") + "/api/health"
        try:
            resp = await http_client.get(url, timeout=1.0)
            if resp.status_code != 200:
                results[svc_key] = {"status": "down", "status_code": resp.status_code}
                continue
            try:
                body = resp.json()
                downstream_status = body.get("status", "healthy")
            except Exception:
                downstream_status = "healthy"
            if downstream_status == "healthy":
                mapped = "up"
            elif downstream_status in ("degraded", "unhealthy"):
                mapped = "degraded"
            else:
                mapped = "up"
            results[svc_key] = {
                "status": mapped,
                "status_code": resp.status_code,
                "reported": downstream_status,
            }
        except Exception as e:
            results[svc_key] = {"status": "down", "error": str(e)}
    return results


@app.get("/health")
async def health_check_root():
    return {
        "status": "healthy",
        "service": "api-gateway",
        "version": GATEWAY_VERSION,
        "checks": {"self": "up"},
    }


@app.get("/live")
async def liveness_check():
    """K8s liveness 端点 - 进程存活检查"""
    return {"status": "alive"}


@app.get("/ready")
async def readiness_check():
    """K8s readiness 端点 - 网关无强依赖，进程存活即就绪"""
    return {"status": "ready"}


# --- 文档聚合核心逻辑 ---


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义 Swagger UI 页面，实现类似 Java 微服务的聚合文档切换"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <link rel="shortcut icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <title>Anime Role Detect API Gateway - 聚合文档</title>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                urls: [
                    {url: "/openapi.json", name: "网关自身"},
                    {url: "/api/core/openapi.json", name: "核心API服务"},
                    {url: "/api/model/openapi.json", name: "模型服务"},
                    {url: "/api/multimedia/openapi.json", name: "多媒体服务"},
                    {url: "/api/search/openapi.json", name: "搜索服务"}
                ],
                dom_id: "#swagger-ui",
                deepLinking: true
            });
        };
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content, media_type="text/html")


async def get_and_fix_openapi(service_key: str):
    """
    抓取微服务的 openapi.json 并修正路径前缀
    确保在网关 docs 页面 'Try it out' 能够路由到网关再转发
    """
    if service_key not in SERVICES:
        return {"openapi": "3.0.0", "info": {"title": "Error"}, "paths": {}}

    svc = SERVICES[service_key]
    docs_path = svc.get("docs_path", "/openapi.json")

    try:
        response = await client.get(f"{svc['url']}{docs_path}")
        if response.status_code != 200:
            return {
                "openapi": "3.0.0",
                "info": {"title": f"{svc['name']} (Unavailable)"},
                "paths": {},
            }

        data = response.json()

        # 更新服务信息，添加友好名称
        if "info" in data:
            data["info"]["title"] = svc["name"]

        # 核心修正：遍历 paths，为所有接口增加网关要求的转发前缀
        prefix = svc["prefix"]
        if "paths" in data:
            new_paths = {}
            for path, methods in data["paths"].items():
                # 根据不同服务类型调整路径映射
                if service_key == "model":
                    # model 服务的路径映射: /api/model/predict -> model_service/api/predict
                    if path.startswith("/api"):
                        full_path = f"/api/model{path[4:]}" if path != "/api" else "/api/model"
                    else:
                        full_path = f"{prefix}{path}"
                elif service_key == "search":
                    # search 服务的路径映射: /api/search/image -> /api/search/image
                    if path.startswith("/api"):
                        full_path = path
                    else:
                        full_path = f"{prefix}{path}".replace("//", "/")
                elif service_key == "api":
                    # core api 服务：路径已经有 /api 前缀，不需要重复添加
                    # 例如 /api/classify -> /api/classify
                    full_path = path
                else:
                    # 普通 multimedia 转发逻辑
                    full_path = f"{prefix}{path}".replace("//", "/")

                new_paths[full_path] = methods
            data["paths"] = new_paths

        # 指向网关自身
        data["servers"] = [{"url": "/", "description": "API Gateway"}]
        return data
    except Exception as e:
        logger.error(f"无法获取 {svc['name']} 文档: {e}")
        return {
            "openapi": "3.0.0",
            "info": {"title": f"{svc['name']} (Connection Error)"},
            "paths": {},
        }


@app.get("/api/model/openapi.json", include_in_schema=False)
async def model_openapi():
    return await get_and_fix_openapi("model")


@app.get("/api/core/openapi.json", include_in_schema=False)
async def core_api_openapi():
    return await get_and_fix_openapi("api")


@app.get("/api/multimedia/openapi.json", include_in_schema=False)
async def multimedia_openapi():
    return await get_and_fix_openapi("multimedia")


@app.get("/api/search/openapi.json", include_in_schema=False)
async def search_openapi():
    return await get_and_fix_openapi("search")


# --- 原有代理路由逻辑 ---


@app.get("/")
async def root():
    return {
        "message": "Anime Role Detect API Gateway",
        "docs": "/docs",
        "status": "/api/services",
        "service_docs": {
            "aggregated": "/docs",
            "model_json": "/api/model/openapi.json",
            "core_json": "/api/core/openapi.json",
            "multimedia_json": "/api/multimedia/openapi.json",
            "search_json": "/api/search/openapi.json",
        },
        "services": {
            "model": "模型服务 - 角色识别、特征提取",
            "api": "核心API服务 - 业务逻辑",
            "multimedia": "多媒体服务 - 视频处理",
            "search": "搜索服务 - 以图搜图",
        },
    }


@app.get("/api/health")
async def health_check_api():
    # 复用网关启动时创建的 httpx client；TestClient / 未启动时临时创建
    http_client = client
    own_client = False
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=2.0, trust_env=False)
        own_client = True
    try:
        downstream = await _aggregate_downstream_health(http_client)
    finally:
        if own_client:
            await http_client.aclose()

    downstream_down = [k for k, v in downstream.items() if v.get("status") != "up"]
    overall = "degraded" if downstream_down else "healthy"
    return {
        "status": overall,
        "service": "API Gateway",
        "version": GATEWAY_VERSION,
        "checks": {"self": "up", "downstream": downstream},
    }


@app.get("/api/services")
async def check_services():
    """检查所有微服务状态"""
    status = {}
    for service_name, service_config in SERVICES.items():
        try:
            response = await client.get(f"{service_config['url']}/api/health")
            status[service_name] = {
                "name": service_config["name"],
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": service_config["url"],
                "prefix": service_config["prefix"],
                "docs": f"{service_config['prefix']}/openapi.json",
            }
        except Exception as e:
            status[service_name] = {
                "name": service_config["name"],
                "status": "unhealthy",
                "url": service_config["url"],
                "error": str(e),
            }
    return {"services": status, "gateway_status": "running"}


@app.api_route("/api/video/result/{filename}", methods=["GET"])
async def proxy_video_download(request: Request, filename: str):
    """
    视频下载代理 — 使用流式响应传输二进制视频文件
    """
    # 网关层熔断（multimedia）：已开路则直接快速失败，避免反复重试打爆已故障上游
    if _breaker_is_open("multimedia"):
        raise HTTPException(status_code=503, detail="多媒体服务暂时不可用（熔断保护）")
    url = f"{config.MULTIMEDIA_SERVICE_URL}/video/result/{filename}"
    logger.info(f"代理视频下载: {url}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(url)
            content = response.content
            _breaker_record_success("multimedia")
            return StreamingResponse(
                iter([content]),
                media_type=response.headers.get("content-type", "video/mp4"),
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Length": str(len(content)),
                },
                status_code=response.status_code,
            )
        except httpx.ConnectError as e:
            _breaker_record_failure("multimedia")
            raise HTTPException(status_code=503, detail=f"多媒体服务连接失败: {e}")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(request: Request, path: str):
    """
    通用请求转发逻辑，带重试机制
    """
    logger.info(f"收到请求: {request.method} /api/{path}")

    # 1. 路由分配逻辑（声明化：单一事实源见 src/services/api_gateway/routing.ROUTE_TABLE）
    service, url = resolve_route(path, SERVICES)
    logger.info(f"转发请求到 [{service}]: {url}")

    # 保留原始请求的查询参数
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("expect", None)
    body = await request.body()

    # 网关层熔断：已开路则直接快速失败（避免反复重试打爆已故障的上游）
    if _breaker_is_open(service):
        logger.warning(f"上游[{service}]熔断中，快速失败返回 503")
        raise HTTPException(status_code=503, detail=f"上游服务[{service}]暂时不可用（熔断保护）")

    max_retries = 3
    retry_delay = 0.5

    for attempt in range(max_retries):
        try:
            response = await client.request(
                method=request.method, url=url, headers=headers, content=body
            )

            # 5xx 视为上游故障：计入熔断，但仍把上游 5xx 转发给客户端（保持原有语义）
            if response.status_code >= 500:
                _breaker_record_failure(service)
                try:
                    content = response.json()
                except ValueError:
                    content = response.text
                return JSONResponse(content=content, status_code=response.status_code)

            try:
                content = response.json()
            except ValueError:
                content = response.text

            _breaker_record_success(service)  # 上游可达 → 重置熔断计数
            return JSONResponse(content=content, status_code=response.status_code)

        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            logger.warning(f"连接失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            logger.error(f"代理请求最终失败: {e}")
            _breaker_record_failure(service)  # 上游不可达 → 累计失败
            raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")
        except httpx.HTTPError as e:
            logger.error(f"代理请求失败: {e}")
            _breaker_record_failure(service)
            raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")
        except Exception as e:
            logger.error(f"代理请求处理失败: {e}\n{traceback.format_exc()}")
            # 网关内部错误（非上游故障）不计入熔断
            raise HTTPException(status_code=500, detail="内部服务器错误")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API网关服务")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        timeout_keep_alive=30,
    )
