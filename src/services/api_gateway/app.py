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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn
import httpx

# 路径配置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from src.core.config.service_config import get_service_config

config = get_service_config()

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("api_gateway")

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


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api-gateway"}


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
async def health_check():
    return {"status": "healthy", "service": "API Gateway"}


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
    url = f"{config.MULTIMEDIA_SERVICE_URL}/video/result/{filename}"
    logger.info(f"代理视频下载: {url}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(url)
            content = response.content
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
            raise HTTPException(status_code=503, detail=f"多媒体服务连接失败: {e}")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(request: Request, path: str):
    """
    通用请求转发逻辑，带重试机制
    """
    logger.info(f"收到请求: {request.method} /api/{path}")

    # 1. 路由分配逻辑
    if (
        path.startswith("search/image")
        or path.startswith("search/build-index")
        or path.startswith("search/stats")
    ):
        service = "search"
        url = f"{config.SEARCH_SERVICE_URL}/api/{path}"
    elif path.startswith("video/"):
        service = "multimedia"
        url = f"{config.MULTIMEDIA_SERVICE_URL}/video/{path[6:]}"
    elif path.startswith("classify") or path.startswith("model/"):
        service = "model"
        if path.startswith("classify"):
            # 处理 classify 路径
            classify_path = path
            # 将 /api/classify/multi-role 映射到 /api/model/detect-multiple
            if classify_path == "classify/multi-role":
                url = f"{config.MODEL_SERVICE_URL}/api/model/detect-multiple"
            else:
                url = f"{config.MODEL_SERVICE_URL}/api/{classify_path}"
        else:
            model_path = path[6:]
            url = f"{config.MODEL_SERVICE_URL}/api/model/{model_path}"
    elif path == "model" or path == "model/health":
        service = "model"
        url = f"{config.MODEL_SERVICE_URL}/api/health"
    else:
        service = "api"
        url = f"{config.CORE_API_URL}/api/{path}"

    logger.info(f"转发请求到 [{service}]: {url}")

    # 保留原始请求的查询参数
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("expect", None)
    body = await request.body()

    max_retries = 3
    retry_delay = 0.5

    for attempt in range(max_retries):
        try:
            response = await client.request(
                method=request.method, url=url, headers=headers, content=body
            )

            try:
                content = response.json()
            except ValueError:
                content = response.text

            return JSONResponse(content=content, status_code=response.status_code)

        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            logger.warning(f"连接失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            logger.error(f"代理请求最终失败: {e}")
            raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")
        except httpx.HTTPError as e:
            logger.error(f"代理请求失败: {e}")
            raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")
        except Exception as e:
            logger.error(f"代理请求处理失败: {e}\n{traceback.format_exc()}")
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
