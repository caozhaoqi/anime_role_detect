#!/usr/bin/env python3
"""
API网关服务
统一管理所有服务的访问，负责路由、认证、监控等功能
类似Java微服务架构，所有API通过网关统一访问
"""
import os
import sys
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn
import httpx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.core.config.service_config import get_service_config
config = get_service_config()

from src.core.logging.global_logger import get_logger
logger = get_logger("api_gateway")

SERVICES = {
    "model": {
        "url": "http://localhost:8000",
        "health_path": "/api/health",
        "docs_path": "/docs",
        "redoc_path": "/redoc"
    },
    "api": {
        "url": "http://localhost:8001",
        "health_path": "/api/health",
        "docs_path": "/docs",
        "redoc_path": "/redoc"
    },
    "multimedia": {
        "url": "http://localhost:8002",
        "health_path": "/api/health",
        "docs_path": "/docs",
        "redoc_path": "/redoc"
    }
}

app = FastAPI(
    title="Anime Role Detect API Gateway",
    description="统一API网关 - 管理所有微服务的入口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/")
async def root():
    return {
        "message": "Anime Role Detect API Gateway",
        "docs": "/docs",
        "services": {
            "api": "/api/*",
            "model": "/api/model/*",
            "search": "/api/search/*",
            "video": "/api/video/*",
            "classify": "/api/classify/*"
        },
        "service_docs": {
            "api_service": "/api/docs",
            "model_service": "/api/model/docs",
            "multimedia_service": "/api/multimedia/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "API Gateway"}

@app.get("/api/services")
async def check_services():
    status = {}
    for service_name, service_config in SERVICES.items():
        try:
            health_path = service_config.get("health_path", "/api/health")
            response = await client.get(f"{service_config['url']}{health_path}")
            status[service_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": service_config['url'],
            }
        except Exception as e:
            status[service_name] = {
                "status": "unhealthy",
                "url": service_config['url'],
                "error": str(e)
            }
    return status

# 服务文档路由 - 代理到各个服务的docs
@app.get("/api/docs")
async def api_service_docs():
    """API服务文档"""
    return RedirectResponse(url="/api/docs/")

@app.get("/api/docs/{path:path}")
async def api_service_docs_proxy(request: Request, path: str):
    return await proxy_to_service(request, "api", f"docs/{path}")

@app.get("/api/model/docs")
async def model_service_docs():
    """模型服务文档"""
    return RedirectResponse(url="/api/model/docs/")

@app.get("/api/model/docs/{path:path}")
async def model_service_docs_proxy(request: Request, path: str):
    return await proxy_to_service(request, "model", f"docs/{path}")

@app.get("/api/multimedia/docs")
async def multimedia_service_docs():
    """多媒体服务文档"""
    return RedirectResponse(url="/api/multimedia/docs/")

@app.get("/api/multimedia/docs/{path:path}")
async def multimedia_service_docs_proxy(request: Request, path: str):
    return await proxy_to_service(request, "multimedia", f"docs/{path}")

@app.get("/api/redoc")
async def api_service_redoc():
    """API服务Redoc文档"""
    return RedirectResponse(url="/api/redoc/")

@app.get("/api/redoc/{path:path}")
async def api_service_redoc_proxy(request: Request, path: str):
    return await proxy_to_service(request, "api", f"redoc/{path}")

@app.get("/api/model/redoc/{path:path}")
async def model_service_redoc_proxy(request: Request, path: str):
    return await proxy_to_service(request, "model", f"redoc/{path}")

@app.get("/api/multimedia/redoc/{path:path}")
async def multimedia_service_redoc_proxy(request: Request, path: str):
    return await proxy_to_service(request, "multimedia", f"redoc/{path}")

# OpenAPI JSON 代理
@app.get("/api/openapi.json")
async def api_openapi_json(request: Request):
    return await proxy_to_service(request, "api", "openapi.json")

@app.get("/api/model/openapi.json")
async def model_openapi_json(request: Request):
    return await proxy_to_service(request, "model", "openapi.json")

@app.get("/api/multimedia/openapi.json")
async def multimedia_openapi_json(request: Request):
    return await proxy_to_service(request, "multimedia", "openapi.json")

async def proxy_to_service(request: Request, service_name: str, path: str):
    """通用代理函数"""
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"服务 {service_name} 不存在")
    
    service_config = SERVICES[service_name]
    url = f"{service_config['url']}/{path}"
    
    logger.info(f"代理请求到 {service_name} 服务: {url}")
    
    try:
        headers = dict(request.headers)
        if "host" in headers:
            del headers["host"]
        if "content-length" in headers:
            del headers["content-length"]
        if "expect" in headers:
            del headers["expect"]

        response = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=await request.body()
        )

        try:
            content = response.json()
        except ValueError:
            content = response.content

        return JSONResponse(
            content=content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    except httpx.HTTPError as e:
        logger.error(f"代理请求失败(httpx): {e}")
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")
    except Exception as e:
        logger.error(f"代理请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(request: Request, path: str):
    logger.info(f"收到请求: {request.method} /api/{path}")

    if path.startswith("search"):
        service = "multimedia"
        url = f"http://localhost:8002/{path}"
    elif path.startswith("video"):
        service = "multimedia"
        url = f"http://localhost:8002/{path}"
    elif path.startswith("classify"):
        service = "api"
        url = f"http://localhost:8001/api/{path}"
    elif path.startswith("model/"):
        service = "model"
        model_path = path[6:] if path.startswith("model/") else path
        url = f"http://localhost:8000/api/{model_path}"
    elif path == "model" or path == "model/health":
        service = "model"
        url = f"http://localhost:8000/api/health"
    else:
        url = f"http://localhost:8001/api/{path}"
        service = "api"

    logger.info(f"转发请求到: {url}")

    try:
        headers = dict(request.headers)

        if "host" in headers:
            del headers["host"]
        if "content-length" in headers:
            del headers["content-length"]
        if "expect" in headers:
            del headers["expect"]

        body = await request.body()
        logger.info(f"Body长度: {len(body)}字节")

        response = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body
        )

        logger.info(f"收到响应: 状态码={response.status_code}, 内容长度={len(response.text)}")

        try:
            content = response.json()
        except ValueError:
            content = response.text

        return JSONResponse(
            content=content,
            status_code=response.status_code
        )
    except httpx.HTTPError as e:
        logger.error(f"代理请求失败(httpx): {e}")
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")
    except Exception as e:
        logger.error(f"代理请求处理失败: {e}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

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
