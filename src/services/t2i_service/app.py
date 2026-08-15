#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t2i_service 入口：独立 FastAPI 进程，跑在 t2i-mac venv。

运行：
    t2i-mac/bin/python src/services/t2i_service/app.py --port 8100
（或在 supervisord / scripts/t2i/run_t2i_service.sh 中启动）
"""
from __future__ import annotations

import argparse

from fastapi import FastAPI

from src.services.t2i_service import config
from src.services.t2i_service.router import router
from src.core.metrics import metrics

app = FastAPI(
    title="Anime Role Detect — T2I Service",
    description="角色图像生成微服务（IP-Adapter 免训练 / LoRA 训练）",
    version="1.0.0",
)

app.include_router(router)

# E: 每次进程启动（含 supervisord autorestart 后的重启）记一次，用于量化重启频率
metrics.inc_counter("t2i.restarts")


@app.get("/")
async def root():
    return {
        "service": "t2i",
        "status": "running",
        "endpoints": [
            "GET  /api/health",
            "GET  /api/t2i/roles",
            "POST /api/t2i/generate",
            "POST /api/t2i/train",
            "GET  /api/t2i/jobs/{job_id}",
            "POST /api/t2i/chat",
            "POST /api/t2i/unload",
        ],
    }


def _graceful_shutdown():
    """卸载权重 + 关闭线程池，避免重启时的信号量泄漏与孤儿内存。"""
    try:
        from src.services.t2i_service.generator import T2IGenerator, _executor

        T2IGenerator.get_instance().unload()
        _executor.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    import atexit
    import signal
    import logging
    import uvicorn

    parser = argparse.ArgumentParser(description="T2I 生成微服务")
    parser.add_argument("--host", type=str, default=config.SERVICE_HOST)
    parser.add_argument("--port", type=int, default=config.SERVICE_PORT)
    args = parser.parse_args()

    atexit.register(_graceful_shutdown)
    try:
        signal.signal(signal.SIGTERM, lambda *_: _graceful_shutdown())
    except (ValueError, OSError):
        pass

    # 自定义日志配置：uvicorn 默认格式缺 asctime，supervisord 也不给子进程 stdout 加时间，
    # 这里在 default/access 两个 formatter 显式补时间戳，落盘的 t2i-service.log/.err.log 才有时间。
    LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s | %(levelprefix)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s | %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": None,
            },
        },
        "handlers": {
            "default": {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"},
            "access": {"formatter": "access", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"},
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_config=LOG_CONFIG,
    )
