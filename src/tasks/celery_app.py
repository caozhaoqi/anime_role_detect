#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery应用配置
用于异步任务处理
"""

import os
from celery import Celery
from celery.signals import worker_ready, worker_shutdown

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "src.core.config.settings")

celery_app = Celery(
    "anime_role_detect",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
    include=["src.tasks.classify_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)


@worker_ready.connect
def on_worker_ready(**kwargs):
    """Worker启动时的回调"""
    from src.core.logging import get_enhanced_logger as get_logger

    logger = get_logger("celery")
    logger.info("Celery worker已就绪")


@worker_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Worker关闭时的回调"""
    from src.core.logging import get_enhanced_logger as get_logger

    logger = get_logger("celery")
    logger.info("Celery worker正在关闭")


if __name__ == "__main__":
    celery_app.start()
