#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery 异步任务队列配置
用于处理耗时任务（如图像处理、视频分析等）
"""

import os
from celery import Celery
from celery.schedules import crontab

from src.core.config.ports import coerce_port

# 获取 Redis 配置
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = coerce_port(os.environ.get("REDIS_PORT"), 6379)
REDIS_DB = int(os.environ.get("REDIS_QUEUE_DB", 1))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# 初始化 Celery 应用
app = Celery(
    "anime_role_detect",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.tasks.image_tasks", "src.tasks.video_tasks", "src.tasks.model_tasks"],
)

# Celery 配置
app.conf.update(
    # 任务序列化方式
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # 任务过期时间（1小时）
    task_expires=3600,
    # 并发工作进程数
    worker_concurrency=4,
    # 任务预取数量
    worker_prefetch_multiplier=1,
    # 任务超时时间（5分钟）
    task_time_limit=300,
    # 软超时时间（4分钟）
    task_soft_time_limit=240,
    # 结果过期时间（1天）
    result_expires=86400,
    # 启用任务时间限制
    worker_enable_remote_control=True,
    # 定时任务配置
    beat_schedule={
        "cleanup-expired-tasks": {
            "task": "src.tasks.cleanup.cleanup_expired_tasks",
            "schedule": crontab(minute=0, hour="*/6"),  # 每6小时执行一次
        },
        "sync-images": {
            "task": "src.tasks.image_tasks.sync_images_task",
            "schedule": crontab(minute=0, hour="*/4"),  # 每4小时同步一次图片
        },
    },
)

# 配置任务路由
app.conf.task_routes = {
    "src.tasks.image_tasks.*": {"queue": "image_queue"},
    "src.tasks.video_tasks.*": {"queue": "video_queue"},
    "src.tasks.model_tasks.*": {"queue": "model_queue"},
}

# 配置任务优先级
app.conf.task_queue_max_priority = 10


def get_celery_app():
    """获取 Celery 应用实例"""
    return app


def start_worker(queues="image_queue,video_queue,model_queue"):
    """启动 Celery Worker"""
    app.worker_main(["worker", "--loglevel=info", "-Q", queues])


def start_beat():
    """启动 Celery Beat（定时任务）"""
    app.start(["beat", "--loglevel=info"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "worker":
            queues = sys.argv[2] if len(sys.argv) > 2 else "image_queue,video_queue,model_queue"
            start_worker(queues)
        elif command == "beat":
            start_beat()
        else:
            print(f"未知命令: {command}")
            sys.exit(1)
    else:
        print("Usage: celery_config.py [worker|beat] [queues]")
        sys.exit(1)
