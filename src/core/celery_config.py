#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery 异步任务队列配置（统一入口 / re-export 薄壳）

历史背景
--------
项目曾同时存在两个 Celery app：
  * 本文件（src.core.celery_config）—— broker/backend 均为 Redis，供 k8s 的
    celery-worker / celery-beat 使用，include image/video/model 任务；
  * src.tasks.celery_app —— broker 为 RabbitMQ（默认）、backend 为 Redis，
    供 /api/classify 异步接口使用，include classify 任务。

二者 broker 不同、注册表不同，导致 classify 任务在 k8s 环境成为「黑洞」
（k8s worker 不认识 classify_image_task）。详见 ADR-002。

现统一为单一 canonical app：src.tasks.celery_app 为唯一实例，本文件仅做
re-export，确保两条入口（`-A src.core.celery_config` 与 `-A src.tasks.celery_app`）
指向同一对象。所有任务模块（classify/image/video/model/cleanup）现均绑定到该实例。
"""

# 关键：直接复用 canonical app，不再各自构造 Celery 实例。
# celery_app 在导入时不会反向 import src.core（仅运行时延迟 import logging），
# 因此本文件（位于 src.core）re-export 它不会形成循环依赖。
from src.tasks.celery_app import celery_app as app

# 向后兼容：原 celery_config 暴露的辅助函数
def get_celery_app():
    """获取 Celery 应用实例"""
    return app


def start_worker(queues="image_queue,video_queue,model_queue,celery"):
    """启动 Celery Worker（默认监听全部队列，含 classify 的默认 celery 队列）"""
    app.worker_main(["worker", "--loglevel=info", "-Q", queues])


def start_beat():
    """启动 Celery Beat（定时任务）"""
    app.start(["beat", "--loglevel=info"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "worker":
            queues = sys.argv[2] if len(sys.argv) > 2 else "image_queue,video_queue,model_queue,celery"
            start_worker(queues)
        elif command == "beat":
            start_beat()
        else:
            print(f"未知命令: {command}")
            sys.exit(1)
    else:
        print("Usage: celery_config.py [worker|beat] [queues]")
        sys.exit(1)
