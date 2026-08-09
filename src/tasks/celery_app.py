#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery应用配置
用于异步任务处理
"""

import os
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready, worker_shutdown

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "src.core.config.settings")

def _resolve_celery_broker() -> str:
    """Broker 解析优先级：
    1. 显式 CELERY_BROKER_URL（最强，用于 k8s/compose 显式指定 amqp）
    2. 由 RABBITMQ_* 环境变量构造 AMQP URL（RabbitMQ 现为默认 broker）
    3. Redis 作为最终回退，确保未部署 RabbitMQ 时仍可降级运行
    """
    if os.environ.get("CELERY_BROKER_URL"):
        return os.environ["CELERY_BROKER_URL"]
    host = os.environ.get("RABBITMQ_HOST", "localhost")
    try:
        port = int(os.environ.get("RABBITMQ_PORT", "5672"))
    except ValueError:
        port = 5672
    user = os.environ.get("RABBITMQ_USER", "guest")
    pwd = os.environ.get("RABBITMQ_PASSWORD", "guest")
    return f"amqp://{user}:{pwd}@{host}:{port}//"


def _resolve_celery_backend() -> str:
    """结果后端解析：优先 CELERY_RESULT_BACKEND，否则由 REDIS_* 构造（便于容器化部署）。"""
    if os.environ.get("CELERY_RESULT_BACKEND"):
        return os.environ["CELERY_RESULT_BACKEND"]
    rh = os.environ.get("REDIS_HOST", "localhost")
    try:
        rp = int(os.environ.get("REDIS_PORT", "6379"))
    except ValueError:
        rp = 6379
    rdb = os.environ.get("REDIS_QUEUE_DB", "2")
    return f"redis://{rh}:{rp}/{rdb}"


celery_app = Celery(
    "anime_role_detect",
    broker=_resolve_celery_broker(),
    backend=_resolve_celery_backend(),
    # 单一 canonical app：classify（默认 celery 队列）+ image/video/model（专用队列）
    include=[
        "src.tasks.classify_tasks",
        "src.tasks.image_tasks",
        "src.tasks.video_tasks",
        "src.tasks.model_tasks",
    ],
)

# RabbitMQ 新版本默认禁止「transient + non-exclusive」队列，而 kombu pidbox 的
# reply / gossip 队列默认正是该类型（durable=False, exclusive=False）；worker 启动
# 时 mingle/gossip 会声明它们并触发 541 INTERNAL_ERROR。将 pidbox 队列改为 durable
# （非 transient，broker 必然允许），不动 broker 策略本身。
_mb = celery_app.control.mailbox
_mb.queue_durable = True
_mb.__dict__.pop("reply_queue", None)  # 清除 cached_property 缓存以便重新生成

# gossip bootstep 会声明一个 transient 事件队列（kombu events Receiver），
# 同样被 RabbitMQ 新版拒绝。单 worker 场景不需要 worker 间 gossip/选举，禁用之。
# 任务消费由 Tasks bootstep 负责，不受影响。仅对 RabbitMQ 4.x 必需，对 3.x 无害。
from celery.worker.consumer import Consumer as _WorkerConsumer

_WorkerConsumer.Blueprint.default_steps = [
    s for s in _WorkerConsumer.Blueprint.default_steps if "gossip" not in s
]

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
    # 关闭 remote control：避免 worker 启动时声明 pidbox reply 队列
    # （RabbitMQ 新版本默认禁止 transient non-exclusive 队列，会导致 worker 反复崩溃）
    worker_enable_remote_control=False,
    # 启动即重试连接 broker，避免 RabbitMQ 未就绪时 worker 直接退出
    broker_connection_retry_on_startup=True,
    # 任务路由：image/video/model 走各自专用队列；classify 不指定 queue，
    # 走默认 celery 队列（本地 supervisord 与 k8s worker 均监听该队列）。
    task_routes={
        "src.tasks.image_tasks.*": {"queue": "image_queue"},
        "src.tasks.video_tasks.*": {"queue": "video_queue"},
        "src.tasks.model_tasks.*": {"queue": "model_queue"},
    },
    # 定时任务（原 src.core.celery_config 持有，合并后统一到 canonical app）
    beat_schedule={
        "cleanup-expired-tasks": {
            "task": "src.tasks.cleanup.cleanup_expired_tasks",
            "schedule": crontab(minute=0, hour="*/6"),
        },
        "sync-images": {
            "task": "src.tasks.image_tasks.sync_images_task",
            "schedule": crontab(minute=0, hour="*/4"),
        },
    },
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
