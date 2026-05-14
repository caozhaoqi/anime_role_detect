#!/bin/bash
# Celery 启动脚本

# 设置 Python 路径
export PYTHONPATH="/Users/caozhaoqi/PycharmProjects/anime_role_detect:$PYTHONPATH"

# 设置 Redis 配置
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_DB="0"

# 启动参数
COMMAND="$1"
QUEUES="${2:-image_queue,video_queue,model_queue}"

case $COMMAND in
    worker)
        echo "启动 Celery Worker..."
        python3 -m celery -A src.core.celery_config worker -Q $QUEUES --loglevel=info
        ;;
    beat)
        echo "启动 Celery Beat..."
        python3 -m celery -A src.core.celery_config beat --loglevel=info
        ;;
    flower)
        echo "启动 Flower 监控..."
        python3 -m celery -A src.core.celery_config flower --port=5555
        ;;
    all)
        echo "启动所有服务..."
        # 启动 Worker
        python3 -m celery -A src.core.celery_config worker -Q $QUEUES --loglevel=info &
        # 等待一秒
        sleep 1
        # 启动 Beat
        python3 -m celery -A src.core.celery_config beat --loglevel=info &
        # 启动 Flower
        python3 -m celery -A src.core.celery_config flower --port=5555 &
        # 等待所有进程
        wait
        ;;
    *)
        echo "Usage: $0 [worker|beat|flower|all] [queues]"
        echo "示例: $0 worker image_queue,video_queue"
        exit 1
        ;;
esac
