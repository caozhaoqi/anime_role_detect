# 使用官方Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache/huggingface
ENV KERAS_HOME=/app/cache/keras
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_PASSWORD=
ENV REDIS_DB=0
ENV CACHE_TTL=3600
ENV LOCAL_CACHE_SIZE=1000
ENV USE_MODEL_SERVICE=true
ENV MODEL_SERVICE_HOST=model-service
ENV MODEL_SERVICE_PORT=8888
ENV RABBITMQ_HOST=rabbitmq
ENV RABBITMQ_PORT=5672
ENV RABBITMQ_USER=guest
ENV RABBITMQ_PASSWORD=guest
ENV RABBITMQ_VHOST=/
ENV QUEUE_NAME=anime_role_detect
ENV EXCHANGE_NAME=anime_role_detect_exchange

# 创建缓存目录
RUN mkdir -p /app/cache/huggingface /app/cache/keras /app/logs

# 暴露端口
EXPOSE 8000

# 启动API服务
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
