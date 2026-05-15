# Stage 1: Builder stage
FROM python:3.9-slim AS builder

WORKDIR /app

# 安装构建基础依赖（用于编译某些 python 库）
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 建议：在 requirements.txt 中将 opencv-python 改为 opencv-python-headless
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.9-slim AS runtime

WORKDIR /app

# 【关键修改】：只安装 OpenCV 运行必须的系统级基础支撑库
# 不要在 apt 中安装 python3-opencv，因为它会引入大量无用的 GUI 依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 拷贝依赖
COPY --from=builder /app/deps /usr/local/lib/python3.9/site-packages

# 拷贝源码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests ./tests
COPY auto_spider_img ./auto_spider_img

# 环境变量设置（脱敏处理）
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/cache/huggingface \
    KERAS_HOME=/app/cache/keras \
    REDIS_HOST=redis \
    MODEL_SERVICE_HOST=model-service \
    RABBITMQ_HOST=rabbitmq

# 创建运行所需目录（解决之前提到的 logs 找不到的问题）
RUN mkdir -p /app/cache/huggingface /app/cache/keras /app/logs /app/temp

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]