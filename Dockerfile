# Simple Dockerfile for Anime Role Detection
# 使用单一阶段构建，简化流程

FROM python:3.9-slim

WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/cache/huggingface

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 升级pip
RUN pip install --upgrade pip

# 复制依赖文件
COPY requirements.txt .

# 安装依赖（使用--quiet减少输出）
RUN pip install -r requirements.txt --quiet

# 创建非root用户
RUN useradd -m -u 1000 appuser
USER appuser

# 复制源代码
COPY src/ ./src/
COPY scripts/ ./scripts/

# 创建必要目录
RUN mkdir -p /app/cache/huggingface /app/logs /app/temp /app/data

# 暴露端口
EXPOSE 8000

# 默认命令
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]