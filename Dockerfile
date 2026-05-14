# Multi-stage Dockerfile for Anime Role Detection API Service
# Stage 1: Builder stage for installing dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies to a separate directory for later copying
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# Stage 2: Runtime stage with minimal dependencies
FROM python:3.9-slim AS runtime

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-highgui4.5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed dependencies from builder stage
COPY --from=builder /app/deps /usr/local/lib/python3.9/site-packages

# Copy only necessary source files
COPY src/ ./src/
COPY models/ ./models/
COPY temp/ ./temp/
COPY logs/ ./logs/

# Set environment variables
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

# Create cache and log directories
RUN mkdir -p /app/cache/huggingface /app/cache/keras /app/logs /app/temp

# Expose port
EXPOSE 8000

# Start API service
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
