# Multi-stage Dockerfile for Backend Services
# Stage 1: Builder stage for installing dependencies

FROM python:3.9-slim AS builder

WORKDIR /app

# Set environment variables for building
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc \
    g++ \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements files
COPY requirements.txt .
COPY requirements-base.txt .

# Combine requirements and filter out duplicates
RUN cat requirements.txt requirements-base.txt | sort | uniq > /tmp/all_requirements.txt

# Install dependencies to a separate directory for later copying
RUN pip wheel --wheel-dir=/app/wheels -r /tmp/all_requirements.txt
RUN pip install --no-cache-dir --target=/app/deps -r /tmp/all_requirements.txt

# Stage 2: Runtime stage with minimal dependencies
FROM python:3.9-slim AS runtime

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/cache/huggingface \
    KERAS_HOME=/app/cache/keras \
    REDIS_HOST=redis \
    MODEL_SERVICE_HOST=model-service \
    RABBITMQ_HOST=rabbitmq \
    DB_HOST=mysql

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Copy installed dependencies from builder stage
COPY --from=builder --chown=appuser:appuser /app/deps /home/appuser/.local/lib/python3.9/site-packages

# Copy source files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/cache/huggingface /app/cache/keras /app/logs /app/temp /app/data

# Expose port
EXPOSE 8000

# Default command - can be overridden per service
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]