# Stage 1: Builder stage for installing dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc \
    g++ \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY requirements-base.txt .
COPY requirements-ml.txt .

# Combine all requirements
RUN cat requirements.txt requirements-base.txt requirements-ml.txt > /tmp/all_requirements.txt

# Install dependencies to a separate directory for later copying
RUN pip install --no-cache-dir --target=/app/deps -r /tmp/all_requirements.txt

# Stage 2: Runtime stage with minimal dependencies
FROM python:3.9-slim AS runtime

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed dependencies from builder stage
COPY --from=builder /app/deps /usr/local/lib/python3.9/site-packages

# Copy source files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/cache/huggingface \
    KERAS_HOME=/app/cache/keras \
    REDIS_HOST=redis \
    MODEL_SERVICE_HOST=model-service \
    RABBITMQ_HOST=rabbitmq \
    DB_HOST=mysql

# Create necessary directories
RUN mkdir -p /app/cache/huggingface /app/cache/keras /app/logs /app/temp /app/data

# Expose port
EXPOSE 8000

# Default command - can be overridden per service
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]