# Character Classification System

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An AI-powered image recognition system designed to identify characters from games and anime.

## ✨ Features

- **Multi-format Recognition**: Supports images and videos
- **Multi-role Detection**: Identify multiple characters in single image (YOLOv8/v10 integration)
- **High Accuracy**: Powered by MobileNetV2, EfficientNet-B0/B3, ResNet50
- **DeepDanbooru Integration**: Enhanced tagging capabilities
- **Attribute Prediction**: Hair color, eye color, clothing attributes
- **RESTful API**: Batch processing support
- **Log Fusion**: Build new models from classification logs
- **Layered Architecture**: Distributed deployment ready
- **Model Warm-up**: Reduced first-request latency
- **Request Debouncing/Throttling**: Prevent duplicate submissions
- **Image Compression**: Optimized upload bandwidth
- **Redis Cache**: Reduce redundant computations
- **Feishu Notifications**: Real-time progress updates
- **Token Auto-refresh**: Seamless authentication

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 16GB+ RAM (required for model loading)
- NVIDIA GPU (recommended for inference speed)
- Redis Server (for caching)
- Docker & Docker Compose (for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/ard-team/anime_role_detect.git
cd anime_role_detect

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-base.txt
pip install -r requirements-ml.txt  # For model training/inference
pip install -r requirements-dev.txt  # For development
pip install supervisor  # For process management

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Running with Supervisor (Recommended)

```bash
# Start Redis (required for caching)
redis-server &

# Start all services using supervisord
supervisord -c supervisord.conf

# Check service status
supervisorctl status

# Stop all services
supervisorctl stop all
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build -d

# Check container status
docker-compose ps

# View logs
docker-compose logs -f <service_name>

# Stop services
docker-compose down

# k8s deployment

# 1. 构建所有镜像
./scripts/k8s/build_k8s_images.sh

# 2. 部署到 K8s（权威源：k8s/base/，详见 k8s/README.md）
kubectl apply -k k8s/base/

# 3. 查看部署状态
kubectl get pods -n anime-role-detect
```

### Service Access

| Service | URL | Port |
|---------|-----|------|
| Frontend | http://localhost:3000 | 3000 |
| API Gateway | http://localhost:8080 | 8080 |
| Model Service | http://localhost:8000 | 8000 |
| API Service | http://localhost:8001 | 8001 |
| Multimedia Service | http://localhost:8002 | 8002 |
| Search Service | http://localhost:8003 | 8003 |
| Supervisor Dashboard | http://localhost:9001 | 9001 |

### API Documentation
- **Swagger Docs**: `http://localhost:8080/docs`
- **Redoc Docs**: `http://localhost:8080/redoc`

### Default Credentials
- **Username**: `admin` / `user`
- **Password**: Set via environment variables `ADMIN_PASSWORD` and `USER_PASSWORD`
- **Note**: Default passwords are auto-generated on first startup if not set

## 📁 Project Structure

```
anime_role_detect/
├── src/                    # Source code
│   ├── api/                # Backend API (port 8001)
│   ├── services/           # Microservices
│   │   ├── api_gateway/    # API Gateway (port 8080)
│   │   ├── model_service/  # Model Service (port 8000)
│   │   ├── multimedia_service/  # Multimedia Service (port 8002)
│   │   ├── search_service/ # Search Service (port 8003)
│   │   ├── cache_service/  # Redis Cache Service
│   │   └── video_service/  # Video Recognition Service
│   ├── core/               # Core functionality
│   ├── frontend/           # Frontend (Next.js)
│   └── run/                # Service management scripts
├── models/                 # Model weights
├── tests/                  # Test suites
├── docs/                   # Documentation
├── skillhub/               # Skill Hub module
├── scripts/                # Utility scripts (spider, data collection)
├── deployment/             # Kubernetes deployment files
├── supervisord.conf        # Process manager configuration
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Backend Dockerfile
├── Dockerfile.model        # Model Service Dockerfile
├── requirements-base.txt   # Base dependencies
├── requirements-ml.txt     # ML dependencies
├── requirements-dev.txt    # Development dependencies
├── pyproject.toml          # Project configuration
└── .env.example           # Environment template
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/classify` | POST | Image classification |
| `/api/classify/multi-role` | POST | Multi-character detection (YOLO) |
| `/api/search/image` | POST | Reverse image search |
| `/api/video/recognize` | POST | Video recognition |
| `/api/health` | GET | Health check |
| `/api/services` | GET | Service status |
| `/api/auth/login` | POST | User login |
| `/api/auth/refresh` | POST | Refresh token |

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection URL | redis://localhost:6379 |
| `JWT_SECRET` | JWT secret key | (required) |
| `JWT_EXPIRE_MINUTES` | Token expiration | 1440 (24h) |
| `MAX_IMAGE_SIZE` | Max upload size (MB) | 10 |
| `DEVICE` | Compute device (cpu/cuda/mps) | auto |

### Docker Configuration

The project includes comprehensive Docker support:

- **docker-compose.yml**: Multi-service orchestration with Redis, MySQL, RabbitMQ, and all application services
- **Dockerfile**: Multi-stage build for backend services
- **Dockerfile.model**: Optimized model service image
- **deployment/Dockerfile.frontend**: Frontend Next.js deployment with Nginx
- **docker_manager.py**: Helper script for Docker operations

## 📊 Model Performance

### Latest Benchmark Results (June 2026)

**Test Dataset**: 1,480 images across 74 character classes

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **93.92%** |
| Top-3 Accuracy | **96.15%** |
| Top-5 Accuracy | **96.89%** |
| Inference Speed | **85.74 FPS** |
| Latency per Image | **11.66ms** |
| First Request Latency | **< 500ms** (with warm-up) |
| Login API Response | **< 220ms** |

### Model Comparison

| Model | Accuracy | FPS |
|-------|----------|-----|
| MobileNetV2 | 94.00% | 379 |
| EfficientNet-B0 | 95.20% | 298 |
| EfficientNet-B3 (Optimized) | **93.92%** | **85.74** |
| ResNet50 | 94.80% | 257 |

**Current Production Model**: `efficientnet_b3_loli_optimized_v2_20260529_133654`

## 🔒 Security

- JWT authentication with secret key rotation
- Password hashing with bcrypt/sha256
- Rate limiting to prevent abuse
- Input validation and sanitization
- HttpOnly Cookie storage
- Content Security Policy (CSP) for XSS protection
- Automatic token refresh mechanism

## 🧪 Testing

### Automated Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run performance tests
python performance_test.py
```


## 📚 Documentation

For detailed documentation:
- `docs/technical_guide.md` - Technical specifications
- `docs/deployment/` - Deployment guides (Kubernetes, Ubuntu)
- `docs/training/` - Model training guides
- `skillhub/docs/` - Skill Hub documentation
- `docs/blog/` - Technical blog posts

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- How to submit bug reports and feature requests
- Code style guidelines
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Version**: v2.3 | **Last Updated**: July 2026 | **Maintainer**: ARD Team

---

**Topics**: anime, character-recognition, image-classification, deep-learning, python-api, computer-vision, yolov8, nextjs, docker, microservices