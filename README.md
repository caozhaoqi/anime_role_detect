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
├── src/                    # Source code (editable install: pip install -e .)
│   ├── api/                # Backend API service (FastAPI, port 8001)
│   │   └── routes/         # API route definitions (classify, auth, collector, ...)
│   ├── services/           # Microservices
│   │   ├── api_gateway/    # API Gateway (FastAPI, port 8080)
│   │   ├── model_service/  # Model Service (port 8000)
│   │   ├── multimedia_service/  # Multimedia Service (port 8002)
│   │   ├── search_service/ # Search Service + worker (port 8003)
│   │   ├── inference_worker/   # CLIP inference worker
│   │   ├── cache_service/  # Redis Cache Service
│   │   ├── video_service/  # Video Recognition Service
│   │   ├── messaging/      # Message queue (aio_pika)
│   │   └── processor/      # Model loaders / image processors
│   ├── core/               # Core functionality (classification, detection, tagging,
│   │                       #   recognition, ocr, logging, config, cache, ...)
│   ├── data/               # Data collection & preprocessing pipelines
│   ├── data_pipeline/      # Data cleaning / building / webui pipelines
│   ├── data_collection/    # (Legacy) keyword-based collector entry
│   ├── models/             # Training / evaluation / prediction modules
│   ├── tasks/              # Celery tasks (classify, image, video, model, cleanup)
│   ├── utils/              # Shared utilities (image, http, concurrency, monitoring)
│   ├── middleware/         # HTTP middleware
│   ├── frontend/           # Frontend (Next.js 15 + React 18 + TypeScript)
│   └── run/                # Service management & monitor dashboard
├── models/                 # Model weights (git-ignored)
├── tests/                  # Test suites (unit / integration / model / workflow)
├── docs/                   # Documentation (architecture, deployment, training, blog)
├── scripts/                # Utility scripts (k8s, monitoring, data_*, evaluation, ...)
│   └── skillhub/           # ⚠️ Archived experiment sub-project (88MB, not referenced)
├── archived/               # Historical / broken modules (spider_image_system, arona, ...)
├── deployment/             # Kubernetes & Docker deployment files
├── k8s/                    # Kustomize overlays (base / ci) + local registry helpers
├── config/                 # Config templates
├── supervisord.conf        # Process manager configuration (11 programs)
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Backend Dockerfile
├── Dockerfile.model        # Model Service Dockerfile
├── requirements-base.txt   # Base dependencies (for base image)
├── requirements-ml.txt     # ML dependencies
├── requirements-model-service.txt
├── requirements-scripts.txt
├── requirements-dev.txt    # Development dependencies
├── pyproject.toml          # Project configuration (v2.3.0)
└── .env.example            # Environment template
```

> **Note on unused / legacy code (2026-07-30 cleanup)**
> - Dead module `src/models/training/convert_model_format.py` (syntax-broken, unreferenced) → moved to `archived/broken_modules/`.
> - `scripts/skillhub/` is a legacy experiment sub-project (bundled venv, not referenced anywhere). Kept for history; excluded from builds.
> - `src/run/start_all.py`, `start_all_stable.py`, `application.py`, `start_core.py` are legacy/alternative launchers (not used by supervisord/k8s/docker). Kept as dev tools.
> - Removed ~30 unused imports / unused variables across `src/` (low-risk lint cleanup).
> - Runtime artifacts (`logs/`, `data/`, `models/`, `*.db`, `dump.rdb`, caches) are git-ignored.

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

## 📊 Model Performance

### Latest Benchmark Results (2026-07-28, `scripts/model_evaluation/benchmark_results.json`)

> ⚠️ **Data note**: the test set (`final_dataset`, 1,275 images, 51 classes × 25) is currently **sampled from the same source as the training set**. The reported accuracy reflects training-state performance; true generalization on an independent dataset is **not yet validated** and is expected to be lower. See `docs/architecture/PROJECT_STRUCTURE.md` for known issues.

**Production model**: `efficientnet_b3` (`models/efficientnet_b3/model_best.pth`), 51 classes, 256×256 input, 45.99 MB, 11.9M params, evaluated on Apple MPS.

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **84.00%** |
| Top-5 Accuracy | **93.96%** |
| Macro-F1 | **0.8401** |
| Single-image Latency | **29.04 ms** (34.44 FPS) |
| Batch (32) Throughput | **31.11 FPS** |
| First Request Latency | **< 500ms** (with warm-up) |

**Weakest classes** (accuracy): `silver_wolf` 64%, `Klee` / `aglaea` / `clorinde` / `kafka` 68%.
**Most confused pair**: `clorinde` → `Furina`.

### Multi-role Detection (YOLOv8n)

`yolov8n.pt` is the COCO-pretrained baseline (6.25 MB, 3.15M params). It is **not fine-tuned** on anime characters — avg confidence 0.444, ~4 FPS on MPS. Fine-tuning is pending (see known issues).

### Model Comparison (reference, training-state)

| Model | Classes | Top-1 | Note |
|-------|---------|-------|------|
| EfficientNet-B3 (production) | 51 | **84.00%** | Current model |
| EfficientNet-B0 / MobileNetV2 / ResNet50 | — | — | Earlier experiments, see `docs/blog/10_training_and_evaluation.md` |

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

# Run model benchmark (produces scripts/model_evaluation/benchmark_results.json)
python scripts/model_evaluation/run_benchmark.py
```


## 📚 Documentation

For detailed documentation:
- `docs/architecture/PROJECT_STRUCTURE.md` - Project structure & known issues
- `docs/deployment/` - Deployment guides (Kubernetes, Ubuntu)
- `docs/training/` - Model training guides
- `docs/blog/` - Technical blog posts

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- How to submit bug reports and feature requests
- Code style guidelines
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Version**: v2.3.0 | **Last Updated**: July 2026 | **Maintainer**: ARD Team

---

**Topics**: anime, character-recognition, image-classification, deep-learning, python-api, computer-vision, yolov8, nextjs, docker, microservices