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
| Monitoring | http://localhost:8888 | 8888 |
| Supervisor Dashboard | http://localhost:9001 | 9001 |
| RabbitMQ Management | http://localhost:15672 | 15672 |

> Infrastructure ports: Redis 6379, MySQL 3306, RabbitMQ 5672, fluent-bit 2020 (Docker Compose only).

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
│   │   └── routes/         # API routes (classification, auth, collector, search, video,
│   │                       #   cleaning, history, models, onnx_inference, async_inference,
│   │                       #   tracing, version, health, misc)
│   ├── services/           # Microservices
│   │   ├── api_gateway/    # API Gateway (port 8080, aggregates Swagger docs)
│   │   ├── model_service/  # Model Service (port 8000, includes keypoint_worker)
│   │   ├── multimedia_service/  # Multimedia Service (port 8002, video rendering)
│   │   ├── search_service/ # Search Service + worker (port 8003, CLIP+FAISS)
│   │   ├── inference_worker/   # CLIP inference worker
│   │   ├── inference_queue/    # Inference queue manager (Redis/Memory fallback)
│   │   ├── cache_service/  # Redis Cache Service
│   │   ├── model/          # Business model services (classify/recognize/NSFW/multi-model/version)
│   │   ├── processor/      # Model loaders / image processors / preprocessors
│   │   ├── support/        # Database service and support layer
│   │   ├── training/       # Training-related services
│   │   └── notification_service.py  # Feishu notifications
│   ├── core/               # Core capabilities
│   │   ├── classification/ # EfficientNet/MobileNet/DeepDanbooru classification
│   │   ├── detection/      # YOLO multi-role detection + anime_face_detector
│   │   ├── recognition/    # CLIP/ArcFace open-set recognition + feature store
│   │   ├── tagging/        # WD-ViT-Tagger + DeepDanbooru tagging
│   │   ├── keypoint/       # MediaPipe keypoints
│   │   ├── ocr/            # EasyOCR
│   │   ├── feature_extraction/  # Feature extraction (incl. CoreML)
│   │   ├── log_fusion/     # Log fusion
│   │   ├── preprocessing/  # Image/data preprocessors
│   │   ├── config/         # Configuration (ServiceConfig / DeviceManager)
│   │   ├── cache/          # Cache abstractions
│   │   ├── logging/        # Structured logging (loguru JSON)
│   │   └── ...             # error / exception / feedback / version / utils
│   ├── data/               # Data collection / cleaning / augmentation / search index
│   ├── data_pipeline/      # Data cleaning pipeline + active_learning + Streamlit webui
│   ├── data_collection/    # (Legacy) keyword-based collector entry
│   ├── models/             # Database models + training / evaluation / prediction / deployment
│   ├── tasks/              # Celery tasks (classify/image/video/model/cleanup)
│   ├── utils/              # Shared utilities (image, http, concurrency, memory, monitoring, config)
│   ├── middleware/         # HTTP middleware (auth_enhanced / monitoring / tracing)
│   ├── frontend/           # Frontend (Next.js 15 + React 18 + TypeScript App Router)
│   ├── run/                # Service management / monitor dashboard / launch scripts
│   ├── cache/              # HuggingFace / Keras model cache directories
│   └── static/             # Static assets
├── models/                 # Model weights (git-ignored)
├── tests/                  # Test suites (unit / integration / model / workflow / regression / performance / benchmark)
├── docs/                   # Documentation (architecture / deployment / training / blog / testing / technical_challenges)
├── scripts/                # Utility scripts (k8s, monitoring, data_*, model_evaluation, coreml, detection, ...)
│   └── skillhub/           # ⚠️ Archived experiment sub-project (88MB, not referenced)
├── archived/               # Historical / broken modules (spider_image_system, arona, ...)
├── deployment/             # Docker deployment files (11 Dockerfile.* + nginx + grafana)
├── k8s/                    # Kustomize (base/ + overlays/ci/)
├── config/                 # Config templates (config.ini / config.py)
├── supervisord.conf        # Process manager configuration (12 programs)
├── docker-compose.yml      # Docker Compose configuration (13 services)
├── Dockerfile              # Backend Dockerfile (root)
├── Dockerfile.model        # Model Service Dockerfile (root)
├── requirements.txt        # Full dependencies
├── requirements-base.txt   # Base dependencies (for base image)
├── requirements-ml.txt     # ML dependencies
├── requirements-model-service.txt
├── requirements-scripts.txt
├── requirements-dev.txt    # Development dependencies
├── pyproject.toml          # Project configuration (v2.3.0, authoritative version source)
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
| `/api/classify/async` | POST | Async classification (task queue) |
| `/api/search/image` | POST | Reverse image search (CLIP+FAISS) |
| `/api/video/recognize` | POST | Video recognition |
| `/api/collect` | POST | Data collection task |
| `/api/cleaning` | POST | Data cleaning task |
| `/api/history` | GET | Recognition history |
| `/api/models` | GET | Model info & version |
| `/api/onnx/infer` | POST | ONNX inference |
| `/api/health` | GET | Health check |
| `/api/services` | GET | Service status |
| `/api/auth/login` | POST | User login |
| `/api/auth/refresh` | POST | Refresh token |
| `/api/version` | GET | Version info |
| `/metrics` | GET | Prometheus metrics |

> Full route definitions in [src/api/routes/](src/api/routes/). Gateway aggregated docs: `http://localhost:8080/docs`.

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

- **docker-compose.yml**: Multi-service orchestration (13 services) with Redis, MySQL, RabbitMQ, fluent-bit, and all application services
- **Root Dockerfile**: Backend service image
- **Root Dockerfile.model**: Model service image
- **deployment/**: 11 Dockerfiles (base / ml-base / api-service / api-gateway / model-service / multimedia-service / search-service / search-worker / inference-worker / monitoring / frontend) + nginx.conf + grafana dashboard
- **Resource limits**: model-service 4G/4cpu, others 256M-1.5G (compressed in compose file)

## 📊 Model Performance

### Latest Benchmark Results

**Production model**: `efficientnet_b3` (`models/efficientnet_b3/model_best.pth`), 51 classes, 256×256 input, 45.99 MB, 11.9M params, evaluated on Apple MPS.

| Metric | Value |
|--------|-------|
| Top-1 Accuracy (no-overlap split, **honest per-image**) | **82.65%** |
| Top-1 Accuracy (same-image test, leaked upper bound) | 84.00% |
| Top-5 Accuracy | **93.96%** |
| Macro-F1 (same-image test) | **0.8401** |
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
| EfficientNet-B3 (production) | 51 | **82.65%** | Honest per-image (no-overlap split); 84.00% is the same-image leaked upper bound |
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
- `docs/architecture/` - Project structure & architecture design
- `docs/deployment/` - Deployment guides (Kubernetes, Ubuntu)
- `docs/training/` - Model training guides + data leakage analysis
- `docs/blog/` - Technical blog posts
- `docs/testing/` - Testing guides
- `docs/technical_challenges/` - Technical challenges & solutions
- `docs/system_design.md` / `docs/system_design_perf.md` - System design & performance optimization plan

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- How to submit bug reports and feature requests
- Code style guidelines
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Version**: v2.3.0 | **Last Updated**: Aug 2026 | **Maintainer**: ARD Team

---

**Topics**: anime, character-recognition, image-classification, deep-learning, python-api, computer-vision, yolov8, nextjs, docker, microservices