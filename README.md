# Character Classification System

![GitHub Actions](https://img.shields.io/github/actions/workflow/status/ard-team/anime_role_detect/ci-cd.yml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Coverage](https://img.shields.io/codecov/c/github/ard-team/anime_role_detect)
![Last Commit](https://img.shields.io/github/last-commit/ard-team/anime_role_detect)

An AI-powered image recognition system designed to identify characters from games and anime.

## ✨ Features

- **Multi-format Recognition**: Supports images and videos
- **Multi-role Detection**: Identify multiple characters in single image
- **High Accuracy**: Powered by MobileNetV2, EfficientNet-B0/B3, ResNet50
- **DeepDanbooru Integration**: Enhanced tagging capabilities
- **Attribute Prediction**: Hair color, eye color, clothing attributes
- **RESTful API**: Batch processing support
- **Log Fusion**: Build new models from classification logs
- **Layered Architecture**: Distributed deployment ready

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 16GB+ RAM (required for model loading)
- NVIDIA GPU (recommended for inference speed)

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

# Configure environment
cp .env.example .env
# Edit .env with your configuration

# Start services
python3 src/application.py start --core
python3 src/application.py start --services gateway
```

### Docker Deployment

```bash
docker-compose up --build -d
```

### API Gateway (Main Entry)
- **Gateway**: `http://localhost:8080`
- **API Docs**: `http://localhost:8080/docs`

## 📁 Project Structure

```
anime_role_detect/
├── src/                    # Source code
│   ├── api/                # Backend API (port 8001)
│   ├── services/           # Microservices
│   │   ├── api_gateway/    # API Gateway (port 8080)
│   │   ├── model_service/  # Model Service (port 8000)
│   │   └── multimedia/     # Multimedia Service (port 8002)
│   ├── core/               # Core functionality
│   └── frontend/           # Frontend (Next.js)
├── models/                 # Model weights
├── tests/                  # Test suites
├── docs/                   # Documentation
├── skillhub/               # Skill Hub module
├── scripts/                # Utility scripts
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
| `/api/classify/multi-role` | POST | Multi-character detection |
| `/api/search/image` | POST | Reverse image search |
| `/api/video/recognize` | POST | Video recognition |
| `/api/health` | GET | Health check |
| `/api/services` | GET | Service status |

## 📊 Model Performance

### Latest Benchmark Results (May 2026)

**Test Dataset**: 1,480 images across 74 character classes

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **93.92%** |
| Top-3 Accuracy | **96.15%** |
| Top-5 Accuracy | **96.89%** |
| Inference Speed | **85.74 FPS** |
| Latency per Image | **11.66ms** |

### Model Comparison

| Model | Accuracy | FPS |
|-------|----------|-----|
| MobileNetV2 | 94.00% | 379 |
| EfficientNet-B0 | 95.20% | 298 |
| EfficientNet-B3 (Optimized) | **93.92%** | **85.74** |
| ResNet50 | 94.80% | 257 |

**Current Production Model**: `efficientnet_b3_loli_optimized_v2_20260529_133654`

## 📚 Documentation

For detailed documentation:
- `docs/technical_guide.md` - Technical specifications
- `docs/deployment/` - Deployment guides
- `docs/training/` - Model training guides
- `skillhub/docs/` - Skill Hub documentation

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- How to submit bug reports and feature requests
- Code style guidelines
- Pull request process

## 🔒 Security

- JWT authentication with secret key rotation
- Password hashing with bcrypt/sha256
- Rate limiting to prevent abuse
- Input validation and sanitization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Version**: v2.0 | **Last Updated**: May 2026 | **Maintainer**: ARD Team

---

**Topics**: anime, character-recognition, image-classification, deep-learning, python-api, computer-vision
