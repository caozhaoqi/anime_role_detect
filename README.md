# Character Classification System

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
# Install dependencies
pip3 install -r requirements.txt

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
└── skillhub/               # Skill Hub module
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

| Model | Accuracy | FPS |
|-------|----------|-----|
| MobileNetV2 | 94.00% | 379 |
| EfficientNet-B0 | 95.20% | 298 |
| EfficientNet-B3 | 96.80% | 188 |
| ResNet50 | 94.80% | 257 |

## 📚 Documentation

For detailed documentation:
- `docs/technical_guide.md` - Technical specifications
- `docs/deployment/` - Deployment guides
- `docs/training/` - Model training guides
- `skillhub/docs/` - Skill Hub documentation

## 📄 License

MIT License

---

**Version**: v2.0 | **Last Updated**: May 2026