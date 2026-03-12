# Character Classification System

## 🎯 System Introduction

The Character Classification System is an AI-based image recognition tool specifically designed to identify characters from games and anime. The system uses advanced deep learning techniques to quickly and accurately identify characters in uploaded images, with support for end-to-end character detection workflows.

## ✨ Key Features

- **Image/Video Recognition**: Supports multiple formats for upload
- **High Accuracy**: Uses CLIP model and Faiss indexing
- **DeepDanbooru Integration**: Improves classification with anime tag recognition
- **Attribute Prediction**: Predicts character attributes (hair color, eye color, clothing)
- **Real-time Feedback**: Provides recognition confidence and detailed results
- **User-friendly Interface**: Intuitive web interface
- **API Support**: RESTful API for batch processing
- **Log Fusion**: Builds new models from classification logs
- **End-to-End Workflow**: Complete process from data collection to model training

## 🚀 Quick Start

### Environment Requirements

- Python 3.7+
- Flask
- PyTorch
- Transformers
- Ultralytics (YOLOv8)
- Faiss
- EfficientNet-B0
- Requests (for DeepDanbooru API integration)

### Install Dependencies

```bash
# Install Flask
pip3 install flask

# Install other dependencies
pip3 install torch torchvision transformers ultralytics faiss-cpu Pillow efficientnet_pytorch requests
```

### Start System

#### 1. Start Backend Service

```bash
# Start backend application
python3 src/backend/api/app.py
```

The backend service will run at `http://127.0.0.1:8000`.

#### 2. Start Frontend Service

```bash
# Enter frontend directory
cd frontend

# Install dependencies (first run)
npm install

# Start Next.js frontend application
npm run dev
```

The frontend service will run at `http://localhost:3000`.

## 📁 Project Structure

```
anime_role_detect/
├── data/                  # Dataset directory
├── models/                # Model storage directory
├── src/                   # Source code
│   ├── backend/           # Backend code
│   ├── core/              # Core functionality
│   ├── data/              # Data-related code
│   ├── frontend/          # Frontend code
│   ├── models/            # Model-related code
│   ├── config/            # Configuration files
│   ├── scripts/           # Utility scripts
│   └── utils/             # Utility functions
├── docs/                  # Detailed documentation
├── cache/                 # Cache directory
├── auto_spider_img/       # Auto spider for images
├── README.md              # Quick start guide
└── README.zh.md           # Chinese documentation
```

## 🌐 Usage

### Web Interface

1. Open your browser and visit `http://localhost:3000`
2. Upload an image of the character you want to identify
3. Wait for the system to analyze the image
4. View the recognition result and confidence

### API Call

```bash
# Basic usage
curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:8000/api/classify

# With model and attributes
curl -X POST -F "file=@path/to/image.jpg" -F "use_model=true" -F "use_attributes=true" http://127.0.0.1:8000/api/classify
```

## � Documentation

For detailed technical documentation, please refer to the `docs/` directory:

- **docs/technical_guide.md**: Complete technical documentation

## 🤝 Contribution

Welcome to submit Issues and Pull Requests to improve system performance and functionality.

## 📄 License

This project is open source under the MIT license.

## 📞 Contact

- Email: zhaoqi.cao@icloud.com
- GitHub: https://github.com/caozhaoqi/anime-role-detect

---

**© 2026 Character Classification System** - Making character recognition simple!

