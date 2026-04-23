# Character Classification System

## 🎯 System Introduction

The Character Classification System is an AI-based image recognition tool specifically designed to identify characters from games and anime. The system uses advanced deep learning techniques to quickly and accurately identify characters in uploaded images, with support for end-to-end character detection workflows.

## ✨ Key Features

- **Image/Video Recognition**: Supports multiple formats for upload
- **Multi-role Detection**: Automatically detects and identifies multiple characters in a single image
- **High Accuracy**: Uses multiple models including MobileNetV2, EfficientNet-B0, EfficientNet-B3, and ResNet50
- **DeepDanbooru Integration**: Improves classification with anime tag recognition
- **Attribute Prediction**: Predicts character attributes (hair color, eye color, clothing)
- **Real-time Feedback**: Provides recognition confidence and detailed results
- **User-friendly Interface**: Intuitive web interface with model selection
- **API Support**: RESTful API for batch processing and multi-role detection
- **Log Fusion**: Builds new models from classification logs
- **End-to-End Workflow**: Complete process from data collection to model training
- **Memory Optimization**: Dynamic model loading/unloading, singleton pattern, reduced memory usage
- **Layered Deployment Architecture**: Supports deploying services on different servers for better scalability

## 📊 Model Information

### Supported Models

| Model | Input Size | Training Epochs | Batch Size | Learning Rate | Test Accuracy |
|-------|------------|-----------------|------------|---------------|---------------|
| MobileNetV2 | 224x224 | 50 (early stopping) | 32 | 0.001 | 94.00% |
| EfficientNet-B0 | 224x224 | 50 (early stopping) | 32 | 0.001 | 95.20% |
| EfficientNet-B3 | 300x300 | 50 (early stopping) | 32 | 0.001 | 96.80% |
| ResNet50 | 224x224 | 50 (early stopping) | 32 | 0.001 | 94.80% |


### Performance
| Model | Test Accuracy | Precision | Recall | F1-Score | Inference Speed (FPS) |
|-------|---------------|-----------|--------|----------|-----------------------|
| MobileNetV2 | 94.00% | 90.55% | 87.99% | 88.60% | 379.34 |
| EfficientNet-B0 | 95.20% | 92.10% | 90.50% | 91.30% | 298.45 |
| EfficientNet-B3 | 96.80% | 94.30% | 93.10% | 93.70% | 187.60 |
| ResNet50 | 94.80% | 91.20% | 89.70% | 90.40% | 256.78 |

## 🚀 Quick Start

### Environment Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- PyTorch
- Transformers
- Ultralytics (YOLOv8)
- Faiss
- EfficientNet-B0
- Requests (for DeepDanbooru API integration)
- Node.js 16+ (for frontend)

### Install Dependencies

```bash
# Install FastAPI and Uvicorn
pip3 install fastapi uvicorn python-multipart

# Install other dependencies
pip3 install torch torchvision transformers ultralytics faiss-cpu Pillow efficientnet_pytorch requests
```

### Start System

#### 1. Start Model Service

```bash
# Start model service
python3 -m uvicorn src.services.model_service.app:app --host 0.0.0.0 --port 8888
```

The model service will run at `http://127.0.0.1:8888`.

#### 2. Start Backend API Service

```bash
# Start backend API service
python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The backend API service will run at `http://127.0.0.1:8000`.

#### 3. Start Frontend Service

```bash
# Enter frontend directory
cd src/frontend

# Install dependencies (first run)
npm install

# Start Next.js frontend application
npm run dev
```

The frontend service will run at `http://localhost:3001`.

## 📁 Project Structure

```
anime_role_detect/
├── data/                  # Dataset directory
├── models/                # Model storage directory
├── src/                   # Source code
│   ├── api/               # API service code
│   ├── core/              # Core functionality
│   ├── frontend/          # Frontend code
│   │   ├── app/           # Next.js application
│   │   ├── components/     # React components
│   │   └── pages/          # Next.js pages
│   ├── middleware/        # Middleware
│   ├── services/          # Service layer
│   │   ├── model_service/  # Model service
│   │   ├── auth_service/   # Authentication service
│   │   └── cache_service/  # Cache service
│   ├── config/            # Configuration files
│   ├── scripts/           # Utility scripts
│   └── utils/             # Utility functions
├── docs/                  # Detailed documentation
├── cache/                 # Cache directory
├── auto_spider_img/       # Auto spider for images
├── README.md              # Quick start guide
└── README.zh.md           # Chinese documentation
```

## 🏗️ System Architecture

### Layered Architecture

The system adopts a layered architecture design, separating different functional modules to improve system maintainability and scalability.

```mermaid
flowchart TD
    subgraph Frontend Layer
        A[Web Interface] --> B[Next.js Application]
        B --> C[API Client]
        B --> D[User Authentication]
    end
    
    subgraph API Layer
        E[API Service] --> F[Request Handler]
        F --> G[Authentication Middleware]
        G --> H[Cache Manager]
        H --> I[Response Builder]
    end
    
    subgraph Service Layer
        J[Model Service] --> K[Feature Extraction]
        K --> L[Model Inference]
        L --> M[Attribute Prediction]
        N[Auth Service] --> O[Token Management]
        P[Cache Service] --> Q[Redis Integration]
    end
    
    subgraph Core Layer
        R[Preprocessing] --> S[Classification]
        S --> T[Tagging]
        T --> U[Keypoint Detection]
        V[NSFW Detection] --> W[Content Filtering]
    end
    
    subgraph Data Layer
        X[Spider System] --> Y[URL Collection]
        Y --> Z[Image Download]
        Z --> AA[Data Storage]
    end
    
    C --> E
    D --> N
    I --> J
    J --> R
    W --> S
    AA --> S
```

### Data Flow Diagram

The data flow through the system follows a clear path from image upload to result generation:

```mermaid
sequenceDiagram
    participant User as User
    participant Frontend as Frontend
    participant API as API Service
    participant Auth as Auth Service
    participant ModelService as Model Service
    participant Core as Core Services
    participant NSFW as NSFW Detection
    
    User->>Frontend: Login
    Frontend->>API: POST /api/auth/login
    API->>Auth: Verify Credentials
    Auth->>API: Return Token
    API->>Frontend: Return Token
    
    User->>Frontend: Upload Image
    Frontend->>API: POST /api/classify (with token)
    API->>Auth: Validate Token
    Auth->>API: Token Valid
    API->>ModelService: Request Prediction
    ModelService->>Core: Preprocess Image
    Core->>NSFW: NSFW Content Detection
    NSFW->>Core: Return NSFW Result
    Core->>Core: Extract Features
    Core->>Core: Classify Character
    Core->>ModelService: Return Features & Prediction
    ModelService->>API: Return Results
    API->>Frontend: Return JSON Response
    Frontend->>User: Display Results
```

### Architecture Overview

The system architecture is designed to support distributed deployment, with each service able to run independently on different servers:

1. **Frontend Layer**:
   - Next.js application with responsive design and dark mode support
   - User interface for image upload, model selection, and result display
   - User authentication interface with login form
   - API client for communicating with backend services
   - Real-time feedback and progress indicators

2. **API Layer**:
   - FastAPI-based RESTful API running on port 8000
   - Request handling and response building
   - Authentication middleware for token validation
   - Cache management for improved performance
   - Error handling and logging
   - Route management for different endpoints

3. **Service Layer**:
   - **Model Service** (port 8888):
     - Core prediction functionality
     - Feature extraction and model inference
     - Attribute prediction and text detection
     - Model loading and management
   - **Auth Service**:
     - User authentication and authorization
     - JWT token generation and validation
     - User role management
   - **Cache Service**:
     - Redis integration for distributed caching
     - Local memory caching for frequently accessed data
     - Cache invalidation strategies

4. **Core Layer**:
   - Preprocessing and image validation
   - Classification models and algorithms
   - Tagging and keypoint detection
   - NSFW detection for content filtering
   - Image processing utilities

5. **Data Layer**:
   - Spider system for data collection
   - URL collection and filtering
   - Image download and storage
   - Data organization and management
   - Dataset preparation for model training

This layered architecture allows for easy scaling and maintenance, with each layer responsible for specific functionality. The services communicate through well-defined API interfaces, enabling independent deployment and scaling.

### System Detection Flow

The system detection flow has been updated to include NSFW detection and improved data management:

1. **Image Upload**:
   - User uploads image through web interface or API
   - System validates image format and size
   - Temporary files are stored in a dedicated directory

2. **Preprocessing**:
   - Image is compressed and normalized
   - System checks for NSFW content using dedicated model
   - Results are stored for further processing

3. **Character Detection**:
   - System automatically detects if there are multiple characters
   - Appropriate detection mode is selected based on character count

4. **Character Classification**:
   - Each character is classified using the selected model
   - Feature extraction and model inference are performed
   - Confidence scores are calculated for each prediction

5. **Attribute Prediction**:
   - Character attributes (hair color, eye color, clothing) are predicted
   - Results are integrated with classification data

6. **Result Generation**:
   - Results are compiled and formatted
   - NSFW detection results are included in the response
   - Data is returned to the user through the API

7. **Data Collection**:
   - Spider system collects character images from various sources
   - URLs are filtered and stored in dedicated directories
   - Images are downloaded and organized for model training

This updated flow ensures that the system can effectively handle NSFW content and provides a more comprehensive detection process.

## �🌐 Usage

### Web Interface

1. Open your browser and visit `http://localhost:3000`
2. Select a model from the dropdown menu (default: EfficientNet-B0)
3. Check the "Multi-role Detection" box if you want to detect multiple characters in the image
4. Upload an image of the character(s) you want to identify
5. Wait for the system to analyze the image
6. View the recognition result and confidence

### Multi-role Detection

When using multi-role detection:
- The system will automatically detect all characters in the image
- Each character will be identified with its own confidence score
- The results will show the number of detected characters and their positions
- Processing time may be longer for images with multiple characters

### How Automatic Detection Works

1. **Image Upload**: User uploads an image through the web interface or API
2. **Preprocessing**: Image is compressed and validated
3. **Character Detection**: System automatically detects if there are multiple characters in the image
4. **Detection Mode Selection**: Based on the number of detected characters, the system selects the appropriate detection mode
5. **Character Classification**: Each character is classified using the selected model
6. **Attribute Prediction**: Character attributes (hair color, eye color, clothing) are predicted
7. **Text Detection**: Any text in the image is detected
8. **Result Generation**: Results are compiled and returned to the user

### API Call

```bash
# Basic usage (auto-detection)
curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:8000/api/classify

# With model and attributes
curl -X POST -F "file=@path/to/image.jpg" -F "use_model=true" -F "use_attributes=true" -F "model_name=efficientnet_b0" http://127.0.0.1:8000/api/classify

# Multi-role detection (force)
curl -X POST -F "file=@path/to/image.jpg" -F "model_name=efficientnet_b0" http://127.0.0.1:8000/api/classify/multi-role
```

## 📚 Documentation

For detailed technical documentation, please refer to the `docs/` directory:

- **docs/technical_guide.md**: Complete technical documentation

## 🤝 Contribution

Welcome to submit Issues and Pull Requests to improve system performance and functionality.

## 📄 License

This project is open source under the MIT license.

## 📞 Contact

- Email: zhaoqi.cao@icloud.com
- GitHub: https://github.com/caozhaoqi/anime-role-detect
