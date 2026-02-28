# Character Classification System User Guide

## 🎯 System Introduction

The Character Classification System is an AI-based image recognition tool specifically designed to identify characters from games and anime. The system uses advanced deep learning techniques to quickly and accurately identify characters in uploaded images, now with support for end-to-end character detection workflows.

## ✨ Core Features

- **Image/Video Upload Recognition**: Supports multiple image and video formats for upload, automatically identifies game characters in images and videos
- **High Accuracy**: Uses CLIP model and Faiss indexing for high recognition accuracy
- **DeepDanbooru Integration**: Integrates DeepDanbooru for anime tag recognition to improve classification accuracy
- **Tag-assisted Inference**: Uses DeepDanbooru tags to adjust classification results, solving "Sameface Syndrome"
- **Real-time Feedback**: Provides recognition confidence and detailed results
- **User-friendly Interface**: Intuitive web interface with simple operation
- **API Support**: Provides RESTful API interface for batch processing
- **Log Fusion**: Supports fusing features from classification logs to build new models
- **End-to-End Workflow**: Complete character detection workflow from data collection to model training
- **EfficientNet-B0 Model**: Uses state-of-the-art EfficientNet-B0 for classification
- **Data Collection**: Supports automatic image collection via Bing Image Search API
- **Dataset Splitting**: Automatically splits collected data into training and validation sets

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

# Install other dependencies (if not already installed)
pip3 install torch torchvision transformers ultralytics faiss-cpu Pillow efficientnet_pytorch requests
```

### Start System

#### 1. Start Backend Service

```bash
# Start Flask backend application
python3 src/backend/web/web_app.py
```

The backend service will run at `http://127.0.0.1:5001`.

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
│   ├── augmented_dataset/ # Augmented dataset
│   ├── split_dataset/     # Split training/validation data
│   └── all_characters/    # All character images
├── models/                # Model storage directory
├── src/                   # Source code
│   ├── backend/           # Backend code
│   │   ├── api/           # API implementation
│   │   └── web/           # Web interface
│   ├── core/              # Core functionality
│   │   ├── classification/ # Classification modules
│   │   ├── feature_extraction/ # Feature extraction modules
│   │   ├── preprocessing/ # Preprocessing modules
│   │   └── logging/       # Logging modules
│   ├── data/              # Data-related code
│   │   ├── collection/    # Data collection scripts
│   │   ├── preprocessing/ # Data preprocessing scripts
│   │   └── augmentation/  # Data augmentation scripts
│   ├── frontend/          # Frontend code
│   ├── models/            # Model-related code
│   │   ├── training/      # Model training scripts
│   │   ├── evaluation/    # Model evaluation scripts
│   │   └── deployment/    # Model deployment scripts
│   ├── config/            # Configuration files
│   ├── scripts/           # Utility scripts
│   └── utils/             # Utility functions
├── cache/                 # Cache directory
├── auto_spider_img/       # Auto spider for images
├── README.md              # English documentation
└── README.zh.md           # Chinese documentation
```

## 🎮 Supported Characters

### Blue Archive

- **Hoshino**
- **Shiroko**
- **Arona**
- **Miyako**
- **Hina**
- **Yuuka**

## 🌐 Usage

### Web Interface Usage

1. Open your browser and visit `http://127.0.0.1:5001`
2. Click the "Select Image File" button to choose the image to be recognized
3. Click the "Identify Character" button, and the system will automatically analyze the image
4. Wait for the analysis to complete, and view the recognition result and confidence

### Role Detection Workflow

1. Open your browser and visit `http://127.0.0.1:5001/workflow`
2. Enter character information in JSON format (e.g., `[{"name": "Aimi Kanazawa", "series": "bangdream_mygo"}]`)
3. Enter test image path
4. Adjust training parameters (batch size, epochs, learning rate, etc.)
5. Click "Start Workflow" to begin end-to-end process
6. Monitor progress in the terminal

### API Call

```bash
# Use curl to upload image and identify (default method)
curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:5001/api/classify

# Use curl with dedicated model
curl -X POST -F "file=@path/to/image.jpg" -F "use_model=true" http://127.0.0.1:5001/api/classify

# Use curl with DeepDanbooru integration
curl -X POST -F "file=@path/to/image.jpg" -F "use_deepdanbooru=true" http://127.0.0.1:5001/api/classify
```

API response example:

```json
{
  "filename": "image.jpg",
  "role": "Blue Archive_Hoshino",
  "similarity": 0.98,
  "boxes": []
}
```

## 📊 System Performance

### Recognition Accuracy

| Character | Accuracy |
|-----------|----------|
| Yuuka     | 100%     |
| Arona     | 83.33%   |
| Miyako    | 60%      |
| Hoshino   | 40%      |
| Shiroko   | 37.50%   |
| Hina      | 30%      |

### Average Processing Time

- Image Upload: ~1 second
- Preprocessing: ~0.5 seconds
- Feature Extraction: ~0.3 seconds
- Classification: ~0.1 seconds
- Total Time: ~2 seconds

### Model Training Performance

- **Training Speed**: ~9.0 batch/s on MPS (Apple Silicon)
- **Per Epoch Time**: ~35-40 minutes
- **Total Training Time**: ~30 hours for 50 epochs
- **Initial Loss**: 4.79
- **Current Loss**: ~3.25 (after 2nd epoch)
- **Best Validation Accuracy**: 0.0562 (after 2nd epoch, training in progress)

## 🔧 Technical Implementation

### Core Technology Stack

| Technology | Purpose |
|------------|---------|
| Python     | Main development language |
| Flask      | Web application framework |
| PyTorch    | Deep learning framework |
| CLIP       | Image feature extraction |
| YOLOv8     | Character detection |
| Faiss      | Similarity search |
| EfficientNet-B0 | Character classification |
| DeepDanbooru | Anime tag recognition |
| HTML/CSS   | Frontend interface |

### End-to-End Workflow

1. **Data Collection**: Collect character images via Bing Image Search API
2. **Dataset Splitting**: Split collected data into training (80%) and validation (20%) sets
3. **Model Training**: Train EfficientNet-B0 model with data augmentation
4. **Model Evaluation**: Evaluate model performance on validation set
5. **Character Detection**: Use trained model to detect characters in new images

### Model Training Pipeline

- **Data Augmentation**: Random resized crop, horizontal/vertical flip, rotation, color jitter
- **Optimizer**: AdamW with weight decay
- **Learning Rate Scheduler**: Cosine annealing
- **Batch Size**: 16
- **Training Epochs**: 50
- **Initial Learning Rate**: 5e-5

### Distributed Training

The system supports distributed training across multiple GPUs for faster training:

```bash
# Start distributed training
python scripts/model_training/train_model_distributed.py --batch_size 8 --num_epochs 50 --learning_rate 5e-5 --weight_decay 1e-4 --num_workers 4
```

**Key Features:**
- Automatic detection of available GPUs
- DistributedDataParallel (DDP) implementation
- Synchronized training across multiple GPUs
- Automatic batch size scaling (batch size per GPU)
- Fallback to single GPU mode if only one GPU is available

**Expected Speedup:**
- 2 GPUs: ~2x faster training
- 4 GPUs: ~4x faster training
- 8 GPUs: ~8x faster training

**Note:** Distributed training requires at least 2 GPUs to be effective.

### DeepDanbooru Integration

#### Implementation Principle

The system integrates DeepDanbooru for anime tag recognition to improve classification accuracy. The implementation follows the **Tag-assisted Inference** approach:

1. **Tag Extraction**: Uses DeepDanbooru to extract tags from input images
2. **Tag Mapping**: Maps extracted tags to character attributes
3. **Score Adjustment**: Adjusts classification scores based on tag matching
4. **Result Reordering**: Reorders classification results based on adjusted scores

#### Key Advantages

- **Solves "Sameface Syndrome"**: DeepDanbooru can identify distinguishing features like hair color, eye color, and clothing
- **Improves Robustness**: Even with different art styles, the system can recognize characters based on key features
- **Faster Convergence**: The model learns faster when guided by tag information
- **Higher Accuracy**: Tag-assisted inference significantly improves recognition accuracy

#### Usage Method

The system provides three inference modes:

1. **Default Mode**: Uses CLIP + Faiss for classification
2. **Dedicated Model Mode**: Uses EfficientNet-B0 for classification
3. **DeepDanbooru Mode**: Uses integrated CLIP + EfficientNet-B0 + DeepDanbooru for classification

To use DeepDanbooru integration, simply add the `use_deepdanbooru=true` parameter to your API request.

## 📈 System Optimization

### Performance Optimization

- **Singleton Pattern**: Avoids repeated model initialization, reduces memory usage
- **Caching Mechanism**: Caches processed results, improves response speed
- **Batch Processing**: Supports batch image classification, improves processing efficiency
- **Asynchronous Loading**: Lazy loading of models, speeds up system startup
- **MPS Acceleration**: Uses Apple Silicon GPU for faster training and inference

### User Experience Optimization

- **Loading Animation**: Adds loading animations during upload and processing
- **Responsive Design**: Adapts to different screen sizes
- **Real-time Feedback**: Provides detailed recognition results and confidence
- **Error Handling**: Friendly error prompts
- **Workflow Interface**: Dedicated interface for end-to-end character detection

## 🔄 Log Fusion Feature

### Feature Introduction

The log fusion feature is an important characteristic of the system, which can:

- **Collect Classification Logs**: Automatically collects results and features from each classification
- **Fuse Features**: Fuses features from multiple classification results into a new model
- **Continuous Learning**: Learns from historical classification data, continuously improving classification accuracy
- **Model Update**: Regularly updates models to maintain system performance

### Usage Method

#### 1. Collect Classification Logs

The system automatically collects results from each classification, including:
- Uploaded images
- Extracted feature vectors
- Classification results
- Confidence scores

#### 2. Fuse Features to Build New Model

```bash
# Run log fusion script
python3 src/core/log_fusion/log_fusion.py --log_dir ./logs --output_model ./models/fused_model
```

#### 3. Use New Model for Classification

The system automatically uses the latest built model for classification without additional configuration.

## 📊 Global Logging System

### System Overview

The global logging system is a unified log management module that records system status, model inference results, model training results, and error logs. It is based on the loguru library and provides a directory structure organized by log type and log rotation functionality.

### Log Directory Structure

The global logging system stores log files in directories organized by type:

```
logs/
├── system/         # System status logs
├── inference/      # Model inference result logs
├── training/       # Model training result logs
└── error/          # Error logs
```

### Log Rotation Configuration

The global logging system is configured with the following log rotation policies:

| Log Type | Rotation Policy | Retention Period | Compression |
|---------|----------------|------------------|-------------|
| System Logs | 100 MB | 7 days | zip |
| Inference Logs | 100 MB | 14 days | zip |
| Training Logs | 200 MB | 30 days | zip |
| Error Logs | 50 MB | 30 days | zip |

### Usage Method

In modules that need to use logging, import the global logging system and use it:

```python
from src.core.logging.global_logger import (
    get_logger, log_system, log_inference, log_training, log_error
)

# Use convenience functions to record logs
log_system("System started successfully")
log_inference("Model inference completed, recognition result: Character A, similarity: 0.95")
log_training("Model training completed, accuracy: 98.5%")
log_error("File upload failed: File too large")

# Use custom logger object
logger = get_logger("module_name")
logger.info("Module initialized successfully")
logger.error("Module error: Invalid parameter")
```

### Log Levels

The global logging system supports the following log levels:

- DEBUG: Detailed debug information
- INFO: General information
- WARNING: Warning information
- ERROR: Error information
- CRITICAL: Critical error information

## 🤝 Contribution Guide

Welcome to submit Issues and Pull Requests to jointly improve system performance and functionality.

## 📄 License

This project is open source under the MIT license.

## 📞 Contact Us

If you have any questions or suggestions, please contact us through:

- Email: zhaoqi.cao@icloud.com
- GitHub: https://github.com/caozhaoqi/anime-role-detect

---

**© 2026 Character Classification System** - Making character recognition simple!

