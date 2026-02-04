# Character Classification System User Guide

## 🎯 System Introduction

The Character Classification System is an AI-based image recognition tool specifically designed to identify characters from games and anime. The system uses advanced deep learning techniques to quickly and accurately identify characters in uploaded images, now with support for end-to-end character detection workflows.

## ✨ Core Features

- **Image Upload Recognition**: Supports multiple image formats for upload, automatically identifies game characters in images
- **High Accuracy**: Uses CLIP model and Faiss indexing for high recognition accuracy
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

### Install Dependencies

```bash
# Install Flask
pip3 install flask

# Install other dependencies (if not already installed)
pip3 install torch torchvision transformers ultralytics faiss-cpu Pillow efficientnet_pytorch
```

### Start System

#### 1. Start Backend Service

```bash
# Start Flask backend application
python3 src/web/web_app.py
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
├── data/                   # Data directory
│   ├── all_characters/            # All character images (including BangDream MyGo!)
│   ├── blue_archive_optimized/    # Optimized Blue Archive data
│   ├── blue_archive_optimized_v2/ # Enhanced Blue Archive data
│   ├── augmented_characters/      # Augmented character data
│   └── split_dataset/             # Split training/validation data
├── src/                    # Source code
│   ├── core/               # Core modules
│   │   ├── classification/         # Classification module
│   │   ├── feature_extraction/     # Feature extraction module
│   │   ├── preprocessing/          # Preprocessing module
│   │   ├── general_classification.py  # General classification module
│   │   └── log_fusion/              # Log fusion module
│   └── web/                # Web application
│       ├── templates/      # HTML templates
│       ├── static/         # Static files
│       └── web_app.py      # Flask application
├── scripts/                # Helper scripts
│   ├── data_collection/    # Data collection scripts
│   ├── data_processing/    # Data processing scripts
│   ├── model_training/     # Model training scripts
│   └── workflow/           # End-to-end workflow scripts
├── tests/                  # Test code
├── README.md               # English documentation
└── README.zh.md            # Chinese documentation
```

## 🎮 Supported Characters

### Blue Archive

- **Hoshino**
- **Shiroko**
- **Arona**
- **Miyako**
- **Hina**
- **Yuuka**

### BangDream MyGo!

- **Aimi Kanazawa** (千早爱音)
- **Ritsuki椎名立希** (椎名立希)
- **Touko Takamatsu** (高松灯)
- **Soyo Nagasaki** (长崎素世)
- **Sakiko Tamagawa** (玉井祥子)
- **Mutsumi Wakaba** (若叶睦)
- **Rana Himemiya** (姬川瑠夏)

### Other Characters

- **Genshin Impact** characters (原神)
- **Spy × Family** characters (间谍过家家)
- **Attack on Titan** characters (进击的巨人)
- **One Piece** characters (海贼王)
- **Naruto** characters (火影忍者)
- **My Hero Academia** characters (我的英雄学院)
- **Tokyo Revengers** characters (东京复仇者)

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
# Use curl to upload image and identify
curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:5001/api/classify
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

- **Training Speed**: ~2.05 batch/s on MPS
- **Per Epoch Time**: ~1 hour 8 minutes
- **Total Training Time**: ~54 hours for 50 epochs
- **Initial Loss**: 4.79
- **Current Loss**: 1.06 (after 1st epoch)

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

## 📋 Latest Updates

### February 2026

- **Added BangDream MyGo! characters** with 50-102 images per character
- **Implemented end-to-end character detection workflow** from data collection to model training
- **Added EfficientNet-B0 model** for character classification
- **Improved API documentation** with GET request support
- **Enhanced data processing pipeline** with automatic dataset splitting
- **Added MPS acceleration** for faster training on Apple Silicon

### Training Progress

- **Current Status**: Training in progress (1st epoch completed)
- **Model**: EfficientNet-B0
- **Training Data**: 131 classes, 133,049 images
- **Validation Data**: 49,741 images
- **Current Loss**: 1.06
- **Training Speed**: ~2.05 batch/s
- **Expected Completion**: February 6, 2026

