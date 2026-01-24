# Character Classification System User Guide

## 🎯 System Introduction

The Character Classification System is an AI-based image recognition tool specifically designed to identify characters from games and anime. The system uses advanced deep learning techniques to quickly and accurately identify characters in uploaded images.

## ✨ Core Features

- **Image Upload Recognition**: Supports multiple image formats for upload, automatically identifies game characters in images
- **High Accuracy**: Uses CLIP model and Faiss indexing for high recognition accuracy
- **Real-time Feedback**: Provides recognition confidence and detailed results
- **User-friendly Interface**: Intuitive web interface with simple operation
- **API Support**: Provides RESTful API interface for batch processing
- **Log Fusion**: Supports fusing features from classification logs to build new models

## 🚀 Quick Start

### Environment Requirements

- Python 3.7+
- Flask
- PyTorch
- Transformers
- Ultralytics (YOLOv8)
- Faiss

### Install Dependencies

```bash
# Install Flask
pip3 install flask

# Install other dependencies (if not already installed)
pip3 install torch torchvision transformers ultralytics faiss-cpu Pillow
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
│   ├── blue_archive_optimized/      # Optimized Blue Archive data
│   └── blue_archive_optimized_v2/   # Enhanced Blue Archive data
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

## 🌐 Usage

### Web Interface Usage

1. Open your browser and visit `http://127.0.0.1:5001`
2. Click the "Select Image File" button to choose the image to be recognized
3. Click the "Identify Character" button, and the system will automatically analyze the image
4. Wait for the analysis to complete, and view the recognition result and confidence

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
| HTML/CSS   | Frontend interface |

### Workflow

1. **Image Upload**: User uploads image to web application
2. **Preprocessing**: Uses YOLOv8 to detect characters in the image, crops and normalizes the image
3. **Feature Extraction**: Uses CLIP model to extract image feature vectors
4. **Similarity Search**: Uses Faiss indexing to search for most similar character features
5. **Result Display**: Shows recognition result and confidence
6. **Log Fusion**: Collects classification logs, fuses features to build new models

## 📈 System Optimization

### Performance Optimization

- **Singleton Pattern**: Avoids repeated model initialization, reduces memory usage
- **Caching Mechanism**: Caches processed results, improves response speed
- **Batch Processing**: Supports batch image classification, improves processing efficiency
- **Asynchronous Loading**: Lazy loading of models, speeds up system startup

### User Experience Optimization

- **Loading Animation**: Adds loading animations during upload and processing
- **Responsive Design**: Adapts to different screen sizes
- **Real-time Feedback**: Provides detailed recognition results and confidence
- **Error Handling**: Friendly error prompts

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
