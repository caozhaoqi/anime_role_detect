# 二次元角色识别与分类系统

## 项目简介

本项目是一个基于深度学习的二次元角色识别与分类系统，能够自动识别图片中的二次元角色并进行分类，同时为图片添加相关标签。

## 技术栈

- **语言**: Python 3.10+
- **核心库**: PyTorch, OpenCV, Transformers (Hugging Face)
- **向量检索**: Faiss
- **模型**: CLIP (用于特征提取), YOLOv8 (用于角色检测), DeepDanbooru (用于标签生成)
- **Web界面**: Gradio

## 系统架构

系统采用模块化设计，分为以下几个核心模块：

1. **预处理模块** (`src/core/preprocessing/preprocessing.py`): 使用YOLOv8模型检测图片中的角色主体，进行图像裁剪和标准化，支持单角色和多角色检测。
2. **特征提取模块** (`src/core/feature_extraction/feature_extraction.py`): 使用CLIP模型将图像映射为高维特征向量，支持批量特征提取。
3. **分类模块** (`src/core/classification/classification.py`): 使用Faiss进行向量索引和相似度计算，实现角色分类，支持增量学习。
4. **数据准备模块** (`src/scripts/data_preparation/data_preparation.py`): 管理数据集和构建向量索引。
5. **标签模块** (`src/core/tagging/tagging.py`): 集成DeepDanbooru为图片添加标签。
6. **异常处理模块** (`src/core/exception_handling/exception_handling.py`): 处理系统运行过程中的异常情况。
7. **Web UI模块** (`src/web/web_ui.py`): 提供用户友好的Web界面，支持单角色和多角色识别。
8. **自动化分类脚本** (`src/scripts/classification_script/classification_script.py`): 批量处理图片并自动归档，支持多角色识别。
9. **主脚本** (`main.py`): 整合所有模块，提供命令行接口。

## 快速开始

### 1. 环境搭建

```bash
# 安装依赖
python3 -m pip install -r requirements.txt
```

### 2. 准备数据集

创建数据集目录结构：

```
dataset/
  角色1/
    1.jpg
    2.jpg
    ...
  角色2/
    1.jpg
    2.jpg
    ...
  ...
```

每个角色目录中应包含20-50张不同角度的照片。

### 3. 构建向量索引

```bash
python3 main.py build_index --data_dir dataset --index_path role_index
```

### 4. 分类图片

```bash
python3 main.py classify --input_dir input --output_dir output --index_path role_index --threshold 0.7
```

### 5. 运行Web UI

```bash
python3 main.py web_ui --index_path role_index --threshold 0.7 --server_port 7860
```

然后在浏览器中访问 `http://localhost:7860` 即可使用Web界面。

### 6. 修正分类结果（增量学习）

当系统分类错误时，可以手动修正并更新索引：

```bash
python3 main.py correct --image_path path/to/image.jpg --correct_role 正确角色名 --index_path role_index
```

## 使用方法

### 命令行接口

```bash
# 构建索引
python3 main.py build_index [--data_dir DATA_DIR] [--index_path INDEX_PATH]

# 分类图片
python3 main.py classify [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--index_path INDEX_PATH] [--threshold THRESHOLD]

# 运行Web UI
python3 main.py web_ui [--index_path INDEX_PATH] [--threshold THRESHOLD] [--share] [--server_port SERVER_PORT]

# 修正分类结果（增量学习）
python3 main.py correct [--image_path IMAGE_PATH] [--correct_role CORRECT_ROLE] [--index_path INDEX_PATH]
```

### Web界面使用

1. 打开Web浏览器，访问 `http://localhost:7860`
2. 点击「上传图片」按钮，选择一张二次元角色图片
3. 根据图片情况选择识别模式：
   - **单角色识别**: 适用于只有一个角色的图片
   - **多角色识别**: 适用于有多个角色的图片
4. 系统会自动处理图片并显示识别结果和标签
5. 可以查看处理后的图片效果
6. **增量学习功能**:
   - 如果识别结果错误，可以在「修正分类」部分输入正确的角色名称
   - 点击「修正并更新索引」按钮，系统会更新向量索引，提高未来的识别准确率

## 系统功能

1. **角色识别**: 自动识别图片中的二次元角色
2. **图片分类**: 根据识别结果将图片分类到对应角色目录
3. **标签生成**: 为图片自动添加相关标签（如：银发、校服、红领结等）
4. **异常处理**: 将无法识别的图片放入Unknown文件夹
5. **批量处理**: 支持批量处理多张图片
6. **Web界面**: 提供用户友好的Web操作界面
7. **性能优化**: 支持GPU加速，提高处理速度
8. **增量学习**: 支持手动修正错误分类并更新索引，提高系统识别准确率
9. **多角色识别**: 支持一张图片中多个角色的同时识别
10. **缓存机制**: 实现模型和结果缓存，提高系统响应速度
11. **模块化设计**: 采用模块化架构，便于维护和扩展

## 性能指标

- **预处理时间**: ~0.1-0.3秒/张
- **特征提取时间**: ~0.5-1.0秒/张
- **分类时间**: ~0.01-0.05秒/张
- **端到端时间**: ~0.6-1.4秒/张

## 注意事项

1. **数据集质量**: 数据集的质量直接影响识别效果，建议为每个角色收集不同角度、不同场景的照片。
2. **模型选择**: 系统默认使用CLIP模型进行特征提取，也可以根据需要更换为其他模型。
3. **阈值调整**: 相似度阈值默认为0.7，可以根据实际情况调整，提高阈值会提高识别准确率但降低召回率。
4. **硬件要求**: 建议使用GPU加速，以提高处理速度。
5. **模型下载**: 首次运行时，系统会自动下载所需的模型文件，可能需要一些时间。

## 未来规划

1. **模型优化**: 进一步优化模型，提高识别准确率和处理速度。
2. **增量学习**: 实现增量学习功能，当用户手动修正错误分类后，系统自动更新索引。
3. **多角色识别**: 支持一张图片中多个角色的识别。
4. **模型量化**: 使用模型量化技术，减小模型体积，提高推理速度。
5. **部署优化**: 优化系统部署，支持Docker容器化部署。

## 目录结构

```
anime_role_detect/
  ├── preprocessing.py          # 预处理模块
  ├── feature_extraction.py     # 特征提取模块
  ├── classification.py         # 分类模块
  ├── data_preparation.py       # 数据准备模块
  ├── tagging.py                # 标签模块
  ├── exception_handling.py     # 异常处理模块
  ├── web_ui.py                 # Web UI模块
  ├── classification_script.py  # 自动化分类脚本
  ├── main.py                   # 主脚本
  ├── test_performance.py       # 性能测试脚本
  ├── requirements.txt          # 依赖包
  ├── README.md                 # 项目说明
  └── README_DETAILED.md        # 详细文档
```

## 依赖包

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
transformers>=4.30.0
huggingface-hub>=0.16.0
faiss-cpu>=1.7.4
ultralytics>=8.0.0
clip-anytorch>=2.5.2
deepdanbooru>=1.0.0
gradio>=3.40.0
numpy>=1.24.0
Pillow>=9.5.0
scipy>=1.10.0
```

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请联系项目维护者。
