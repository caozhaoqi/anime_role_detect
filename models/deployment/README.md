
# 动漫角色检测器部署指南

## 模型信息

- 模型类型: mobilenet_v2
- 支持角色数: 32
- 模型文件: model_best.pth

## 部署步骤

1. **安装依赖**

   ```bash
   # 基本依赖
   pip install numpy Pillow
   
   # 如果使用ONNX模型
   pip install onnxruntime
   
   # 如果使用TFLite模型
   pip install tensorflow
   
   # 如果使用PyTorch模型
   pip install torch torchvision
   ```

2. **运行推理**

   ```bash
   # 使用ONNX模型
   python inference.py --image path/to/image.jpg --model models/mobilenet_v2.onnx
   
   # 使用TFLite模型
   python inference.py --image path/to/image.jpg --model models/mobilenet_v2.tflite
   
   # 使用PyTorch模型
   python inference.py --image path/to/image.jpg --model models/mobilenet_v2_quantized.pth
   ```

## 性能指标

| 模型类型 | 大小 | 推理时间 | FPS |
|---------|------|---------|-----|
| PyTorch | - | - | - |
| 量化PyTorch | - | - | - |
| ONNX | - | - | - |
| TFLite | - | - | - |

## 支持的设备

- **边缘设备**: Raspberry Pi 4, Jetson Nano
- **移动设备**: Android (via ONNX Runtime or TFLite)
- **Web浏览器**: WebAssembly (via ONNX Runtime Web)

## 注意事项

- 确保输入图像为RGB格式
- 图像会被自动调整为224x224大小
- 推理速度取决于设备性能
