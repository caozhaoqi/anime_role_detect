#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型部署优化脚本

实现模型量化、ONNX转换和边缘设备部署支持。
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.quantization
import onnx
import onnxruntime
import numpy as np
import json
import logging
from torchvision import transforms, models
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('deployment_optimizer')

class ModelOptimizer:
    """模型优化器类"""
    
    def __init__(self, model_path, model_type, num_classes, device='cpu'):
        """初始化模型优化器
        
        Args:
            model_path: 模型路径
            model_type: 模型类型
            num_classes: 类别数量
            device: 设备
        """
        self.model_path = model_path
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device
        
        # 加载模型
        self.model = self._get_model(model_type, num_classes)
        self._load_model_weights(model_path)
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"模型优化器初始化完成，模型: {model_type}, 类别数: {num_classes}")
    
    def _get_model(self, model_type, num_classes):
        """获取模型
        
        Args:
            model_type: 模型类型
            num_classes: 类别数量
        
        Returns:
            模型
        """
        if model_type == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_type == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model
    
    def _load_model_weights(self, model_path):
        """加载模型权重
        
        Args:
            model_path: 模型路径
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        logger.info(f"模型权重加载完成: {model_path}")
    
    def quantize_model(self, output_path):
        """量化模型（动态量化）
        
        Args:
            output_path: 输出路径
        
        Returns:
            量化后的模型
        """
        logger.info("开始模型量化...")
        
        # 暂时将模型移到CPU上进行量化
        model_cpu = self.model.to('cpu')
        
        # 准备量化模型
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # 保存量化模型
        torch.save(quantized_model.state_dict(), output_path)
        logger.info(f"量化模型已保存到: {output_path}")
        
        # 评估量化前后的模型大小
        original_size = os.path.getsize(self.model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"原始模型大小: {original_size:.2f} MB")
        logger.info(f"量化模型大小: {quantized_size:.2f} MB")
        logger.info(f"压缩率: {((original_size - quantized_size) / original_size) * 100:.2f}%")
        
        return quantized_model
    
    def convert_to_onnx(self, output_path, input_shape=(1, 3, 224, 224)):
        """将模型转换为ONNX格式
        
        Args:
            output_path: 输出路径
            input_shape: 输入形状
        """
        logger.info("开始模型转换为ONNX格式...")
        
        # 创建示例输入
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # 导出ONNX模型
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        logger.info(f"ONNX模型已保存到: {output_path}")
        
        # 验证ONNX模型
        self._validate_onnx_model(output_path)
    
    def _validate_onnx_model(self, onnx_path):
        """验证ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
        """
        logger.info("验证ONNX模型...")
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        
        # 检查模型
        try:
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX模型验证通过")
        except Exception as e:
            logger.error(f"ONNX模型验证失败: {e}")
    
    def evaluate_onnx_performance(self, onnx_path, test_image=None):
        """评估ONNX模型性能
        
        Args:
            onnx_path: ONNX模型路径
            test_image: 测试图像路径
        
        Returns:
            性能指标
        """
        logger.info("评估ONNX模型性能...")
        
        # 创建ONNX Runtime会话
        session = onnxruntime.InferenceSession(onnx_path)
        
        # 获取输入输出名称
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 准备输入数据
        if test_image:
            # 使用真实图像测试
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = Image.open(test_image).convert('RGB')
            image = transform(image).unsqueeze(0).numpy()
        else:
            # 使用随机数据测试
            image = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # 测试推理时间
        import time
        start_time = time.time()
        outputs = session.run([output_name], {input_name: image})
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # ms
        fps = 1000 / inference_time
        
        logger.info(f"ONNX模型推理时间: {inference_time:.2f} ms")
        logger.info(f"ONNX模型推理速度: {fps:.2f} FPS")
        
        return {
            'inference_time_ms': inference_time,
            'fps': fps
        }
    
    def create_tflite_model(self, output_path):
        """创建TFLite模型（需要通过ONNX转换）
        
        Args:
            output_path: 输出路径
        """
        logger.info("开始创建TFLite模型...")
        
        # 首先转换为ONNX
        onnx_path = output_path.replace('.tflite', '.onnx')
        self.convert_to_onnx(onnx_path)
        
        # 使用ONNX转换工具转换为TFLite（这里需要安装相应工具）
        # 注意：实际使用时需要安装 onnx-tf 和 tensorflow
        try:
            from onnx_tf.backend import prepare
            import tensorflow as tf
            
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 转换为TensorFlow模型
            tf_rep = prepare(onnx_model)
            
            # 保存为TensorFlow SavedModel
            tf_saved_model_path = output_path.replace('.tflite', '_saved_model')
            tf_rep.export_graph(tf_saved_model_path)
            
            # 转换为TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
            
            # 启用量化
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # 转换
            tflite_model = converter.convert()
            
            # 保存TFLite模型
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite模型已保存到: {output_path}")
            
        except ImportError as e:
            logger.warning(f"无法创建TFLite模型，缺少依赖: {e}")
            logger.info("请安装 onnx-tf 和 tensorflow 以支持TFLite转换")
    
    def optimize_for_edge(self, output_dir):
        """为边缘设备优化模型
        
        Args:
            output_dir: 输出目录
        
        Returns:
            优化结果
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # 1. 复制原始模型
        original_path = os.path.join(output_dir, f'{self.model_type}_original.pth')
        import shutil
        shutil.copyfile(self.model_path, original_path)
        results['original_model'] = original_path
        
        # 2. 转换为ONNX
        onnx_path = os.path.join(output_dir, f'{self.model_type}.onnx')
        self.convert_to_onnx(onnx_path)
        results['onnx_model'] = onnx_path
        
        # 3. 评估ONNX性能
        onnx_perf = self.evaluate_onnx_performance(onnx_path)
        results['onnx_performance'] = onnx_perf
        
        # 4. 尝试创建TFLite模型
        tflite_path = os.path.join(output_dir, f'{self.model_type}.tflite')
        self.create_tflite_model(tflite_path)
        results['tflite_model'] = tflite_path
        
        # 5. 保存优化结果
        results_path = os.path.join(output_dir, 'optimization_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"优化结果已保存到: {results_path}")
        
        return results

def create_deployment_files(model_path, model_type, output_dir, class_to_idx=None):
    """创建部署所需的文件
    
    Args:
        model_path: 模型路径
        model_type: 模型类型
        output_dir: 输出目录
        class_to_idx: 类别映射
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 创建配置文件
    config = {
        'model_type': model_type,
        'model_path': os.path.basename(model_path),
        'input_shape': [1, 3, 224, 224],
        'preprocessing': {
            'resize': 256,
            'crop': 224,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'class_to_idx': class_to_idx
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"配置文件已保存到: {config_path}")
    
    # 2. 创建推理脚本
    inference_script = '''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边缘设备推理脚本
"""

import os
import json
import numpy as np
import time
from PIL import Image

class AnimeRoleDetector:
    """动漫角色检测器"""
    
    def __init__(self, model_path, config_path):
        """初始化检测器
        
        Args:
            model_path: 模型路径
            config_path: 配置路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.model_path = model_path
        self.class_to_idx = self.config.get('class_to_idx', {})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 加载模型
        self.model = self._load_model()
        
        print(f"初始化完成，支持 {len(self.class_to_idx)} 个角色")
    
    def _load_model(self):
        """加载模型"""
        # 根据模型类型选择加载方式
        model_type = self.config.get('model_type', 'mobilenet_v2')
        
        if self.model_path.endswith('.onnx'):
            # 使用ONNX Runtime
            import onnxruntime
            return onnxruntime.InferenceSession(self.model_path)
        elif self.model_path.endswith('.tflite'):
            # 使用TFLite
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            # 使用PyTorch
            import torch
            import torch.nn as nn
            from torchvision import models
            
            # 获取模型
            if model_type == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_to_idx))
            elif model_type == 'mobilenet_v2':
                model = models.mobilenet_v2(pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_to_idx))
            elif model_type == 'resnet18':
                model = models.resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, len(self.class_to_idx))
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 加载权重
            device = torch.device('cpu')
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            model.eval()
            return model
    
    def preprocess(self, image):
        """预处理图像
        
        Args:
            image: PIL图像
        
        Returns:
            预处理后的图像
        """
        # 调整大小
        image = image.resize((self.config['preprocessing']['resize'], self.config['preprocessing']['resize']))
        
        # 中心裁剪
        width, height = image.size
        crop_size = self.config['preprocessing']['crop']
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        image = image.crop((left, top, right, bottom))
        
        # 转换为数组
        image = np.array(image).astype(np.float32)
        
        # 归一化
        mean = self.config['preprocessing']['mean']
        std = self.config['preprocessing']['std']
        image = (image / 255.0 - mean) / std
        
        # 调整维度
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = np.expand_dims(image, axis=0)  # 添加批次维度
        
        return image
    
    def predict(self, image_path):
        """预测图像
        
        Args:
            image_path: 图像路径
        
        Returns:
            预测结果
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 预处理
        preprocessed = self.preprocess(image)
        
        # 推理
        start_time = time.time()
        
        try:
            # 尝试ONNX Runtime
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            outputs = self.model.run([output_name], {input_name: preprocessed.astype(np.float32)})
            predictions = outputs[0]
        except AttributeError:
            try:
                # 尝试TFLite
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                self.model.set_tensor(input_details[0]['index'], preprocessed.astype(np.float32))
                self.model.invoke()
                predictions = self.model.get_tensor(output_details[0]['index'])
            except AttributeError:
                # 尝试PyTorch
                import torch
                input_tensor = torch.from_numpy(preprocessed).float()
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                predictions = outputs.numpy()
        
        inference_time = (time.time() - start_time) * 1000
        
        # 后处理
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        class_name = self.idx_to_class.get(predicted_class, '未知')
        
        return {
            'class_name': class_name,
            'confidence': float(confidence),
            'inference_time_ms': inference_time,
            'fps': 1000 / inference_time
        }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='边缘设备推理脚本')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--model', type=str, default='models/deployment/mobilenet_v2.onnx', help='模型路径')
    parser.add_argument('--config', type=str, default='models/deployment/config.json', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = AnimeRoleDetector(args.model, args.config)
    
    # 预测
    result = detector.predict(args.image)
    
    # 打印结果
    print(f"预测角色: {result['class_name']}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    print(f"推理速度: {result['fps']:.2f} FPS")
'''
    
    inference_script_path = os.path.join(output_dir, 'inference.py')
    with open(inference_script_path, 'w', encoding='utf-8') as f:
        f.write(inference_script)
    
    # 添加执行权限
    os.chmod(inference_script_path, 0o755)
    logger.info(f"推理脚本已保存到: {inference_script_path}")
    
    # 3. 创建README文件
    readme = """
# 动漫角色检测器部署指南

## 模型信息

- 模型类型: {model_type}
- 支持角色数: {num_classes}
- 模型文件: {model_basename}

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
   python inference.py --image path/to/image.jpg --model models/{model_type}.onnx
   
   # 使用TFLite模型
   python inference.py --image path/to/image.jpg --model models/{model_type}.tflite
   
   # 使用PyTorch模型
   python inference.py --image path/to/image.jpg --model models/{model_type}_quantized.pth
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
""".format(
        model_type=model_type,
        num_classes=len(class_to_idx) if class_to_idx else 0,
        model_basename=os.path.basename(model_path)
    )
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    logger.info(f"README文件已保存到: {readme_path}")

def main():
    parser = argparse.ArgumentParser(description='模型部署优化脚本')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2',
                       choices=['efficientnet_b0', 'mobilenet_v2', 'resnet18'],
                       help='模型类型')
    parser.add_argument('--num-classes', type=int, default=32,
                       help='类别数量')
    parser.add_argument('--output-dir', type=str, default='models/deployment',
                       help='输出目录')
    parser.add_argument('--test-image', type=str, default=None,
                       help='测试图像路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化模型优化器
    optimizer = ModelOptimizer(
        model_path=args.model_path,
        model_type=args.model_type,
        num_classes=args.num_classes,
        device=device
    )
    
    # 为边缘设备优化模型
    results = optimizer.optimize_for_edge(args.output_dir)
    
    # 加载类别映射（如果存在）
    class_to_idx = None
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
    
    # 创建部署文件
    create_deployment_files(
        model_path=args.model_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
        class_to_idx=class_to_idx
    )
    
    logger.info("部署优化完成！")
    logger.info(f"优化结果保存到: {args.output_dir}")

if __name__ == '__main__':
    main()
