#!/usr/bin/env python3
"""
角色检测脚本
检测图片中所有角色的功能
"""
import os
import argparse
import logging
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('detect_all_characters')

class CharacterClassifier(torch.nn.Module):
    """角色分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用EfficientNet-B0作为基础模型
        self.backbone = models.efficientnet_b0(pretrained=False)
        # 替换分类头
        self.backbone.classifier[1] = torch.nn.Linear(
            self.backbone.classifier[1].in_features, 
            num_classes
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

class CharacterDetector:
    """角色检测器"""
    
    def __init__(self, model_path, threshold=0.5):
        """初始化检测器
        
        Args:
            model_path: 模型路径
            threshold: 检测阈值
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.threshold = threshold
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
    
    def load_model(self, model_path):
        """加载模型
        
        Args:
            model_path: 模型路径
        
        Returns:
            加载好的模型
        """
        logger.info(f"加载模型: {model_path}")
        # 加载模型状态
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取类别数量
        if 'class_to_idx' in checkpoint:
            num_classes = len(checkpoint['class_to_idx'])
            # 创建类别名称映射
            self.idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
        else:
            # 默认类别数量
            num_classes = 116
            self.idx_to_class = {}
        
        # 初始化模型
        model = CharacterClassifier(num_classes)
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        logger.info("模型加载完成")
        return model
    
    def get_transform(self):
        """获取图像变换
        
        Returns:
            图像变换
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def get_class_names(self):
        """获取类别名称
        
        Returns:
            类别名称列表
        """
        if hasattr(self, 'idx_to_class') and self.idx_to_class:
            return [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]
        else:
            # 默认类别名称（如果没有从模型中获取）
            return [
                '原神_丽莎', '原神_凯亚', '原神_安柏', '原神_温迪', '原神_琴', '原神_空', 
                '原神_芭芭拉', '原神_荧', '原神_迪卢克', '原神_雷泽', '幻塔_凛夜', 
                '我推的孩子_星野爱', '明日方舟_德克萨斯', '明日方舟_能天使', '明日方舟_阿米娅', 
                '明日方舟_陈', '绝区零_安比', '绝区零_杰克', '蔚蓝档案_优花梨', '蔚蓝档案_宫子', 
                '蔚蓝档案_日奈', '蔚蓝档案_星野', '蔚蓝档案_白子', '蔚蓝档案_阿罗娜', 
                '鸣潮_守岸人', '鸣潮_椿'
            ]
    
    def detect_character(self, image):
        """检测单个角色
        
        Args:
            image: PIL图像
        
        Returns:
            检测结果
        """
        # 预处理图像
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        
        # 获取预测结果
        if hasattr(self, 'idx_to_class') and predicted_class in self.idx_to_class:
            predicted_name = self.idx_to_class[predicted_class]
        else:
            predicted_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else f'未知角色_{predicted_class}'
        
        return {
            'character': predicted_name,
            'confidence': float(confidence),
            'class_id': int(predicted_class)
        }
    
    def detect_all_characters(self, image_path):
        """检测图片中所有角色
        
        Args:
            image_path: 图像路径
        
        Returns:
            检测结果列表
        """
        logger.info(f"检测图片: {image_path}")
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        logger.info(f"图像大小: {image.size}")
        
        # 检测角色
        result = self.detect_character(image)
        
        # 检查置信度
        if result['confidence'] < self.threshold:
            logger.warning(f"检测置信度低: {result['confidence']:.4f}")
        
        logger.info(f"检测结果: {result['character']} (置信度: {result['confidence']:.4f})")
        
        return [result]
    
    def detect_multiple_characters(self, image_path, grid_size=3):
        """使用网格分割检测图片中多个角色
        
        Args:
            image_path: 图像路径
            grid_size: 网格大小
        
        Returns:
            检测结果列表
        """
        logger.info(f"使用网格分割检测图片: {image_path}")
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        logger.info(f"图像大小: {width}x{height}")
        
        # 计算网格大小
        grid_width = width // grid_size
        grid_height = height // grid_size
        
        results = []
        
        # 遍历网格
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算网格坐标
                left = j * grid_width
                top = i * grid_height
                right = min((j + 1) * grid_width, width)
                bottom = min((i + 1) * grid_height, height)
                
                # 裁剪图像
                crop_image = image.crop((left, top, right, bottom))
                
                # 检测角色
                result = self.detect_character(crop_image)
                
                # 添加位置信息
                result['position'] = {
                    'left': left,
                    'top': top,
                    'right': right,
                    'bottom': bottom
                }
                
                # 检查置信度
                if result['confidence'] >= self.threshold:
                    results.append(result)
                    logger.info(f"网格 ({i},{j}) 检测结果: {result['character']} (置信度: {result['confidence']:.4f})")
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='角色检测脚本')
    parser.add_argument('--model_path', type=str, default='models/character_classifier_best_improved.pth', help='模型路径')
    parser.add_argument('--image_path', type=str, required=True, help='图像路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='检测阈值')
    parser.add_argument('--grid_size', type=int, default=3, help='网格大小')
    parser.add_argument('--multiple', action='store_true', help='检测多个角色')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.image_path):
        logger.error(f"图像文件不存在: {args.image_path}")
        return
    
    # 创建检测器
    detector = CharacterDetector(args.model_path, threshold=args.threshold)
    
    # 检测角色
    if args.multiple:
        results = detector.detect_multiple_characters(args.image_path, grid_size=args.grid_size)
    else:
        results = detector.detect_all_characters(args.image_path)
    
    # 打印检测结果
    print("\n=== 检测结果 ===")
    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. 角色: {result['character']}")
            print(f"   置信度: {result['confidence']:.4f}")
            if 'position' in result:
                pos = result['position']
                print(f"   位置: ({pos['left']}, {pos['top']}) - ({pos['right']}, {pos['bottom']})")
            print()
        print(f"共检测到 {len(results)} 个角色")
    else:
        print("未检测到角色")

if __name__ == "__main__":
    main()
