#!/usr/bin/env python3
import os
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('analyze_image')

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

class ImageAnalyzer:
    """图像分析器"""
    
    def __init__(self, model_path, num_classes=26):
        """初始化分析器
        
        Args:
            model_path: 模型路径
            num_classes: 类别数量
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
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
        model = CharacterClassifier(self.num_classes)
        
        # 加载模型状态
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 检查是否包含完整的训练状态
        if 'model_state_dict' in checkpoint:
            logger.info("从完整训练状态中提取模型权重")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.info("直接加载模型权重")
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
        return [
            '原神_丽莎', '原神_凯亚', '原神_安柏', '原神_温迪', '原神_琴', '原神_空', 
            '原神_芭芭拉', '原神_荧', '原神_迪卢克', '原神_雷泽', '幻塔_凛夜', 
            '我推的孩子_星野爱', '明日方舟_德克萨斯', '明日方舟_能天使', '明日方舟_阿米娅', 
            '明日方舟_陈', '绝区零_安比', '绝区零_杰克', '蔚蓝档案_优花梨', '蔚蓝档案_宫子', 
            '蔚蓝档案_日奈', '蔚蓝档案_星野', '蔚蓝档案_白子', '蔚蓝档案_阿罗娜', 
            '鸣潮_守岸人', '鸣潮_椿'
        ]
    
    def analyze(self, image_path):
        """分析图像
        
        Args:
            image_path: 图像路径
        
        Returns:
            分析结果
        """
        logger.info(f"分析图像: {image_path}")
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        logger.info(f"图像大小: {image.size}")
        
        # 预处理图像
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        logger.info(f"预处理后张量形状: {input_tensor.shape}")
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        
        # 获取预测结果
        predicted_name = self.class_names[predicted_class]
        
        # 获取前5个预测结果
        top5_indices = np.argsort(probabilities)[::-1][:5]
        top5_results = [(self.class_names[idx], probabilities[idx]) for idx in top5_indices]
        
        # 分析结果
        analysis = {
            'image_path': image_path,
            'image_size': image.size,
            'predicted_class': predicted_class,
            'predicted_name': predicted_name,
            'confidence': float(confidence),
            'top5_results': top5_results,
            'all_probabilities': {self.class_names[i]: float(probabilities[i]) for i in range(len(probabilities))}
        }
        
        logger.info(f"预测结果: {predicted_name} (置信度: {confidence:.4f})")
        logger.info("前5个预测结果:")
        for name, prob in top5_results:
            logger.info(f"  {name}: {prob:.4f}")
        
        return analysis

def main():
    """主函数"""
    # 模型路径
    model_path = 'models/character_classifier_best_improved.pth'
    
    # 待分析图像路径
    image_path = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/tests/img/微信图片_20260204115846_481_347.jpg'
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(image_path):
        logger.error(f"图像文件不存在: {image_path}")
        return
    
    # 创建分析器
    analyzer = ImageAnalyzer(model_path)
    
    # 分析图像
    analysis = analyzer.analyze(image_path)
    
    # 打印详细分析结果
    print("\n=== 详细分析结果 ===")
    print(f"图像路径: {analysis['image_path']}")
    print(f"图像大小: {analysis['image_size']}")
    print(f"预测类别: {analysis['predicted_class']}")
    print(f"预测角色: {analysis['predicted_name']}")
    print(f"置信度: {analysis['confidence']:.4f}")
    print("\n前5个预测结果:")
    for i, (name, prob) in enumerate(analysis['top5_results'], 1):
        print(f"{i}. {name}: {prob:.4f}")
    
    print("\n=== 分析总结 ===")
    print(f"分类模型认为这张图片中的角色是: {analysis['predicted_name']}")
    print(f"模型对这个预测的置信度是: {analysis['confidence']:.4f}")
    print("\n模型识别过程:")
    print("1. 加载预训练的EfficientNet-B0分类模型")
    print("2. 对输入图像进行预处理: 调整大小(256x256) -> 中心裁剪(224x224) -> 转换为张量 -> 归一化")
    print("3. 将预处理后的图像输入模型，获取各分类的概率分布")
    print("4. 选择概率最高的类别作为预测结果")
    
    # 分析预测结果的可靠性
    if analysis['confidence'] > 0.5:
        print("\n预测可靠性: 高")
        print("模型对这个预测结果非常有信心")
    elif analysis['confidence'] > 0.3:
        print("\n预测可靠性: 中")
        print("模型对这个预测结果有一定信心，但可能存在不确定性")
    else:
        print("\n预测可靠性: 低")
        print("模型对这个预测结果信心不足，可能需要进一步验证")

if __name__ == "__main__":
    main()
