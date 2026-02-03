#!/usr/bin/env python3
"""
测试生成模型效果的脚本
生成图像并通过检测模型验证其分类结果
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_generator')


class View(nn.Module):
    """重塑层"""
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)


class Generator(nn.Module):
    """简化版生成器模型，专为二次元角色设计"""
    
    def __init__(self, latent_dim=128):
        """初始化生成器
        
        Args:
            latent_dim: 潜在空间维度
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        # 输入处理
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 上采样网络
        self.upsample = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出层
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        """前向传播
        
        Args:
            z: 随机噪声 [batch_size, latent_dim]
            
        Returns:
            生成的图像 [batch_size, 3, 224, 224]
        """
        # 处理输入噪声
        x = self.fc(z)
        # 重塑为特征图
        x = x.view(x.size(0), 256, 7, 7)
        # 上采样生成图像
        return self.upsample(x)


class CharacterClassifier(nn.Module):
    """角色分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用EfficientNet-B0作为基础模型
        from torchvision.models import efficientnet_b0
        self.backbone = efficientnet_b0(pretrained=False)
        # 替换分类头
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


class GeneratorTester:
    """生成模型测试器"""
    
    def __init__(self, generator_path, detection_model_path, device=None):
        """初始化测试器
        
        Args:
            generator_path: 生成器模型路径
            detection_model_path: 检测模型路径
            device: 测试设备
        """
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 加载生成器
        checkpoint = torch.load(generator_path, map_location=self.device)
        # 从检查点中读取潜在空间维度
        self.latent_dim = checkpoint.get('latent_dim', 128)
        self.generator = Generator(latent_dim=self.latent_dim).to(self.device)
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.generator.eval()
        logger.info(f"生成器潜在空间维度: {self.latent_dim}")
        
        # 加载检测模型
        self.detection_model = None
        self.class_names = []
        self._load_detection_model(detection_model_path)
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"初始化完成，使用设备: {self.device}")
        logger.info(f"生成器加载成功: {generator_path}")
        logger.info(f"检测模型加载成功: {detection_model_path}")
    
    def _load_detection_model(self, model_path):
        """加载检测模型
        
        Args:
            model_path: 检测模型路径
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 提取分类信息
            if 'class_to_idx' in checkpoint:
                class_to_idx = checkpoint['class_to_idx']
                self.class_names = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])
            elif 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
            else:
                logger.error("检测模型中未找到分类信息")
                return
            
            # 初始化模型
            num_classes = len(self.class_names)
            self.detection_model = CharacterClassifier(num_classes=num_classes).to(self.device)
            self.detection_model.load_state_dict(checkpoint['model_state_dict'])
            self.detection_model.eval()
            
            logger.info(f"检测模型包含 {num_classes} 个角色类别")
            logger.info(f"类别列表: {self.class_names}")
            
        except Exception as e:
            logger.error(f"加载检测模型失败: {e}")
    
    def generate_and_test(self, num_images=10):
        """生成图像并测试
        
        Args:
            num_images: 生成图像数量
        """
        logger.info(f"生成 {num_images} 张图像并测试...")
        
        # 生成随机噪声
        noise = torch.randn(num_images, self.latent_dim, device=self.device)
        
        # 生成图像
        self.generator.eval()
        with torch.no_grad():
            generated_images = self.generator(noise)
        
        # 反标准化
        generated_images_np = generated_images.cpu().numpy()
        generated_images_np = (generated_images_np * 0.5) + 0.5
        generated_images_np = (generated_images_np * 255).astype(np.uint8)
        
        # 测试检测模型分类结果
        correct_predictions = 0
        predictions = []
        
        for i, img_np in enumerate(generated_images_np):
            # 转换为PIL图像并预处理
            img_pil = Image.fromarray(img_np.transpose(1, 2, 0))
            img_tensor = self.transform(img_np.transpose(1, 2, 0)).unsqueeze(0).to(self.device)
            
            # 通过检测模型分类
            with torch.no_grad():
                outputs = self.detection_model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
            
            predicted_class = self.class_names[predicted.item()]
            predictions.append((predicted_class, confidence))
            
            # 检查是否分类为目标类别（假设目标类别是第一个类别）
            if predicted.item() == 0:
                correct_predictions += 1
            
            logger.info(f"图像 {i+1}: 预测类别={predicted_class}, 置信度={confidence:.4f}")
        
        # 计算准确率
        accuracy = correct_predictions / num_images
        logger.info(f"分类准确率: {accuracy:.4f} ({correct_predictions}/{num_images})")
        
        # 保存测试结果
        self._save_test_results(generated_images_np, predictions)
        
        return accuracy, predictions
    
    def _save_test_results(self, generated_images, predictions):
        """保存测试结果
        
        Args:
            generated_images: 生成的图像
            predictions: 预测结果
        """
        # 创建保存目录
        os.makedirs('test_results', exist_ok=True)
        
        # 保存图像网格
        num_images = len(generated_images)
        rows = (num_images + 3) // 4
        cols = min(4, num_images)
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i, (ax, img, (pred_class, confidence)) in enumerate(zip(axes, generated_images, predictions)):
            img = img.transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_title(f"{pred_class}\nConf: {confidence:.2f}")
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(len(generated_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = 'test_results/generator_test_results.png'
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"测试结果已保存: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试生成模型效果')
    
    # 模型参数
    parser.add_argument('--generator', type=str, 
                        default='trained_generators/generator_epoch_500_class_0.pth',
                        help='生成器模型路径')
    parser.add_argument('--detection_model', type=str, 
                        default='models/character_classifier_best_improved.pth',
                        help='检测模型路径')
    parser.add_argument('--num_images', type=int, default=10, 
                        help='生成图像数量')
    
    args = parser.parse_args()
    
    logger.info('开始测试生成模型...')
    logger.info(f'生成器模型: {args.generator}')
    logger.info(f'检测模型: {args.detection_model}')
    
    # 初始化测试器
    tester = GeneratorTester(
        generator_path=args.generator,
        detection_model_path=args.detection_model
    )
    
    # 测试生成模型
    accuracy, predictions = tester.generate_and_test(
        num_images=args.num_images
    )
    
    logger.info(f'测试完成！分类准确率: {accuracy:.4f}')


if __name__ == "__main__":
    main()
