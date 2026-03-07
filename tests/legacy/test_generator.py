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


class UpsampleBlock(nn.Module):
    """上采样块：Upsample + Conv2d + BN + LeakyReLU
    代替 ConvTranspose2d 以消除棋盘格效应
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    """条件生成器模型，专为二次元角色设计
    使用 Upsample + Conv2d 结构
    """
    def __init__(self, latent_dim=128, num_classes=26):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        
        # 输入处理：将噪声映射到 7x7x256 的特征图
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 上采样网络
        self.upsample = nn.Sequential(
            # 7x7 -> 14x14
            UpsampleBlock(256, 256),
            # 14x14 -> 28x28
            UpsampleBlock(256, 128),
            # 28x28 -> 56x56
            UpsampleBlock(128, 64),
            # 56x56 -> 112x112
            UpsampleBlock(64, 32),
            # 112x112 -> 224x224
            UpsampleBlock(32, 16),
            # 输出层
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()  # 改为 Sigmoid 以匹配 ImageNet 标准化
        )
    
    def forward(self, z, labels=None):
        """前向传播
        
        Args:
            z: 随机噪声 [batch_size, latent_dim]
            labels: 类别标签 [batch_size]
            
        Returns:
            生成的图像 [batch_size, 3, 224, 224]
        """
        # 如果没有提供标签，使用默认标签 0
        if labels is None:
            labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        
        # 处理类别嵌入
        class_emb = self.class_embedding(labels)
        # 连接噪声和类别嵌入
        z = torch.cat([z, class_emb], dim=1)
        
        # 处理输入噪声
        x = self.fc(z)
        # 重塑为特征图
        x = x.view(-1, 256, 7, 7)
        # 上采样生成图像
        x = self.upsample(x)
        return x


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
        # 处理可能的 state_dict 键名不匹配问题
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # 从检查点中读取潜在空间维度
            self.latent_dim = checkpoint.get('latent_dim', 128)
        else:
            state_dict = checkpoint
            # 直接使用固定的潜在空间维度
            self.latent_dim = 128
        # 使用固定的类别数量，与训练时一致
        num_classes = 26
        
        # 检查state_dict中的fc.0.weight形状，确定输入维度
        if 'fc.0.weight' in state_dict:
            input_dim = state_dict['fc.0.weight'].shape[1]
            logger.info(f"从state_dict中检测到输入维度: {input_dim}")
            # 计算正确的latent_dim
            if input_dim == 256:
                # 输入维度是256，latent_dim应该是128
                correct_latent_dim = 128
            else:
                # 输入维度不是256，使用默认值
                correct_latent_dim = 128
            
            # 创建一个临时模型来加载权重，使用正确的输入维度
            class TempGenerator(nn.Module):
                def __init__(self, latent_dim=128, num_classes=26):
                    super().__init__()
                    self.latent_dim = latent_dim
                    self.num_classes = num_classes
                    
                    # 类别嵌入
                    self.class_embedding = nn.Embedding(num_classes, latent_dim)
                    
                    # 输入处理
                    self.fc = nn.Sequential(
                        nn.Linear(input_dim, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Linear(512, 256 * 7 * 7),
                        nn.BatchNorm1d(256 * 7 * 7),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                    
                    # 上采样网络 - 使用UpsampleBlock，与训练时保持一致
                    self.upsample = nn.Sequential(
                        # 7x7 -> 14x14
                        UpsampleBlock(256, 256),
                        # 14x14 -> 28x28
                        UpsampleBlock(256, 128),
                        # 28x28 -> 56x56
                        UpsampleBlock(128, 64),
                        # 56x56 -> 112x112
                        UpsampleBlock(64, 32),
                        # 112x112 -> 224x224
                        UpsampleBlock(32, 16),
                        # 输出层
                        nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.Sigmoid()
                    )
                
                def forward(self, z, labels=None):
                    # 如果没有提供标签，使用默认标签 0
                    if labels is None:
                        labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
                    
                    # 处理类别嵌入
                    class_emb = self.class_embedding(labels)
                    # 连接噪声和类别嵌入
                    z = torch.cat([z, class_emb], dim=1)
                    
                    # 处理输入噪声
                    x = self.fc(z)
                    # 重塑为特征图
                    x = x.view(x.size(0), 256, 7, 7)
                    # 上采样生成图像
                    x = self.upsample(x)
                    return x
            
            self.generator = TempGenerator(latent_dim=correct_latent_dim, num_classes=num_classes).to(self.device)
            # 更新latent_dim
            self.latent_dim = correct_latent_dim
        else:
            # 如果没有fc.0.weight，使用默认模型
            self.generator = Generator(latent_dim=self.latent_dim, num_classes=num_classes).to(self.device)
        
        # 加载state_dict，忽略不匹配的键
        self.generator.load_state_dict(state_dict, strict=False)
        self.generator.eval()
        logger.info(f"生成器潜在空间维度: {self.latent_dim}")
        logger.info(f"生成器类别数量: {num_classes}")
        logger.info("使用 strict=False 加载模型权重，忽略结构不匹配的键")
        
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
        
        # 创建目标类别标签（使用第一个类别，即原神_丽莎）
        target_labels = torch.full((num_images,), 0, dtype=torch.long, device=self.device)
        
        # 尝试使用条件GAN方式生成图像
        self.generator.eval()
        try:
            # 生成图像（条件GAN）
            with torch.no_grad():
                generated_images = self.generator(noise, target_labels)
        except TypeError:
            # 如果是旧的非条件GAN模型，只传递噪声参数
            with torch.no_grad():
                generated_images = self.generator(noise)
        
        # 反标准化 - 由于使用了 Sigmoid 激活函数，生成的图像值范围已经是 [0, 1]
        generated_images_np = generated_images.cpu().numpy()
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
