#!/usr/bin/env python3
"""
基于检测模型反推的图像生成脚本
使用预训练的角色分类模型作为指导，生成符合特定角色类别的图像
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import logging

class CharacterDataset(Dataset):
    """角色图像数据集，用于加载真实参考图像
    
    Args:
        root_dir: 数据集根目录
        transform: 图像变换
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) 
                          if img.endswith('.jpg') or img.endswith('.png')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('generate_from_detection')


class View(nn.Module):
    """重塑层"""
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)


class Generator(nn.Module):
    """简化版生成器模型，专为二次元角色设计
    
    从随机噪声生成224x224x3的RGB图像
    """
    
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
    """角色分类器模型（用于加载预训练权重）"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用EfficientNet-B0作为基础模型
        self.backbone = models.efficientnet_b0(pretrained=True)
        # 替换分类头
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 
            num_classes
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


class Discriminator(nn.Module):
    """判别器网络，用于判断图像是真实的还是生成的"""
    
    def __init__(self):
        """初始化判别器"""
        super().__init__()
        
        self.model = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 1x1
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像 [batch_size, 3, 224, 224]
            
        Returns:
            判别结果 [batch_size, 1]
        """
        output = self.model(x)
        return output.view(output.size(0), 1)


class DetectionGuidedGenerator:
    """基于检测模型指导的生成器训练器"""
    
    def __init__(self, detection_model_path, num_classes, latent_dim=128, device=None):
        """初始化
        
        Args:
            detection_model_path: 预训练检测模型路径
            num_classes: 类别数量
            latent_dim: 潜在空间维度
            device: 训练设备
        """
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # 初始化生成器
        self.generator = Generator(latent_dim=latent_dim).to(self.device)
        
        # 初始化判别器
        self.discriminator = Discriminator().to(self.device)
        
        # 加载预训练的检测模型
        self.detection_model = CharacterClassifier(num_classes=num_classes).to(self.device)
        checkpoint = torch.load(detection_model_path, map_location=self.device)
        self.detection_model.load_state_dict(checkpoint['model_state_dict'])
        self.detection_model.eval()  # 设置为评估模式
        
        # 冻结检测模型参数
        for param in self.detection_model.parameters():
            param.requires_grad = False
        
        # 加载预训练VGG16用于感知损失
        self.vgg = models.vgg16(pretrained=True).features.to(self.device)
        # 冻结VGG参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 优化器 - 调整初始学习率
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-5)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-5)
        
        # 学习率调度器 - 调整衰减策略
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_g, T_max=1000, eta_min=1e-7)
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_d, T_max=1000, eta_min=1e-7)
        
        # 初始化损失函数
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()
        self.criterion_mse = nn.MSELoss()
        
        # 加载真实角色图像数据集用于感知损失
        self.real_data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] 归一化
        ])
        
        # 加载原神_丽莎的真实图像
        lisa_data_dir = 'data/all_characters/原神_丽莎'
        if os.path.exists(lisa_data_dir):
            self.real_dataset = CharacterDataset(lisa_data_dir, transform=self.real_data_transform)
            logger.info(f"真实角色图像数据集加载成功，包含 {len(self.real_dataset)} 张原神_丽莎图像")
        else:
            self.real_dataset = None
            logger.warning("真实角色图像数据集未找到，将使用随机噪声作为目标")
        
        # 初始化真实数据加载器为None，在train方法中根据batch_size创建
        self.real_dataloader = None
        
        logger.info(f"初始化完成，使用设备: {self.device}")
        logger.info(f"检测模型加载成功: {detection_model_path}")
        logger.info(f"潜在空间维度: {latent_dim}")
        logger.info("VGG网络加载成功，用于感知损失计算")
        logger.info("判别器网络初始化完成")
    
    def tv_loss(self, img):
        """总变分损失，用于抑制高频噪声
        
        Args:
            img: 生成的图像 [batch_size, 3, 224, 224]
            
        Returns:
            总变分损失值
        """
        # 计算水平方向的差异
        horizontal_diff = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        # 计算垂直方向的差异
        vertical_diff = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        return horizontal_diff + vertical_diff
    
    def perceptual_loss(self, generated_img, target_img):
        """感知损失，计算生成图像与目标图像在VGG特征空间的距离
        
        Args:
            generated_img: 生成的图像 [batch_size, 3, 224, 224]
            target_img: 目标图像 [batch_size, 3, 224, 224]
            
        Returns:
            感知损失值
        """
        # 计算生成图像的VGG特征
        gen_features = self.vgg(generated_img)
        # 计算目标图像的VGG特征
        target_features = self.vgg(target_img)
        # 计算特征距离
        return self.criterion_mse(gen_features, target_features)
    
    def diversity_loss(self, generated_imgs):
        """多样性损失，惩罚批量内图像相似度
        
        Args:
            generated_imgs: 生成的图像 [batch_size, 3, 224, 224]
            
        Returns:
            多样性损失值
        """
        batch_size = generated_imgs.size()[0]
        if batch_size < 2:
            return 0
        
        # 展平图像
        imgs_flat = generated_imgs.view(batch_size, -1)
        
        # 计算批量内距离（避免使用torch.pdist，兼容MPS）
        # 计算所有图像对之间的L2距离
        distances = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                dist = torch.norm(imgs_flat[i] - imgs_flat[j])
                distances.append(dist)
        
        if not distances:
            return 0
        
        # 计算平均距离的倒数作为损失
        avg_dist = torch.mean(torch.stack(distances))
        eps = 1e-8
        return 1.0 / (avg_dist + eps)
    
    def train(self, num_epochs=2000, batch_size=16, target_class=0, save_interval=200):
        """训练生成器
        
        Args:
            num_epochs: 训练轮数
            batch_size: 批量大小
            target_class: 目标角色类别
            save_interval: 保存间隔
        """
        logger.info(f"开始训练，目标类别: {target_class}")
        logger.info(f"训练轮数: {num_epochs}, 批量大小: {batch_size}")
        
        # 创建保存目录
        os.makedirs('generated_images', exist_ok=True)
        os.makedirs('trained_generators', exist_ok=True)
        # 确保目录存在
        if not os.path.exists('generated_images'):
            os.makedirs('generated_images')
        if not os.path.exists('trained_generators'):
            os.makedirs('trained_generators')
        
        # 固定噪声用于生成样本
        fixed_noise = torch.randn(8, self.latent_dim, device=self.device)
        
        # 根据batch_size创建真实数据加载器
        if hasattr(self, 'real_dataset') and self.real_dataset is not None:
            self.real_dataloader = DataLoader(self.real_dataset, batch_size=batch_size, shuffle=True)
            logger.info(f"创建真实数据加载器，批量大小: {batch_size}")
        
        # 加载真实的原神_丽莎图像作为目标参考
        if hasattr(self, 'real_dataloader') and self.real_dataloader is not None:
            try:
                # 尝试从数据加载器获取真实图像
                real_batch = next(iter(self.real_dataloader))
                # 确保批量大小匹配
                if real_batch.size(0) >= batch_size:
                    target_img = real_batch[:batch_size].to(self.device)
                else:
                    # 如果批量不足，重复填充
                    repeat_times = (batch_size + real_batch.size(0) - 1) // real_batch.size(0)
                    target_img = real_batch.repeat(repeat_times, 1, 1, 1)[:batch_size].to(self.device)
                logger.info(f"使用真实原神_丽莎图像作为目标参考")
            except StopIteration:
                # 如果数据加载器迭代完毕，重置迭代器
                self.real_dataloader = DataLoader(self.real_dataset, batch_size=batch_size, shuffle=True)
                real_batch = next(iter(self.real_dataloader))
                target_img = real_batch[:batch_size].to(self.device)
        else:
            # 回退到基于目标类别的参考图像
            target_img = torch.randn(batch_size, 3, 224, 224, device=self.device)
            # 调整颜色分布，模拟丽莎的紫色头发和黄色元素
            target_img[:, 0, :, :] *= 0.8  # 红色通道减弱
            target_img[:, 1, :, :] *= 0.9  # 绿色通道轻微减弱
            target_img[:, 2, :, :] *= 1.2  # 蓝色通道增强（紫色倾向）
            target_img = torch.clamp(target_img, -1, 1)
            logger.warning("使用随机噪声作为目标参考")
        
        for epoch in range(num_epochs):
            # 生成随机噪声
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # 生成图像
            generated_images = self.generator(noise)
            
            # 强化数据增强
            augmented_images = generated_images.clone()
            
            # 1. 更强烈的随机旋转 (-90到90度)
            for i in range(batch_size):
                angle = random.uniform(-90, 90)
                img_pil = Image.fromarray((((augmented_images[i].detach().cpu().numpy() * 0.5) + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0))
                img_pil = img_pil.rotate(angle, expand=False)
                augmented_images[i] = torch.tensor(np.array(img_pil).transpose(2, 0, 1) / 255.0 * 2 - 1, device=self.device, dtype=torch.float32)
            
            # 2. 水平翻转，增加图像多样性
            for i in range(batch_size):
                if random.random() > 0.5:
                    img_pil = Image.fromarray((((augmented_images[i].detach().cpu().numpy() * 0.5) + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0))
                    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    augmented_images[i] = torch.tensor(np.array(img_pil).transpose(2, 0, 1) / 255.0 * 2 - 1, device=self.device, dtype=torch.float32)
            
            # 3. 更大范围的随机裁剪和调整大小
            for i in range(batch_size):
                crop_size = random.randint(150, 200)
                img_pil = Image.fromarray((((augmented_images[i].detach().cpu().numpy() * 0.5) + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0))
                left = random.randint(0, 224 - crop_size)
                top = random.randint(0, 224 - crop_size)
                img_pil = img_pil.crop((left, top, left + crop_size, top + crop_size))
                img_pil = img_pil.resize((224, 224))
                augmented_images[i] = torch.tensor(np.array(img_pil).transpose(2, 0, 1) / 255.0 * 2 - 1, device=self.device, dtype=torch.float32)
            
            # 4. 随机模糊
            for i in range(batch_size):
                img_pil = Image.fromarray((((augmented_images[i].detach().cpu().numpy() * 0.5) + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0))
                if random.random() > 0.5:
                    blur_radius = random.uniform(0.5, 2.0)
                    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                augmented_images[i] = torch.tensor(np.array(img_pil).transpose(2, 0, 1) / 255.0 * 2 - 1, device=self.device, dtype=torch.float32)
            
            # 5. 随机色彩调整
            for i in range(batch_size):
                img_pil = Image.fromarray((((augmented_images[i].detach().cpu().numpy() * 0.5) + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0))
                if random.random() > 0.5:
                    # 随机对比度调整
                    factor = random.uniform(0.7, 1.3)
                    img_pil = ImageEnhance.Contrast(img_pil).enhance(factor)
                if random.random() > 0.5:
                    # 随机亮度调整
                    factor = random.uniform(0.7, 1.3)
                    img_pil = ImageEnhance.Brightness(img_pil).enhance(factor)
                if random.random() > 0.5:
                    # 随机饱和度调整
                    factor = random.uniform(0.7, 1.3)
                    img_pil = ImageEnhance.Color(img_pil).enhance(factor)
                augmented_images[i] = torch.tensor(np.array(img_pil).transpose(2, 0, 1) / 255.0 * 2 - 1, device=self.device, dtype=torch.float32)
            
            # 6. 添加随机噪声，调整噪声强度
            noise_aug = torch.randn_like(augmented_images) * random.uniform(0.02, 0.04)
            augmented_images = augmented_images + noise_aug
            augmented_images = torch.clamp(augmented_images, -1, 1)
            
            # 每2个epoch训练一次判别器，避免判别器过强
            if epoch % 2 == 0:
                # 训练判别器
                self.optimizer_d.zero_grad()
                
                # 判别真实图像（使用目标图像作为伪真实图像）
                real_outputs = self.discriminator(target_img)
                real_labels = torch.ones(batch_size, 1, device=self.device)
                loss_d_real = self.criterion_bce(real_outputs, real_labels)
                
                # 判别生成图像
                fake_outputs = self.discriminator(augmented_images.detach())
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                loss_d_fake = self.criterion_bce(fake_outputs, fake_labels)
                
                # 总判别器损失
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                self.optimizer_d.step()
            
            # 训练生成器
            self.optimizer_g.zero_grad()
            
            # 通过检测模型获取分类结果
            detection_outputs = self.detection_model(augmented_images)
            
            # 计算交叉熵损失
            target_labels = torch.full((batch_size,), target_class, dtype=torch.long, device=self.device)
            loss_ce = self.criterion_ce(detection_outputs, target_labels)
            
            # 计算对抗损失
            fake_outputs = self.discriminator(augmented_images)
            loss_adv = self.criterion_bce(fake_outputs, real_labels)
            
            # 计算感知损失
            loss_perceptual = self.perceptual_loss(augmented_images, target_img)
            
            # 计算总变分损失
            loss_tv = self.tv_loss(augmented_images)
            
            # 计算多样性损失
            loss_div = self.diversity_loss(augmented_images)
            
            # 计算目标类别概率损失，直接最大化目标类别的概率
            target_probs = torch.softmax(detection_outputs, dim=1)[:, target_class]
            loss_target_prob = -torch.mean(torch.log(target_probs + 1e-8))
            
            # 总生成器损失 - 增加目标类别概率损失的权重
            loss_g = 0.5 * loss_ce + 0.2 * loss_adv + 0.1 * loss_perceptual + 1e-8 * loss_tv + 0.2 * loss_div + 1.0 * loss_target_prob
            loss_g.backward()
            self.optimizer_g.step()
            
            # 更新学习率
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # 记录进度
            if (epoch + 1) % 10 == 0:
                current_lr_g = self.optimizer_g.param_groups[0]['lr']
                current_lr_d = self.optimizer_d.param_groups[0]['lr']
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], ')
                logger.info(f'  Generator Loss: CE={loss_ce.item():.4f}, ADV={loss_adv.item():.4f}, PER={loss_perceptual.item():.4f}, TV={loss_tv.item():.4f}, DIV={loss_div.item():.4f}, Total={loss_g.item():.4f}')
                logger.info(f'  Discriminator Loss: {loss_d.item():.4f}')
                logger.info(f'  LR: G={current_lr_g:.6f}, D={current_lr_d:.6f}')
            
            # 保存生成的样本
            if (epoch + 1) % save_interval == 0:
                self.generate_samples(fixed_noise, epoch + 1, target_class)
                
                # 保存生成器和判别器
                generator_path = f'trained_generators/generator_epoch_{epoch+1}_class_{target_class}.pth'
                discriminator_path = f'trained_generators/discriminator_epoch_{epoch+1}_class_{target_class}.pth'
                
                torch.save({
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.optimizer_g.state_dict(),
                    'scheduler_state_dict': self.scheduler_g.state_dict(),
                    'epoch': epoch,
                    'loss': loss_g.item(),
                    'latent_dim': self.latent_dim
                }, generator_path)
                
                torch.save({
                    'model_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.optimizer_d.state_dict(),
                    'scheduler_state_dict': self.scheduler_d.state_dict(),
                    'epoch': epoch,
                    'loss': loss_d.item()
                }, discriminator_path)
                
                logger.info(f'生成器已保存: {generator_path}')
                logger.info(f'判别器已保存: {discriminator_path}')
    
    def generate_samples(self, noise, epoch, target_class):
        """生成样本并保存
        
        Args:
            noise: 随机噪声
            epoch: 当前轮数
            target_class: 目标类别
        """
        self.generator.eval()
        with torch.no_grad():
            generated_images = self.generator(noise)
        self.generator.train()
        
        # 反标准化并转换为PIL图像
        generated_images = generated_images.cpu().numpy()
        generated_images = (generated_images * 0.5) + 0.5  # 反Tanh激活
        generated_images = (generated_images * 255).astype(np.uint8)
        
        # 保存图像
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < generated_images.shape[0]:
                img = generated_images[i].transpose(1, 2, 0)
                ax.imshow(img)
                ax.axis('off')
        
        plt.tight_layout()
        save_path = f'generated_images/samples_epoch_{epoch}_class_{target_class}.png'
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f'生成样本已保存: {save_path}')
    
    def generate(self, num_images=10, target_class=0):
        """生成指定数量的图像
        
        Args:
            num_images: 生成图像数量
            target_class: 目标类别
            
        Returns:
            生成的图像列表
        """
        self.generator.eval()
        
        noise = torch.randn(num_images, self.latent_dim, device=self.device)
        
        with torch.no_grad():
            generated_images = self.generator(noise)
        
        self.generator.train()
        
        # 反标准化
        generated_images = generated_images.cpu().numpy()
        generated_images = (generated_images * 0.5) + 0.5
        generated_images = (generated_images * 255).astype(np.uint8)
        
        return generated_images


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于检测模型反推的图像生成')
    
    # 模型参数
    parser.add_argument('--detection_model', type=str, 
                        default='models/character_classifier_best_improved.pth',
                        help='预训练检测模型路径')
    parser.add_argument('--num_classes', type=int, default=26, 
                        help='类别数量')
    parser.add_argument('--latent_dim', type=int, default=256, 
                        help='潜在空间维度')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3000, 
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='批量大小')
    parser.add_argument('--target_class', type=int, default=0, 
                        help='目标角色类别')
    parser.add_argument('--save_interval', type=int, default=100, 
                        help='保存间隔')
    
    args = parser.parse_args()
    
    logger.info('开始基于检测模型反推生成模型...')
    logger.info(f'检测模型: {args.detection_model}')
    logger.info(f'目标类别: {args.target_class}')
    logger.info(f'训练轮数: {args.num_epochs}')
    
    # 初始化生成器
    generator = DetectionGuidedGenerator(
        detection_model_path=args.detection_model,
        num_classes=args.num_classes,
        latent_dim=args.latent_dim
    )
    
    # 开始训练
    generator.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        target_class=args.target_class,
        save_interval=args.save_interval
    )
    
    logger.info('训练完成！')


if __name__ == "__main__":
    main()
