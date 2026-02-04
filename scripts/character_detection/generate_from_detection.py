#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('generate_from_detection')

class CharacterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(root_dir) else []
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        return x * attn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return nn.LeakyReLU(0.2, inplace=True)(self.block(x) + self.shortcut(x))

class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=False)
        self.style = nn.Linear(latent_dim, out_channels * 2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        # 样式调制
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        mean, std = style.chunk(2, dim=1)
        x = x * (std + 1.0) + mean
        return self.act(x)

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, num_layers=4):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=26):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        # 映射网络
        self.mapping = MappingNetwork(latent_dim)
        # 初始块
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        # 样式块
        self.style_blocks = nn.ModuleList([
            StyleBlock(512, 256, latent_dim),  # 14x14
            StyleBlock(256, 128, latent_dim),  # 28x28
            StyleBlock(128, 64, latent_dim),   # 56x56
            StyleBlock(64, 32, latent_dim),    # 112x112
            StyleBlock(32, 16, latent_dim)     # 224x224
        ])
        # 注意力块
        self.attention = AttentionBlock(64)
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()  # 输出 [-1, 1]
        )

    def forward(self, z, labels):
        # 嵌入标签
        label_emb = self.label_emb(labels)
        # 结合噪声和标签嵌入
        z = z + label_emb
        # 映射到样式空间
        style = self.mapping(z)
        # 初始块
        x = self.fc(style).view(-1, 512, 7, 7)
        # 样式块
        for i, block in enumerate(self.style_blocks):
            x = block(x, style)
            # 在中间层添加注意力
            if i == 2:  # 56x56 时添加
                x = self.attention(x)
        # 输出
        return self.output(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_f, out_f, bn=True):
            block = [nn.Conv2d(in_f, out_f, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
            if bn: block.append(nn.BatchNorm2d(out_f))
            return block
        self.features = nn.Sequential(
            *conv_block(3, 64, bn=False),   # 112x112
            *conv_block(64, 128),           # 56x56
            *conv_block(128, 256),          # 28x28
            *conv_block(256, 512),          # 14x14
            nn.AdaptiveAvgPool2d(1)         # 1x1
        )
        self.output = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        output = self.output(features)
        return output.view(-1, 1), features

    def get_features(self, x):
        return self.features(x)

class DetectionGuidedGenerator:
    def __init__(self, detection_model_path, num_classes, latent_dim=128, device=None):
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        self.generator = Generator(latent_dim, num_classes).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # 加载分类器并冻结
        self.classifier = models.efficientnet_b0(num_classes=num_classes).to(self.device)
        state_dict = torch.load(detection_model_path, map_location=self.device)
        self.classifier.load_state_dict(state_dict.get('model_state_dict', state_dict), strict=False)
        self.classifier.eval()
        for p in self.classifier.parameters(): p.requires_grad = False
        
        # 加载VGG感知模型并冻结
        vgg_mod = models.vgg16(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg_mod[:23])).to(self.device).eval() # 到relu4_2
        for p in self.vgg_layers.parameters(): p.requires_grad = False

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        # 归一化辅助：将生成的 [-1, 1] 转为符合 ImageNet 要求的分布
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def normalize(self, img):
        # 从 [-1, 1] 转到 [0, 1] 然后应用 ImageNet 归一化
        img = (img + 1.0) / 2.0
        return (img - self.mean) / self.std

    def train(self, root_dir, target_class=0, num_epochs=3000, batch_size=16):
        os.makedirs('generated_images', exist_ok=True)
        logger.info(f"开始训练，目标类别: {target_class}")
        logger.info(f"训练轮数: {num_epochs}, 批量大小: {batch_size}")
        
        # 数据增强
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        # 类别名称映射
        class_names = ['原神_丽莎', '原神_凯亚', '原神_安柏', '原神_温迪', '原神_琴', '原神_空', '原神_芭芭拉', '原神_荧', '原神_迪卢克', '原神_雷泽', '幻塔_凛夜', '我推的孩子_星野爱', '明日方舟_德克萨斯', '明日方舟_能天使', '明日方舟_阿米娅', '明日方舟_陈', '绝区零_安比', '绝区零_杰克', '蔚蓝档案_优花梨', '蔚蓝档案_宫子', '蔚蓝档案_日奈', '蔚蓝档案_星野', '蔚蓝档案_白子', '蔚蓝档案_阿罗娜', '鸣潮_守岸人', '鸣潮_椿']
        
        # 加载数据集
        if target_class < len(class_names):
            class_name = class_names[target_class]
        else:
            class_name = f'class_{target_class}'
        
        dataset_path = os.path.join(root_dir, class_name)
        logger.info(f"加载数据集: {dataset_path}")
        
        # 验证数据集是否存在
        if not os.path.exists(dataset_path):
            logger.error(f"数据集路径不存在: {dataset_path}")
            logger.error("请检查数据目录结构是否正确，或选择其他可用的目标类别")
            logger.error("可用的类别包括: 原神_丽莎 (0), 原神_凯亚 (1), 原神_安柏 (2), 原神_温迪 (3), 原神_琴 (4), 原神_空 (5), 原神_芭芭拉 (6), 原神_荧 (7), 原神_迪卢克 (8), 原神_雷泽 (9), 幻塔_凛夜 (10), 我推的孩子_星野爱 (11), 明日方舟_德克萨斯 (12), 明日方舟_能天使 (13), 明日方舟_阿米娅 (14), 明日方舟_陈 (15), 绝区零_安比 (16), 绝区零_杰克 (17), 蔚蓝档案_优花梨 (18), 蔚蓝档案_宫子 (19), 蔚蓝档案_日奈 (20), 蔚蓝档案_星野 (21), 蔚蓝档案_白子 (22), 蔚蓝档案_阿罗娜 (23), 鸣潮_守岸人 (24), 鸣潮_椿 (25)")
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        dataset = CharacterDataset(dataset_path, transform=transform)
        dataset_size = len(dataset)
        logger.info(f"数据集大小: {dataset_size}")
        
        # 检查数据集大小
        if dataset_size == 0:
            logger.error(f"数据集为空，请检查数据集路径是否包含图像文件")
            logger.error("请检查数据目录结构是否正确，或选择其他可用的目标类别")
            logger.error("可用的类别包括: 原神_丽莎 (0), 原神_凯亚 (1), 原神_安柏 (2), 原神_温迪 (3), 原神_琴 (4), 原神_空 (5), 原神_芭芭拉 (6), 原神_荧 (7), 原神_迪卢克 (8), 原神_雷泽 (9), 幻塔_凛夜 (10), 我推的孩子_星野爱 (11), 明日方舟_德克萨斯 (12), 明日方舟_能天使 (13), 明日方舟_阿米娅 (14), 明日方舟_陈 (15), 绝区零_安比 (16), 绝区零_杰克 (17), 蔚蓝档案_优花梨 (18), 蔚蓝档案_宫子 (19), 蔚蓝档案_日奈 (20), 蔚蓝档案_星野 (21), 蔚蓝档案_白子 (22), 蔚蓝档案_阿罗娜 (23), 鸣潮_守岸人 (24), 鸣潮_椿 (25)")
            raise ValueError(f"数据集为空，无法进行训练。请选择其他可用的数据集")
        
        # 检查数据集大小
        if len(dataset) < batch_size:
            logger.warning(f"数据集大小 ({len(dataset)}) 小于批量大小 ({batch_size})，调整批量大小为 {len(dataset)}")
            batch_size = len(dataset)
            
        if batch_size < 1:
            logger.error(f"数据集大小为 0，无法进行训练")
            raise ValueError(f"数据集大小为 0，无法进行训练")
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        logger.info(f"数据加载器创建完成，批次数量: {len(dataloader)}")
        
        # 检查批次数量
        if len(dataloader) == 0:
            logger.warning(f"批次数量为 0，无法进行训练")
            logger.warning(f"请增加数据集大小或减小批量大小")
            # 生成一些样本，然后退出
            logger.info("生成一些样本...")
            self.save_samples(0, target_class)
            return
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            total_d_loss = 0.0
            total_g_adv = 0.0
            total_g_feat = 0.0
            total_g_cls = 0.0
            total_g_per = 0.0
            total_g_tv = 0.0
            total_g_loss = 0.0
            
            for i, real_imgs in enumerate(dataloader):
                real_imgs = real_imgs.to(self.device)
                current_batch_size = real_imgs.size(0)
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                labels = torch.full((current_batch_size,), target_class, dtype=torch.long, device=self.device)
                
                # ---------------------
                #  训练判别器
                # ---------------------
                self.optimizer_d.zero_grad()
                fake_imgs = self.generator(z, labels)
                
                # 判别器前向传播，获取输出和特征
                d_out_real, d_feat_real = self.discriminator(real_imgs)
                d_out_fake, d_feat_fake = self.discriminator(fake_imgs.detach())
                
                # 判别器损失（带标签平滑）
                d_loss = (nn.BCELoss()(d_out_real, torch.ones(current_batch_size, 1, device=self.device)*0.9) +
                          nn.BCELoss()(d_out_fake, torch.zeros(current_batch_size, 1, device=self.device))) / 2
                d_loss.backward()
                self.optimizer_d.step()

                # ---------------------
                #  训练生成器
                # ---------------------
                self.optimizer_g.zero_grad()
                
                # 1. 对抗损失
                d_out_fake, d_feat_fake = self.discriminator(fake_imgs)
                g_adv = nn.BCELoss()(d_out_fake, torch.ones(current_batch_size, 1, device=self.device))
                
                # 2. 特征匹配损失
                _, d_feat_real = self.discriminator(real_imgs)
                g_feat = nn.MSELoss()(d_feat_fake, d_feat_real)
                
                # 3. 分类引导（关键：输入前需归一化）
                norm_fake = self.normalize(fake_imgs)
                cls_out = self.classifier(norm_fake)
                g_cls = nn.CrossEntropyLoss()(cls_out, labels)
                
                # 4. 感知损失（VGG）
                norm_real = self.normalize(real_imgs)
                g_per = nn.MSELoss()(self.vgg_layers(norm_fake), self.vgg_layers(norm_real))
                
                # 5. TV Loss
                g_tv = torch.mean(torch.abs(fake_imgs[:, :, :, :-1] - fake_imgs[:, :, :, 1:])) + \
                       torch.mean(torch.abs(fake_imgs[:, :, :-1, :] - fake_imgs[:, :, 1:, :]))

                # 权重分配：感知损失引导形状，分类损失引导特征，对抗损失引导纹理
                g_loss = 0.1 * g_adv + 0.5 * g_feat + 3.0 * g_cls + 20.0 * g_per + 0.01 * g_tv
                g_loss.backward()
                self.optimizer_g.step()
                
                # 累计损失
                total_d_loss += d_loss.item()
                total_g_adv += g_adv.item()
                total_g_feat += g_feat.item()
                total_g_cls += g_cls.item()
                total_g_per += g_per.item()
                total_g_tv += g_tv.item()
                total_g_loss += g_loss.item()

            # 计算平均损失
            avg_d_loss = total_d_loss / len(dataloader)
            avg_g_adv = total_g_adv / len(dataloader)
            avg_g_feat = total_g_feat / len(dataloader)
            avg_g_cls = total_g_cls / len(dataloader)
            avg_g_per = total_g_per / len(dataloader)
            avg_g_tv = total_g_tv / len(dataloader)
            avg_g_loss = total_g_loss / len(dataloader)
            
            # 计算训练时间
            epoch_time = time.time() - epoch_start_time
            
            # 记录日志
            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
            logger.info(f"时间: {epoch_time:.2f}秒")
            logger.info(f"判别器损失: {avg_d_loss:.4f}")
            logger.info(f"生成器损失: {avg_g_loss:.4f}")
            logger.info(f"  对抗损失: {avg_g_adv:.4f}")
            logger.info(f"  特征匹配损失: {avg_g_feat:.4f}")
            logger.info(f"  分类损失: {avg_g_cls:.4f}")
            logger.info(f"  感知损失: {avg_g_per:.4f}")
            logger.info(f"  总变差损失: {avg_g_tv:.4f}")

            if (epoch + 1) % 20 == 0:
                logger.info(f"生成图像，保存到 generated_images/epoch_{epoch+1}.png")
                self.save_samples(epoch+1, target_class)

    def save_samples(self, epoch, target_class):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(8, self.latent_dim, device=self.device)
            labels = torch.full((8,), target_class, dtype=torch.long, device=self.device)
            imgs = (self.generator(z, labels).cpu() + 1.0) / 2.0
        
        grid = np.transpose(imgs.numpy(), (0, 2, 3, 1))
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(np.clip(grid[i], 0, 1))
            ax.axis('off')
        plt.savefig(f'generated_images/epoch_{epoch}.png')
        plt.close()
        self.generator.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/character_classifier_best_improved.pth')
    parser.add_argument('--data_dir', type=str, default='data/all_characters')
    parser.add_argument('--target_class', type=int, default=23, help='目标类别，23 为阿罗娜')
    args = parser.parse_args()

    trainer = DetectionGuidedGenerator(args.model_path, num_classes=26)
    trainer.train(args.data_dir, target_class=args.target_class)