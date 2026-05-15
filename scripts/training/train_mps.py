#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二次元角色识别模型训练 - MPS加速版本
支持飞书推送训练进展
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
import json
import logging
import time
import copy
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载飞书通知配置（与 spider_via_api.py 相同方式）
def load_notification_config():
    """加载通知配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notification_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            notification_config = json.load(f)
        
        # 设置飞书通知环境变量
        os.environ['NOTIFICATION_ENABLED'] = 'true'
        os.environ['NOTIFICATION_PLATFORM'] = notification_config['platform']
        os.environ['FEISHU_APP_ID'] = notification_config['feishu']['app_id']
        os.environ['FEISHU_APP_SECRET'] = notification_config['feishu']['app_secret']
        os.environ['FEISHU_RECEIVE_ID'] = notification_config['feishu']['receive_id']
        os.environ['FEISHU_RECEIVE_ID_TYPE'] = notification_config['feishu']['receive_id_type']
        logger.info(f"✅ 已加载通知配置: {config_path}")
    else:
        logger.warning(f"⚠️ 未找到通知配置文件: {config_path}")

# 加载配置
load_notification_config()

# 自动检测最佳设备
def get_best_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        logger.info("✅ MPS设备可用")
        return torch.device('mps')
    elif torch.cuda.is_available():
        logger.info("✅ CUDA设备可用")
        return torch.device('cuda')
    else:
        logger.info("⚠️ 仅CPU可用")
        return torch.device('cpu')

# 训练配置
DEVICE = get_best_device()
IMAGE_SIZE = 192
BATCH_SIZE = 16  # MPS可以使用更大的batch size
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
PATIENCE = 10
EARLY_STOP_THRESHOLD = 0.001

# 数据路径
DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset'
MODEL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/models/mobilenetv2_loli_74_mps'


def send_feishu_message(title, message):
    """发送飞书消息"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.services.notification_service import send_notification
        success = send_notification(f"**{title}**\n\n{message}", level="info")
        if success:
            logger.info("✅ 飞书通知发送成功")
            return True
        else:
            logger.warning("❌ 飞书通知发送失败（通知服务返回失败）")
            return False
    except Exception as e:
        logger.warning(f"❌ 发送飞书通知失败: {e}")
        return False


def train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    """训练模型"""
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if batch_idx % 50 == 0 and phase == 'train':
                    logger.info(f'  Batch {batch_idx}/{len(dataloaders[phase])}, Loss: {loss.item():.4f}')
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            logger.info(f'  {phase} Loss: {epoch_loss:.4f}, {phase} Acc: {epoch_acc:.4%}')
            
            if phase == 'val':
                progress = (epoch + 1) / num_epochs * 100
                send_feishu_message(f"📊 训练进度 {progress:.1f}%", 
                                   f"Epoch {epoch+1}/{num_epochs}\n验证准确率: {epoch_acc:.4%}\n验证损失: {epoch_loss:.4f}")
                
                if epoch_acc > best_acc + EARLY_STOP_THRESHOLD:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    logger.info(f'  保存最佳模型，准确率: {best_acc:.4%}')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f'  验证准确率连续 {patience} 轮未提升，提前停止训练')
                        model.load_state_dict(best_model_wts)
                        return model, best_acc

    time_elapsed = time.time() - since
    logger.info(f'训练完成，耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'最佳验证准确率: {best_acc:.4%}')
    
    model.load_state_dict(best_model_wts)
    return model, best_acc


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🎬 二次元角色识别模型训练 (74个角色) - MPS加速版")
    logger.info("=" * 60)
    
    # 发送开始通知
    send_feishu_message("🚀 训练任务开始 (MPS加速)", f"""数据集: {DATA_DIR}
配置:
- Epochs: {NUM_EPOCHS}
- Batch Size: {BATCH_SIZE}
- Image Size: {IMAGE_SIZE}
- Learning Rate: {LEARNING_RATE}
- 设备: {DEVICE}

时间: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
    
    # 创建模型目录
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"使用设备: {DEVICE}")
    logger.info(f"加载数据: {DATA_DIR}")
    
    # 简化数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    full_dataset = ImageFolder(DATA_DIR, transform=train_transform)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"数据集: {len(full_dataset)} 样本, {len(full_dataset.classes)} 类别")
    logger.info(f"训练集: {train_size}")
    logger.info(f"验证集: {val_size}")
    
    # 创建数据加载器 (num_workers=2 提高加载速度)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=2, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=2, pin_memory=True)
    }
    
    # 保存类别映射
    class_to_idx = full_dataset.class_to_idx
    with open(Path(MODEL_DIR) / 'class_to_idx.json', 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 保存类别映射")
    
    # 创建轻量模型 MobileNetV2
    num_classes = len(full_dataset.classes)
    logger.info(f"\n创建模型: mobilenet_v2, 类别数: {num_classes}")
    
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    try:
        # 开始训练
        logger.info(f"\n🚀 开始训练")
        model, best_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)
        
        # 保存模型
        torch.save(model.state_dict(), Path(MODEL_DIR) / 'model_best.pth')
        torch.save(model, Path(MODEL_DIR) / 'model_full.pth')
        
        logger.info(f"✅ 模型已保存: {Path(MODEL_DIR) / 'model_best.pth'}")
        logger.info(f"mobilenet_v2 训练完成，准确率: {best_acc:.4%}")
        
        # 保存训练结果
        results = {
            'model': 'mobilenet_v2',
            'accuracy': best_acc.item(),
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'image_size': IMAGE_SIZE,
            'device': str(DEVICE),
            'num_classes': num_classes,
            'train_samples': train_size,
            'val_samples': val_size,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(Path(MODEL_DIR) / 'training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 训练结果已保存")
        
        # 发送完成通知
        send_feishu_message("🎉 训练任务完成", f"""训练配置:
- 数据集: {len(full_dataset.classes)} 个角色, {len(full_dataset)} 张图片
- 模型: MobileNetV2
- Epochs: {NUM_EPOCHS}
- Batch Size: {BATCH_SIZE}
- Image Size: {IMAGE_SIZE}
- 设备: {DEVICE}

训练结果:
- ✅ 最佳验证准确率: {best_acc:.4%}
- 📁 模型保存路径: {MODEL_DIR}

时间: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        send_feishu_message("❌ 训练异常", f"训练失败: {str(e)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎬 训练任务结束")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()