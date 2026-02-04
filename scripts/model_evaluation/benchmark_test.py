#!/usr/bin/env python3
"""
模型Benchmark测试脚本
评估已有模型的性能指标
"""
import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark_test')


class CharacterDataset(Dataset):
    """字符数据集"""
    
    def __init__(self, root_dir, transform=None, max_classes=26):
        """初始化
        
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
            max_classes: 最大类别数（默认26，与模型训练时一致）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # 构建类别映射（只使用前26个类别，与模型训练时一致）
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])[:max_classes]
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
        
        # 加载图像路径和标签
        for cls, idx in self.class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CharacterClassifier(nn.Module):
    """字符分类器"""
    
    def __init__(self, num_classes=60):
        """初始化
        
        Args:
            num_classes: 类别数量
        """
        super(CharacterClassifier, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )
    
    def forward(self, x):
        return self.backbone(x)


def load_model(model_path, num_classes=60):
    """
    加载模型
    
    Args:
        model_path: 模型路径
        num_classes: 类别数量
        
    Returns:
        加载好的模型
    """
    model = CharacterClassifier(num_classes=num_classes)
    
    # 加载模型权重
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 检查是否有嵌套的model_state_dict
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        logger.info(f"成功加载模型: {model_path}")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return None
    
    return model


def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        性能指标字典
    """
    model.to(device)
    model.eval()
    
    true_labels = []
    pred_labels = []
    inference_times = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 记录推理时间
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            inference_times.append((end_time - start_time) / len(images))
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            
            # 收集标签
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    
    # 计算性能指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    
    # 计算推理速度
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time
    
    # 生成详细的分类报告
    class_report = classification_report(true_labels, pred_labels, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time': avg_inference_time,
        'fps': fps,
        'classification_report': class_report
    }

def main():
    """
    主函数
    """
    # 测试数据集路径
    val_dir = 'data/split_dataset_v2/val'
    
    # 模型路径列表
    model_paths = [
        'models/character_classifier_best.pth',
        'models/character_classifier_best_improved.pth',
        'models/character_classifier_best_v2.pth',
        'models/character_classifier_final.pth',
        'models/character_classifier_final_improved.pth',
        'models/character_classifier_final_v2.pth'
    ]
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    dataset = CharacterDataset(val_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 确定设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 评估每个模型
    results = {}
    for model_path in model_paths:
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            continue
        
        logger.info(f"\n开始评估模型: {model_path}")
        
        # 加载模型（使用26个类别，因为模型是为26个类别训练的）
        model = load_model(model_path, num_classes=26)
        if model is None:
            continue
        
        # 评估模型
        start_time = time.time()
        metrics = evaluate_model(model, dataloader, device)
        eval_time = time.time() - start_time
        
        logger.info(f"评估完成，耗时: {eval_time:.2f}秒")
        logger.info(f"准确率: {metrics['accuracy']:.4f}")
        logger.info(f"精确率: {metrics['precision']:.4f}")
        logger.info(f"召回率: {metrics['recall']:.4f}")
        logger.info(f"F1分数: {metrics['f1']:.4f}")
        logger.info(f"平均推理时间: {metrics['avg_inference_time']:.6f}秒")
        logger.info(f"推理速度: {metrics['fps']:.2f} FPS")
        
        # 保存结果
        model_name = os.path.basename(model_path)
        results[model_name] = metrics
    
    # 生成测试报告
    logger.info("\n" + "="*80)
    logger.info("模型Benchmark测试报告")
    logger.info("="*80)
    
    # 打印摘要
    logger.info("\n性能摘要:")
    logger.info("-"*80)
    logger.info(f"{'模型名称':<40} {'准确率':<10} {'F1分数':<10} {'推理速度(FPS)':<15}")
    logger.info("-"*80)
    
    for model_name, metrics in results.items():
        logger.info(f"{model_name:<40} {metrics['accuracy']:.4f}    {metrics['f1']:.4f}    {metrics['fps']:.2f}")
    
    logger.info("-"*80)
    
    # 找出最佳模型
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        logger.info(f"\n最佳模型: {best_model[0]}")
        logger.info(f"最佳F1分数: {best_model[1]['f1']:.4f}")
        logger.info(f"对应的准确率: {best_model[1]['accuracy']:.4f}")
        logger.info(f"对应的推理速度: {best_model[1]['fps']:.2f} FPS")
    
    logger.info("\n测试完成！")


if __name__ == "__main__":
    main()
