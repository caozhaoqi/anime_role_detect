import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

# 配置日志
logger.add("test_model_performance.log", rotation="500 MB")

# 模型路径
MODEL_DIR = Path("../models")

# 测试数据路径
TEST_DATA_DIR = Path("../data/downloaded_images")

# 类别映射
CLASS_MAPPING = {
    "阿罗娜": "蔚蓝档案_阿罗娜",
    "普拉娜": "蔚蓝档案_普拉娜"
}

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CharacterDataset(Dataset):
    """自定义数据集"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 加载数据
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                # 转换类别名
                mapped_class = CLASS_MAPPING.get(class_name, class_name)
                for img_name in os.listdir(class_path):
                    # 跳过SVG文件
                    if img_name.endswith(".svg"):
                        continue
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(mapped_class)
    
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

def load_model(model_path):
    """加载模型"""
    try:
        # 尝试加载模型
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 检查是否是状态字典
        if isinstance(model_data, dict):
            # 尝试从状态字典中恢复模型
            # 检查是否包含 model_state_dict
            if 'model_state_dict' in model_data:
                state_dict = model_data['model_state_dict']
            elif 'model' in model_data:
                state_dict = model_data['model']
            elif 'state_dict' in model_data:
                state_dict = model_data['state_dict']
            else:
                state_dict = model_data
            
            # 检查模型类型
            model_type = model_data.get('model_type', 'mobilenet_v2')
            
            # 检查类别数量
            num_classes = 2  # 默认类别数
            if 'class_to_idx' in model_data:
                num_classes = len(model_data['class_to_idx'])
            
            # 根据模型类型创建模型
            if model_type == 'efficientnet':
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(pretrained=False)
                # 修改分类器
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif model_type == 'resnet18':
                from torchvision.models import resnet18
                model = resnet18(pretrained=False)
                # 修改分类器
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                from torchvision.models import mobilenet_v2
                model = mobilenet_v2(pretrained=False)
                # 修改分类器
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            
            # 加载状态字典
            model.load_state_dict(state_dict)
        else:
            # 直接使用加载的模型
            model = model_data
        
        model.eval()
        return model
    except Exception as e:
        logger.error(f"加载模型失败 {model_path}: {e}")
        return None

def get_model_classes(model):
    """获取模型的类别"""
    if hasattr(model, 'classes'):
        return model.classes
    elif hasattr(model, 'class_to_idx'):
        return list(model.class_to_idx.keys())
    else:
        logger.warning("无法获取模型类别，使用默认类别")
        return list(CLASS_MAPPING.values())

def test_model(model, test_loader, model_name):
    """测试模型"""
    logger.info(f"测试模型: {model_name}")
    
    # 获取模型类别
    model_classes = get_model_classes(model)
    logger.info(f"模型类别: {model_classes}")
    
    # 预测结果
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            # 转换标签为索引
            label_indices = []
            for label in labels:
                # 查找最匹配的类别
                matched = False
                for i, cls in enumerate(model_classes):
                    if label in cls or cls in label:
                        label_indices.append(i)
                        matched = True
                        break
                if not matched:
                    # 如果没有匹配的类别，添加为未知类别
                    label_indices.append(len(model_classes))
            
            # 预测
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 保存结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_indices)
    
    # 计算准确率
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    total = len(all_labels)
    accuracy = correct / total
    logger.info(f"准确率: {accuracy:.4f} ({correct}/{total})")
    
    # 生成分类报告
    # 获取所有唯一的预测类别
    unique_preds = set(all_preds)
    max_class = max(unique_preds) if unique_preds else 0
    
    # 生成目标类别名称
    target_names = model_classes + ["未知"]
    # 如果预测类别超过目标类别数量，添加更多类别名称
    while len(target_names) <= max_class:
        target_names.append(f"类别_{len(target_names)}")
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, labels=list(range(len(target_names))))
    logger.info(f"分类报告:\n{report}")
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    logger.info(f"混淆矩阵已保存为 {model_name}_confusion_matrix.png")
    
    return accuracy, report

def main():
    """主函数"""
    # 创建测试数据集
    test_dataset = CharacterDataset(TEST_DATA_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    logger.info(f"测试数据集大小: {len(test_dataset)}")
    logger.info(f"测试数据类别: {set(test_dataset.labels)}")
    
    # 遍历所有模型（递归搜索子目录）
    model_files = []
    for root, _, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith(".pth") or file.endswith(".pt"):
                model_files.append(Path(root) / file)
    
    logger.info(f"找到 {len(model_files)} 个模型文件")
    
    results = []
    
    for model_file in model_files:
        model_name = model_file.stem
        logger.info(f"\n=======================================")
        logger.info(f"测试模型: {model_name}")
        logger.info(f"模型路径: {model_file}")
        
        # 加载模型
        model = load_model(model_file)
        if model is None:
            continue
        
        # 测试模型
        accuracy, report = test_model(model, test_loader, model_name)
        results.append((model_name, accuracy))
    
    # 输出结果
    logger.info("\n=======================================")
    logger.info("测试结果汇总:")
    for model_name, accuracy in sorted(results, key=lambda x: x[1], reverse=True):
        logger.info(f"{model_name}: {accuracy:.4f}")

if __name__ == "__main__":
    main()
