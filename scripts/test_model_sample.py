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
logger.add("test_model_sample.log", rotation="500 MB")

# 模型路径
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")

# 测试数据路径
TEST_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded_images")

# 类别映射
CLASS_MAPPING = {
    "阿罗娜": "阿罗娜",
    "日奈": "日奈",
    "普拉娜": "普拉娜",
    "千夏": "千夏",
    "亚子": "亚子",
    "枫香": "枫香",
    "伊织": "伊织",
    "可莉": "可莉",
    "提宝": "提宝",
    "火花": "火花",
    "纳西妲": "纳西妲"
}

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CharacterDataset(Dataset):
    """自定义数据集，支持抽样"""
    def __init__(self, root_dir, transform=None, sample_per_class=5):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 加载数据，每个类别抽样
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                # 转换类别名
                mapped_class = CLASS_MAPPING.get(class_name, class_name)
                # 获取该类别的所有图像
                class_images = []
                for img_name in os.listdir(class_path):
                    # 跳过SVG文件
                    if img_name.endswith(".svg"):
                        continue
                    img_path = os.path.join(class_path, img_name)
                    class_images.append(img_path)
                
                # 抽样
                sampled_images = class_images[:sample_per_class]
                for img_path in sampled_images:
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
            
            # 如果模型类型未知，根据文件路径推断
            if model_type == 'mobilenet_v2':
                model_path_str = str(model_path)
                if 'efficientnet_b0' in model_path_str:
                    model_type = 'efficientnet_b0'
                elif 'efficientnet_b3' in model_path_str:
                    model_type = 'efficientnet_b3'
                elif 'resnet50' in model_path_str:
                    model_type = 'resnet50'
            
            # 检查类别数量
            num_classes = 11  # 默认类别数
            if 'class_to_idx' in model_data:
                num_classes = len(model_data['class_to_idx'])
            
            # 根据模型类型创建模型
            if 'efficientnet_b0' in model_type:
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(pretrained=False)
                # 修改分类器为我们训练时使用的结构
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.15),
                    nn.Linear(512, num_classes)
                )
            elif 'efficientnet_b3' in model_type:
                from torchvision.models import efficientnet_b3
                model = efficientnet_b3(pretrained=False)
                # 修改分类器为我们训练时使用的结构
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(model.classifier[1].in_features, 768),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(768),
                    nn.Dropout(p=0.15),
                    nn.Linear(768, num_classes)
                )
            elif 'resnet50' in model_type:
                from torchvision.models import resnet50
                model = resnet50(pretrained=False)
                # 修改分类器为我们训练时使用的结构
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(num_ftrs, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.15),
                    nn.Linear(512, num_classes)
                )
            else:  # mobilenet_v2
                from torchvision.models import mobilenet_v2
                model = mobilenet_v2(pretrained=False)
                # 修改分类器为我们训练时使用的结构
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.15),
                    nn.Linear(512, num_classes)
                )
            
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

def test_model(model, test_loader, model_name, class_to_idx):
    """测试模型"""
    logger.info(f"测试模型: {model_name}")
    
    # 从class_to_idx构建idx_to_class
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model_classes = list(class_to_idx.keys())
    logger.info(f"模型类别: {model_classes}")
    logger.info(f"类别映射: {class_to_idx}")
    
    # 预测结果
    all_preds = []
    all_labels = []
    all_image_paths = []
    all_pred_labels = []
    all_true_labels = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            # 转换标签为索引
            label_indices = []
            for label in labels:
                # 直接使用class_to_idx查找
                if label in class_to_idx:
                    label_indices.append(class_to_idx[label])
                else:
                    # 如果没有匹配的类别，添加为未知类别
                    label_indices.append(len(class_to_idx))
            
            # 预测
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 保存结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_indices)
            
            # 保存预测和真实标签的字符串形式
            for pred_idx, true_idx, true_label in zip(preds.cpu().numpy(), label_indices, labels):
                if pred_idx in idx_to_class:
                    pred_label = idx_to_class[pred_idx]
                else:
                    pred_label = "未知"
                all_pred_labels.append(pred_label)
                all_true_labels.append(true_label)
    
    # 计算准确率
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    total = len(all_labels)
    accuracy = correct / total
    logger.info(f"准确率: {accuracy:.4f} ({correct}/{total})")
    
    # 打印预测结果
    logger.info("\n预测结果示例:")
    for i in range(min(10, len(all_true_labels))):
        logger.info(f"真实: {all_true_labels[i]} | 预测: {all_pred_labels[i]}")
    
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
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    plt.savefig(f"{model_name}_sample_confusion_matrix.png")
    logger.info(f"混淆矩阵已保存为 {model_name}_sample_confusion_matrix.png")
    
    return accuracy, report

def main():
    """主函数"""
    # 创建测试数据集（抽样）
    test_dataset = CharacterDataset(TEST_DATA_DIR, transform=transform, sample_per_class=5)
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
        
        # 加载模型数据，获取class_to_idx
        try:
            model_data = torch.load(model_file, map_location=torch.device('cpu'))
            class_to_idx = model_data.get('class_to_idx', {})
            logger.info(f"从模型文件加载类别映射: {class_to_idx}")
        except Exception as e:
            logger.error(f"加载模型数据失败: {e}")
            continue
        
        # 加载模型
        model = load_model(model_file)
        if model is None:
            continue
        
        # 测试模型
        accuracy, report = test_model(model, test_loader, model_name, class_to_idx)
        results.append((model_name, accuracy))
    
    # 输出结果
    logger.info("\n=======================================")
    logger.info("测试结果汇总:")
    for model_name, accuracy in sorted(results, key=lambda x: x[1], reverse=True):
        logger.info(f"{model_name}: {accuracy:.4f}")

if __name__ == "__main__":
    main()