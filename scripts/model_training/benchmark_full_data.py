import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224
BATCH_SIZE = 32

def load_class_to_idx(model_dir):
    """加载训练时的类别映射"""
    class_map_path = Path(model_dir) / 'class_to_idx.json'
    if class_map_path.exists():
        with open(class_map_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

class RoleDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_to_idx=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx if class_to_idx else {}
        
        exclude = {'trash', 'trash_nsfw', 'trash_multi_face', '其他', '.DS_Store'}
        
        # 如果没有提供类别映射，使用数据集中的类别顺序
        if self.class_to_idx is None:
            self.class_to_idx = {}
            for idx, role_dir in enumerate(sorted(self.data_dir.iterdir())):
                if not role_dir.is_dir() or role_dir.name in exclude or role_dir.name.startswith('.'):
                    continue
                self.class_to_idx[role_dir.name] = idx
        
        # 遍历所有角色文件夹
        for role_dir in sorted(self.data_dir.iterdir()):
            if not role_dir.is_dir() or role_dir.name in exclude or role_dir.name.startswith('.'):
                continue
            
            role_name = role_dir.name
            if role_name not in self.class_to_idx:
                logger.warning(f"跳过未在类别映射中的角色: {role_name}")
                continue
            
            label = self.class_to_idx[role_name]
            
            for img_path in role_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), label))
        
        logger.info(f"加载数据集: {len(self.samples)} 张图片, {len(self.class_to_idx)} 个角色")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            logger.warning(f"加载失败 {img_path}: {e}")
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), label, img_path

def load_full_model(model_path):
    """加载完整模型文件"""
    try:
        model = torch.load(model_path, map_location=DEVICE, weights_only=False)
        logger.info(f"成功加载完整模型: {model_path}")
        return model
    except Exception as e:
        logger.error(f"加载完整模型失败: {e}")
        return None

def test_model(model, dataloader, num_classes):
    model.eval()
    correct = 0
    total = 0
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    all_predictions = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
                
                # 确保索引在范围内
                if 0 <= pred < num_classes and 0 <= label < num_classes:
                    confusion_matrix[label][pred] += 1
                all_predictions.append(pred)
                all_labels.append(label)
                all_paths.append(paths[i])
    
    accuracy = 100 * correct / total
    return accuracy, class_correct, class_total, confusion_matrix, all_predictions, all_labels, all_paths

def analyze_results(accuracy, class_correct, class_total, confusion_matrix, 
                   all_predictions, all_labels, all_paths, dataset, model_type, class_to_idx):
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_type} 基准测试结果")
    logger.info(f"{'='*60}")
    logger.info(f"总体准确率: {accuracy:.2f}%")
    
    num_classes = len(class_total)
    
    # 创建索引到类名的映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    logger.info(f"\n各类别准确率:")
    low_acc_classes = []
    high_acc_classes = []
    
    for class_idx in sorted(class_total.keys()):
        class_name = idx_to_class.get(class_idx, f"未知_{class_idx}")
        if class_total[class_idx] > 0:
            acc = 100 * class_correct[class_idx] / class_total[class_idx]
            logger.info(f"  {class_name}: {acc:.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")
            
            if acc < 30:
                low_acc_classes.append((class_name, acc, class_total[class_idx]))
            elif acc > 70:
                high_acc_classes.append((class_name, acc, class_total[class_idx]))
    
    logger.info(f"\n低准确率类别 (<30%): {len(low_acc_classes)} 个")
    for class_name, acc, count in sorted(low_acc_classes, key=lambda x: x[1]):
        logger.info(f"  {class_name}: {acc:.2f}% ({count}张)")
    
    logger.info(f"\n高准确率类别 (>70%): {len(high_acc_classes)} 个")
    for class_name, acc, count in sorted(high_acc_classes, key=lambda x: -x[1]):
        logger.info(f"  {class_name}: {acc:.2f}% ({count}张)")
    
    logger.info(f"\n混淆矩阵分析:")
    most_confused = []
    for i in range(min(confusion_matrix.shape[0], len(idx_to_class))):
        for j in range(min(confusion_matrix.shape[1], len(idx_to_class))):
            if i != j and confusion_matrix[i][j] > 0:
                class_i = idx_to_class.get(i, f"未知_{i}")
                class_j = idx_to_class.get(j, f"未知_{j}")
                confusion_ratio = confusion_matrix[i][j] / class_total[i] * 100 if class_total[i] > 0 else 0
                if confusion_ratio > 20:
                    most_confused.append((class_i, class_j, confusion_matrix[i][j], confusion_ratio))
    
    most_confused.sort(key=lambda x: -x[3])
    logger.info(f"高混淆度类别对 (>20%): {len(most_confused)} 个")
    for class_i, class_j, count, ratio in most_confused[:10]:
        logger.info(f"  {class_i} -> {class_j}: {count}次 ({ratio:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'low_acc_classes': low_acc_classes,
        'high_acc_classes': high_acc_classes,
        'most_confused': most_confused
    }

def main():
    logger.info("="*60)
    logger.info("模型基准测试 - 使用完整数据集")
    logger.info("="*60)
    
    data_dir = './data/organized_images'
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results_file = './models/benchmark_results_full_data.json'
    all_results = {}
    
    model_configs = [
        ('mobilenet_v2', './models/mobilenet_v2_loli68_20imgs'),
        ('efficientnet_b0', './models/efficientnet_b0_loli68_20imgs'),
        ('resnet18', './models/resnet18_loli68_20imgs')
    ]
    
    for model_type, model_dir in model_configs:
        logger.info(f"\n{'#'*60}")
        logger.info(f"测试模型: {model_type}")
        logger.info(f"模型目录: {model_dir}")
        logger.info(f"{'#'*60}")
        
        # 加载训练时的类别映射
        class_to_idx = load_class_to_idx(model_dir)
        if class_to_idx is None:
            logger.error(f"无法加载类别映射文件")
            continue
        
        num_classes = len(class_to_idx)
        logger.info(f"类别数: {num_classes}")
        
        # 使用训练时的类别映射创建数据集
        dataset = RoleDataset(data_dir, transform=transform, class_to_idx=class_to_idx)
        if len(dataset) == 0:
            logger.error("数据集为空！")
            continue
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # 加载模型
        model_path = Path(model_dir) / 'model_full.pth'
        model = load_full_model(str(model_path))
        if model is None:
            logger.error(f"无法加载模型")
            continue
        
        accuracy, class_correct, class_total, confusion_matrix, all_predictions, all_labels, all_paths = test_model(
            model, dataloader, num_classes
        )
        
        analysis = analyze_results(
            accuracy, class_correct, class_total, confusion_matrix,
            all_predictions, all_labels, all_paths, dataset, model_type, class_to_idx
        )
        
        all_results[model_type] = {
            'accuracy': accuracy,
            'low_acc_classes': [(name, acc, count) for name, acc, count in analysis['low_acc_classes']],
            'high_acc_classes': [(name, acc, count) for name, acc, count in analysis['high_acc_classes']],
            'most_confused': [(i, j, count, ratio) for i, j, count, ratio in analysis['most_confused'][:10]]
        }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n结果已保存到: {results_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info("基准测试总结")
    logger.info(f"{'='*60}")
    for model_type, results in all_results.items():
        logger.info(f"{model_type}: {results['accuracy']:.2f}%")
        logger.info(f"  低准确率类别: {len(results['low_acc_classes'])} 个")
        logger.info(f"  高准确率类别: {len(results['high_acc_classes'])} 个")
        logger.info(f"  高混淆度对: {len(results['most_confused'])} 个")

if __name__ == '__main__':
    main()
