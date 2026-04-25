import os
import json
import logging
import argparse
import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.model_training.incremental_train import (
    SimpleImageDataset, TransformSubset,
    get_available_device
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('confusion_matrix_analysis')

IMAGE_SIZE = 224

def load_model_and_predict(model_path, data_dir, device):
    """加载模型并在数据集上进行预测"""
    logger.info(f"加载模型: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state = checkpoint['model_state_dict']
    model_type = checkpoint.get('model_type', 'efficientnet_b0')
    class_to_idx = checkpoint['class_to_idx']

    logger.info(f"模型类型: {model_type}")
    logger.info(f"类别数量: {len(class_to_idx)}")

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if model_type == 'efficientnet_b0':
        from torchvision import models
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_to_idx))
    elif model_type == 'mobilenet_v2':
        from torchvision import models
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_to_idx))
    elif model_type == 'resnet18':
        from torchvision import models
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
    else:
        from torchvision import models
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_to_idx))

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = SimpleImageDataset(data_dir, transform=val_transform)

    val_dataset = torch.utils.data.Subset(full_dataset, range(len(full_dataset)))

    dataset = TransformSubset(val_dataset, transform=None)

    all_preds = []
    all_labels = []

    logger.info(f"在 {len(dataset)} 张图片上进行预测...")

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="预测"):
            img_path, label = dataset.subset.dataset.samples[dataset.subset.indices[idx]]
            try:
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                image = val_transform(image).unsqueeze(0).to(device)

                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

                all_preds.append(predicted.item())
                all_labels.append(label)
            except Exception as e:
                logger.error(f"预测失败 {img_path}: {e}")
                continue

    return np.array(all_preds), np.array(all_labels), idx_to_class

def analyze_confusion_matrix(y_true, y_pred, idx_to_class, top_n=10):
    """分析混淆矩阵，找出最容易混淆的类别对"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=[idx_to_class[i] for i in range(len(idx_to_class))],
                yticklabels=[idx_to_class[i] for i in range(len(idx_to_class))])
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    logger.info("混淆矩阵已保存: confusion_matrix.png")
    plt.close()

    confusion_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true_class': idx_to_class[i],
                    'pred_class': idx_to_class[j],
                    'count': int(cm[i, j])
                })

    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

    most_confused = confusion_pairs[:top_n]

    logger.info("\n" + "=" * 60)
    logger.info(f"最容易混淆的 {top_n} 个类别对:")
    logger.info("=" * 60)
    for i, pair in enumerate(most_confused, 1):
        logger.info(f"{i:2d}. 真实: {pair['true_class']:20s} -> 预测: {pair['pred_class']:20s} (错误数: {pair['count']})")

    report = classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(len(idx_to_class))])
    logger.info("\n分类报告:")
    logger.info(report)

    return most_confused, cm

def main():
    parser = argparse.ArgumentParser(description='混淆矩阵分析 - 找出模型最容易混淆的类别对')
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--data_dir', required=True, help='数据目录')
    parser.add_argument('--top_n', type=int, default=10, help='显示最容易混淆的类别对数量')
    parser.add_argument('--output', default='confusion_analysis.json', help='输出文件路径')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("开始混淆矩阵分析")
    logger.info("=" * 60)

    device = get_available_device()
    logger.info(f"使用设备: {device}")

    y_pred, y_true, idx_to_class = load_model_and_predict(args.model_path, args.data_dir, device)

    most_confused, cm = analyze_confusion_matrix(y_true, y_pred, idx_to_class, args.top_n)

    results = {
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'total_samples': len(y_true),
        'num_classes': len(idx_to_class),
        'accuracy': (y_pred == y_true).mean() * 100,
        'most_confused_pairs': most_confused,
        'confusion_matrix_shape': cm.shape
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"分析结果已保存: {args.output}")

    logger.info("\n" + "=" * 60)
    logger.info("分析完成")
    logger.info("=" * 60)
    logger.info(f"总样本数: {results['total_samples']}")
    logger.info(f"类别数: {results['num_classes']}")
    logger.info(f"准确率: {results['accuracy']:.2f}%")

if __name__ == '__main__':
    main()
