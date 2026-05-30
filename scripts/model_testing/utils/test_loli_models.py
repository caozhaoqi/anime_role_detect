import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from loguru import logger
import json

# 配置日志
logger.add("test_loli_models.log", rotation="500 MB")

# 模型路径
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")

# 测试数据路径
TEST_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded_images")

# 数据预处理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CharacterDataset(Dataset):
    """自定义数据集"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # 加载数据
        class_names = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 加载图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"加载失败 {img_path}: {e}")
            # 创建空白图像
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label


def get_model(model_type, num_classes):
    """根据类型获取模型"""
    if model_type == "mobilenet_v2":
        from torchvision.models import mobilenet_v2

        model = mobilenet_v2(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes),
        )
    elif model_type == "efficientnet_b0":
        from torchvision.models import efficientnet_b0

        model = efficientnet_b0(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes),
        )
    elif model_type == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    return model


def load_model(model_path, class_to_idx):
    """加载模型"""
    try:
        # 加载完整模型
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        model.eval()

        # 提取模型类型
        model_type = "mobilenet_v2"
        if "efficientnet" in str(model_path):
            model_type = "efficientnet_b0"
        elif "resnet" in str(model_path):
            model_type = "resnet18"

        return model, model_type
    except Exception as e:
        logger.error(f"加载模型失败 {model_path}: {e}")
        return None, None


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
    all_pred_labels = []
    all_true_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            # 预测
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # 保存结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 保存预测和真实标签的字符串形式
            for pred_idx, true_idx in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if pred_idx in idx_to_class:
                    pred_label = idx_to_class[pred_idx]
                else:
                    pred_label = "未知"
                if true_idx in idx_to_class:
                    true_label = idx_to_class[true_idx]
                else:
                    true_label = "未知"
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

    return accuracy


def main():
    """主函数"""
    # 加载类别映射
    class_to_idx_path = MODEL_DIR / "mobilenet_v2_loli8" / "class_to_idx.json"
    if not class_to_idx_path.exists():
        logger.error(f"类别映射文件不存在: {class_to_idx_path}")
        return

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    # 创建测试数据集
    test_dataset = CharacterDataset(TEST_DATA_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    logger.info(f"测试数据集大小: {len(test_dataset)}")
    logger.info(f"测试数据类别: {list(class_to_idx.keys())}")

    # 测试模型列表
    models_to_test = [
        ("mobilenet_v2", MODEL_DIR / "mobilenet_v2_loli8" / "model_full.pth"),
        ("efficientnet_b0", MODEL_DIR / "efficientnet_b0_loli8" / "model_full.pth"),
        ("resnet18", MODEL_DIR / "resnet18_loli8" / "model_full.pth"),
    ]

    results = []

    for model_name, model_path in models_to_test:
        if not model_path.exists():
            logger.warning(f"模型文件不存在: {model_path}")
            continue

        logger.info(f"\n=======================================")
        logger.info(f"测试模型: {model_name}")
        logger.info(f"模型路径: {model_path}")

        # 加载模型
        model, loaded_model_type = load_model(model_path, class_to_idx)
        if model is None:
            continue

        # 测试模型
        accuracy = test_model(model, test_loader, model_name, class_to_idx)
        results.append((model_name, accuracy))

    # 输出结果
    logger.info("\n=======================================")
    logger.info("测试结果汇总:")
    for model_name, accuracy in sorted(results, key=lambda x: x[1], reverse=True):
        logger.info(f"{model_name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()
