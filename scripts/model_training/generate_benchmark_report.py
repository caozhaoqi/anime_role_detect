#!/usr/bin/env python3
"""
生成完整的基准测试报告
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile
import warnings
from datetime import datetime

# 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')

# 配置
TRAIN_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
FINAL_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")
REPORT_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/logs")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_and_preprocess_image(img_path):
    """加载并预处理图片"""
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img)
    except:
        return None


def test_model_on_dataset(model, device, dataset_dir, class_to_idx, idx_to_class, model_name, dataset_name):
    """在指定数据集上测试模型"""
    print(f"\n🔍 测试 {model_name} 在 {dataset_name}...")
    
    test_images = []
    for char_dir in dataset_dir.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            char_lower = char_name.lower()
            
            matched_class = None
            for cls in class_to_idx.keys():
                if cls.lower() == char_lower:
                    matched_class = cls
                    break
            
            if matched_class:
                label_idx = class_to_idx[matched_class]
                for img_path in list(char_dir.glob('*.jpg')) + list(char_dir.glob('*.png')):
                    test_images.append((str(img_path), label_idx, matched_class))
    
    if not test_images:
        return None
    
    print(f"  测试图片数: {len(test_images)}")
    
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    batch_size = 32
    for i in tqdm(range(0, len(test_images), batch_size), desc=f"测试{dataset_name}"):
        batch = test_images[i:i+batch_size]
        batch_tensors = []
        batch_labels = []
        
        for img_path, label_idx, class_name in batch:
            tensor = load_and_preprocess_image(img_path)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_labels.append(label_idx)
                if class_name not in class_total:
                    class_correct[class_name] = 0
                    class_total[class_name] = 0
                class_total[class_name] += 1
        
        if not batch_tensors:
            continue
            
        batch_tensor = torch.stack(batch_tensors).to(device)
        batch_labels = torch.tensor(batch_labels).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, preds = torch.max(outputs, 1)
            
            for pred, label in zip(preds, batch_labels):
                total += 1
                true_class = idx_to_class[label.item()]
                if pred == label:
                    correct += 1
                    class_correct[true_class] += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    
    return {
        'dataset_name': dataset_name,
        'total_images': total,
        'correct': correct,
        'accuracy': accuracy,
        'class_correct': class_correct,
        'class_total': class_total
    }


def main():
    print("=" * 70)
    print("📊 基准测试报告生成")
    print("=" * 70)
    
    device = get_device()
    print(f"📱 使用设备: {device}")
    
    # 获取训练集类别
    print("\n📂 加载训练集类别...")
    train_classes = sorted([d.name for d in TRAIN_DATA_DIR.iterdir() if d.is_dir()])
    class_to_idx = {cls: i for i, cls in enumerate(train_classes)}
    idx_to_class = {i: cls for i, cls in enumerate(train_classes)}
    num_classes = len(train_classes)
    print(f"  训练集类别数: {num_classes}")
    
    # 加载模型
    print("\n📦 加载模型...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model_path = MODEL_DIR / "mobilenetv2_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"✅ 加载模型: {model_path}")
    
    # 测试训练数据集
    train_result = test_model_on_dataset(
        model, device, TRAIN_DATA_DIR, class_to_idx, idx_to_class,
        "MobileNetV2", "training_dataset"
    )
    
    # 测试 final_dataset
    final_result = test_model_on_dataset(
        model, device, FINAL_DATA_DIR, class_to_idx, idx_to_class,
        "MobileNetV2", "final_dataset"
    )
    
    # 生成报告
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = []
    report.append("=" * 70)
    report.append("📊 基准测试报告")
    report.append("=" * 70)
    report.append(f"生成时间: {report_time}")
    report.append(f"模型: MobileNetV2")
    report.append(f"设备: {device}")
    report.append("")
    
    report.append("=" * 70)
    report.append("一、测试数据集概览")
    report.append("=" * 70)
    report.append(f"训练数据集角色数: {num_classes}")
    report.append(f"训练数据集路径: {TRAIN_DATA_DIR}")
    
    if final_result:
        final_chars = len(final_result['class_total'])
        report.append(f"final_dataset角色数: {final_chars}")
        report.append(f"final_dataset路径: {FINAL_DATA_DIR}")
    
    report.append("")
    
    report.append("=" * 70)
    report.append("二、整体准确率对比")
    report.append("=" * 70)
    
    if train_result:
        report.append(f"训练数据集测试:")
        report.append(f"  图片数: {train_result['total_images']}")
        report.append(f"  正确数: {train_result['correct']}")
        report.append(f"  准确率: {train_result['accuracy']:.2f}%")
    
    report.append("")
    
    if final_result:
        report.append(f"final_dataset测试:")
        report.append(f"  图片数: {final_result['total_images']}")
        report.append(f"  正确数: {final_result['correct']}")
        report.append(f"  准确率: {final_result['accuracy']:.2f}%")
    
    report.append("")
    
    report.append("=" * 70)
    report.append("三、各类别准确率分析")
    report.append("=" * 70)
    
    if train_result:
        train_class_acc = [(k, train_result['class_correct'][k] / train_result['class_total'][k] * 100) 
                          for k in train_result['class_total'] if train_result['class_total'][k] > 0]
        train_class_acc.sort(key=lambda x: -x[1])
        
        report.append("\n训练数据集 - TOP 10 最佳:")
        for cls, acc in train_class_acc[:10]:
            report.append(f"  {cls}: {acc:.1f}% ({train_result['class_correct'][cls]}/{train_result['class_total'][cls]})")
        
        report.append("\n训练数据集 - TOP 10 最差:")
        for cls, acc in train_class_acc[-10:]:
            report.append(f"  {cls}: {acc:.1f}% ({train_result['class_correct'][cls]}/{train_result['class_total'][cls]})")
    
    report.append("")
    
    if final_result:
        final_class_acc = [(k, final_result['class_correct'][k] / final_result['class_total'][k] * 100) 
                          for k in final_result['class_total'] if final_result['class_total'][k] > 0]
        final_class_acc.sort(key=lambda x: -x[1])
        
        report.append("\nfinal_dataset - TOP 10 最佳:")
        for cls, acc in final_class_acc[:10]:
            report.append(f"  {cls}: {acc:.1f}% ({final_result['class_correct'][cls]}/{final_result['class_total'][cls]})")
        
        report.append("\nfinal_dataset - TOP 10 最差:")
        for cls, acc in final_class_acc[-10:]:
            report.append(f"  {cls}: {acc:.1f}% ({final_result['class_correct'][cls]}/{final_result['class_total'][cls]})")
    
    report.append("")
    
    report.append("=" * 70)
    report.append("四、结论与建议")
    report.append("=" * 70)
    
    if train_result and final_result:
        diff = train_result['accuracy'] - final_result['accuracy']
        report.append(f"准确率差异: {diff:.2f}%")
        report.append("")
        
        if diff > 20:
            report.append("⚠️ 模型在新数据上表现显著下降，建议:")
            report.append("  1. 扩充训练数据集，增加更多样化的图片")
            report.append("  2. 使用数据增强技术提高泛化能力")
            report.append("  3. 调整模型架构或训练参数")
        elif diff > 10:
            report.append("📝 模型在新数据上表现有一定下降，建议:")
            report.append("  1. 继续采集更多训练数据")
            report.append("  2. 检查final_dataset数据质量")
        else:
            report.append("✅ 模型在新数据上表现稳定，泛化能力良好")
    
    report.append("")
    report.append("=" * 70)
    report.append("报告结束")
    report.append("=" * 70)
    
    # 输出报告
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # 保存报告
    report_file = REPORT_DIR / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_file.write_text(report_text)
    print(f"\n📄 报告已保存: {report_file}")


if __name__ == "__main__":
    main()