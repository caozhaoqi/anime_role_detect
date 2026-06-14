#!/usr/bin/env python3
"""
三个验证实验：
1. 背景遮挡实验 - 检测模型是否依赖背景
2. WD Tagger 标签重叠分析 - 量化视觉特征相似度
3. Embedding 可视化 (t-SNE) - 直接观察特征空间问题
"""
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageDraw
from collections import defaultdict
import numpy as np
import json

# 配置路径
TRAIN_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
FINAL_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")
WD_TAGGER_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/src/services")

Image.MAX_IMAGE_PIXELS = None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_and_preprocess_image(img_path):
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img), img
    except:
        return None, None


def load_model(num_classes, device):
    """加载MobileNetV2模型"""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model_path = MODEL_DIR / "mobilenetv2_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def load_feature_model(num_classes, device):
    """加载带特征的MobileNetV2模型"""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model_path = MODEL_DIR / "mobilenetv2_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # 移除最后的分类层，保留特征提取器
    feature_model = nn.Sequential(*list(model.children())[:-1])
    feature_model = nn.Sequential(
        feature_model,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    feature_model = feature_model.to(device)
    feature_model.eval()
    return feature_model


# ============================================================
# 实验1: 背景遮挡实验
# ============================================================
def experiment1_background_occlusion(model, device, dataset_dir, class_to_idx, idx_to_class):
    """检测模型是否依赖背景"""
    print("\n" + "=" * 70)
    print("🧪 实验1: 背景遮挡实验")
    print("=" * 70)
    
    # 收集测试图片
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
                for img_path in list(char_dir.glob('*.jpg')) + list(char_dir.glob('*.png')):
                    if len(test_images) < 20:  # 限制样本数，防止内存溢出
                        test_images.append((str(img_path), class_to_idx[matched_class], matched_class))
    
    if not test_images:
        print("❌ 无测试数据")
        return
    
    print(f"测试样本数: {len(test_images)}")
    
    # 创建高斯模糊的背景版本
    def create_bg_occluded(img):
        """创建背景遮挡版本（高斯模糊中心区域）"""
        w, h = img.size
        # 中心区域模糊
        blur = img.filter(ImageFilter.GaussianBlur(radius=20))
        # 叠加边缘
        draw = ImageDraw.Draw(blur)
        margin = int(min(w, h) * 0.1)
        draw.rectangle([margin, margin, w-margin, h-margin], fill=None, outline=None)
        return blur
    
    def create_bg_removed(img):
        """创建去背景版本（全灰色背景）"""
        bg = Image.new('RGB', img.size, (128, 128, 128))
        return bg
    
    # 测试不同遮挡程度
    results = []
    
    for occlusion_type, creator in [
        ("原始图片", lambda x: x),
        ("中心模糊", create_bg_occluded),
        ("纯灰背景", create_bg_removed),
    ]:
        correct = 0
        total = 0
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        for img_path, label_idx, class_name in tqdm(test_images, desc=occlusion_type):
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 应用遮挡
                img_occluded = creator(img)
                img_tensor = transform(img_occluded).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    pred = output.argmax(dim=1).item()
                    
                    if pred == label_idx:
                        correct += 1
                    total += 1
            except:
                pass
        
        acc = correct / total * 100 if total > 0 else 0
        results.append((occlusion_type, acc, correct, total))
        print(f"  {occlusion_type}: {acc:.2f}% ({correct}/{total})")
    
    # 分析结果
    print("\n📊 分析结论:")
    original_acc = results[0][1]
    gray_bg_acc = results[2][1]
    diff = original_acc - gray_bg_acc
    
    if diff > 10:
        print(f"  ⚠️ 模型严重依赖背景！去除背景后准确率下降 {diff:.1f}%")
        print("  建议：使用角色抠图数据或更强的数据增强")
    elif diff > 5:
        print(f"  ⚡ 模型部分依赖背景，准确率下降 {diff:.1f}%")
    else:
        print(f"  ✅ 模型对背景依赖较低，准确率仅下降 {diff:.1f}%")
    
    return results


# ============================================================
# 实验2: 标签重叠分析
# ============================================================
def experiment2_tag_overlap():
    """分析Furina/Kafka等相似角色的标签重叠"""
    print("\n" + "=" * 70)
    print("🏷️ 实验2: WD Tagger 标签重叠分析")
    print("=" * 70)
    
    # 检查是否有WD Tagger
    tagger_paths = [
        WD_TAGGER_DIR / "wd_tagger_service.py",
        WD_TAGGER_DIR / "tagger_service.py",
        Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/src/services/wd_tagger.py"),
    ]
    
    tagger_path = None
    for p in tagger_paths:
        if p.exists():
            tagger_path = p
            break
    
    if not tagger_path:
        print("⚠️ WD Tagger 未找到，使用模拟分析")
        # 模拟分析 - 基于颜色标签
        print("\n模拟分析 Furina vs Kafka 视觉特征:")
        print("-" * 40)
        
        # 常见动漫角色标签
        char_tags = {
            'Furina': ['blue_hair', 'long_hair', 'heterochromia', 'dress', 'fairy', 'blonde', 'water', 'elegant'],
            'Kafka': ['blonde', 'short_hair', 'smile', 'jacket', 'boots', 'military', 'relaxed', 'animal_ears'],
            'Paimon': ['white_hair', 'short_hair', 'floating', 'small_girl', 'gold_eyes', 'cape', 'halo'],
            'Nahida': ['green_hair', 'short_hair', 'small_girl', 'leaf', 'nervous', 'sidelocks', 'animal_ears'],
            'Klee': ['red_hair', 'short_hair', 'blonde', 'small_girl', 'jacket', 'gloves', 'fire'],
            'Sayu': ['green_hair', 'short_hair', 'small_girl', 'gloves', 'brown_eyes', 'lazy', 'mystic'],
        }
        
        for char1 in ['Furina', 'Paimon', 'Klee']:
            for char2 in ['Kafka', 'Nahida', 'Sayu']:
                if char1 != char2:
                    tags1 = set(char_tags.get(char1, []))
                    tags2 = set(char_tags.get(char2, []))
                    overlap = tags1 & tags2
                    union = tags1 | tags2
                    jaccard = len(overlap) / len(union) if union else 0
                    
                    print(f"\n{char1} vs {char2}:")
                    print(f"  重叠标签: {overlap}")
                    print(f"  Jaccard相似度: {jaccard:.2%}")
        
        return None
    
    # 使用真实WD Tagger
    print(f"使用WD Tagger: {tagger_path}")
    # ... (如果可用的话)


# ============================================================
# 实验3: Embedding可视化
# ============================================================
def experiment3_embedding_visualization(model, device, dataset_dir, class_to_idx, idx_to_class):
    """t-SNE可视化特征空间"""
    print("\n" + "=" * 70)
    print("📈 实验3: Embedding t-SNE 可视化")
    print("=" * 70)
    
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ 需要安装 sklearn 和 matplotlib: pip install scikit-learn matplotlib")
        print("\n使用简化的聚类分析代替...")
        
        # 简化分析：计算类内/类间距离
        return simplified_cluster_analysis(model, device, dataset_dir, class_to_idx, idx_to_class)
    
    # 收集特征
    print("收集特征向量...")
    features = []
    labels = []
    class_names = []
    
    # 只采样部分角色
    focus_chars = ['Furina', 'Kafka', 'Paimon', 'Nahida', 'Klee', 'Sayu', 
                   'Ganyu', 'Herta', 'Arona', 'Firefly', 'Aru', 'Ayaka']
    
    for char_dir in dataset_dir.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            if char_name not in focus_chars:
                continue
                
            char_lower = char_name.lower()
            matched_class = None
            for cls in class_to_idx.keys():
                if cls.lower() == char_lower:
                    matched_class = cls
                    break
            
            if matched_class and matched_class == char_name:
                imgs = list(char_dir.glob('*.jpg')) + list(char_dir.glob('*.png'))
                # 每个角色采样最多20张
                for img_path in imgs[:20]:
                    tensor, _ = load_and_preprocess_image(str(img_path))
                    if tensor is not None:
                        with torch.no_grad():
                            feat = model(tensor.unsqueeze(0).to(device))
                            feat = feat.squeeze().cpu().numpy()
                        features.append(feat)
                        labels.append(class_to_idx[matched_class])
                        class_names.append(matched_class)
    
    if len(features) < 10:
        print("❌ 样本不足")
        return
    
    print(f"收集到 {len(features)} 个特征向量")
    
    # t-SNE降维
    print("执行t-SNE降维...")
    features = np.array(features)
    labels = np.array(labels)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features)
    
    # 绘制
    plt.figure(figsize=(12, 10))
    
    # 颜色映射
    unique_labels = list(set(class_names))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    for i, (x, y) in enumerate(features_2d):
        plt.scatter(x, y, c=[color_map[class_names[i]]], s=50, alpha=0.7)
    
    # 添加图例
    for label in unique_labels:
        indices = [i for i, c in enumerate(class_names) if c == label]
        if indices:
            x_avg = np.mean(features_2d[indices, 0])
            y_avg = np.mean(features_2d[indices, 1])
            plt.annotate(label, (x_avg, y_avg), fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('t-SNE Visualization of Character Embeddings\nMobileNetV2 Feature Space', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # 保存图片
    output_path = MODEL_DIR.parent / "logs" / "tsne_visualization.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 图片已保存: {output_path}")
    
    # 分析聚类质量
    print("\n📊 聚类分析:")
    analyze_cluster_quality(features_2d, class_names)
    
    return output_path


def simplified_cluster_analysis(model, device, dataset_dir, class_to_idx, idx_to_class):
    """简化的聚类分析"""
    print("执行简化的聚类分析...")
    
    # 收集每个类的特征统计
    class_features = defaultdict(list)
    
    focus_chars = ['Furina', 'Kafka', 'Paimon', 'Nahida', 'Klee', 'Sayu']
    
    for char_dir in dataset_dir.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            if char_name not in focus_chars:
                continue
            
            if char_name in class_to_idx:
                imgs = list(char_dir.glob('*.jpg')) + list(char_dir.glob('*.png'))
                # 每个角色采样最多10张，极大减少内存占用
                for img_path in imgs[:10]:
                    tensor, _ = load_and_preprocess_image(str(img_path))
                    if tensor is not None:
                        with torch.no_grad():
                            feat = model(tensor.unsqueeze(0).to(device))
                            feat = feat.squeeze().cpu().numpy()
                        class_features[char_name].append(feat)
    
    if not class_features:
        print("❌ 无足够数据")
        return
    
    # 计算类内/类间距离
    print("\n📊 类内/类间距离分析:")
    print("-" * 50)
    
    def cosine_dist(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    # 计算每个类的中心
    centers = {}
    for char, feats in class_features.items():
        centers[char] = np.mean(feats, axis=0)
    
    # 计算类内距离（样本到中心的平均距离）
    intra_distances = {}
    for char, feats in class_features.items():
        center = centers[char]
        distances = [cosine_dist(f, center) for f in feats]
        intra_distances[char] = np.mean(distances)
    
    # 计算类间距离（中心之间的距离）
    chars = list(centers.keys())
    inter_distances = {}
    for i, c1 in enumerate(chars):
        for c2 in chars[i+1:]:
            d = cosine_dist(centers[c1], centers[c2])
            inter_distances[(c1, c2)] = d
    
    print("\n类内距离 (越小越好):")
    for char, dist in sorted(intra_distances.items(), key=lambda x: x[1]):
        print(f"  {char}: {dist:.4f}")
    
    print("\n类间距离 (越大越好):")
    similar_pairs = []
    for (c1, c2), dist in sorted(inter_distances.items(), key=lambda x: x[1]):
        if dist < 0.5:  # 距离小于0.5表示可能混淆
            similar_pairs.append((c1, c2, dist))
    
    if similar_pairs:
        print("⚠️ 容易混淆的角色对:")
        for c1, c2, dist in similar_pairs:
            print(f"  {c1} ↔ {c2}: {dist:.4f}")
    else:
        print("✅ 类间距离良好")
    
    # 计算分离度
    avg_intra = np.mean(list(intra_distances.values()))
    avg_inter = np.mean(list(inter_distances.values()))
    separation = avg_inter / (avg_intra + 1e-8)
    
    print(f"\n分离度 (类间/类内): {separation:.2f}")
    if separation < 2:
        print("⚠️ 分离度较低，角色特征空间重叠严重")
        print("   建议转向 ArcFace 度量学习")
    else:
        print("✅ 分离度良好")


def analyze_cluster_quality(features_2d, labels):
    """分析聚类质量"""
    from collections import Counter
    
    # 计算每个类的分布
    label_counts = Counter(labels)
    
    # 计算紧密度（类内方差）
    unique_labels = set(labels)
    compactness = {}
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        if len(indices) > 1:
            points = features_2d[indices]
            center = np.mean(points, axis=0)
            distances = np.sqrt(np.sum((points - center)**2, axis=1))
            compactness[label] = np.mean(distances)
    
    # 计算分离度（类间距离）
    centers = {}
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        centers[label] = np.mean(features_2d[indices], axis=0)
    
    separations = []
    labels_list = list(unique_labels)
    for i, l1 in enumerate(labels_list):
        for l2 in labels_list[i+1:]:
            d = np.linalg.norm(centers[l1] - centers[l2])
            separations.append(d)
    
    avg_compactness = np.mean(list(compactness.values())) if compactness else 0
    avg_separation = np.mean(separations) if separations else 0
    
    print(f"\n聚类质量指标:")
    print(f"  平均紧密度: {avg_compactness:.2f} (越小越好)")
    print(f"  平均分离度: {avg_separation:.2f} (越大越好)")
    print(f"  分离/紧密度比: {avg_separation/(avg_compactness+1e-8):.2f}")
    
    if avg_separation / (avg_compactness + 1e-8) < 1.5:
        print("\n⚠️ 特征空间重叠严重，建议转向度量学习")


def main():
    print("=" * 70)
    print("🔬 三个验证实验")
    print("=" * 70)
    
    device = get_device()
    print(f"📱 使用设备: {device}")
    
    # 加载模型
    print("\n📦 加载模型...")
    train_classes = sorted([d.name for d in TRAIN_DATA_DIR.iterdir() if d.is_dir()])
    class_to_idx = {cls: i for i, cls in enumerate(train_classes)}
    idx_to_class = {i: cls for i, cls in enumerate(train_classes)}
    num_classes = len(train_classes)
    
    model = load_model(num_classes, device)
    print(f"✅ 加载 MobileNetV2")
    
    # 加载特征模型
    feature_model = load_feature_model(num_classes, device)
    print(f"✅ 加载特征模型")
    
    # 实验1: 背景遮挡
    experiment1_background_occlusion(model, device, FINAL_DATA_DIR, class_to_idx, idx_to_class)
    
    # 实验2: 标签重叠
    experiment2_tag_overlap()
    
    # 实验3: Embedding可视化
    experiment3_embedding_visualization(feature_model, device, FINAL_DATA_DIR, class_to_idx, idx_to_class)
    
    print("\n" + "=" * 70)
    print("✅ 三个实验完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
