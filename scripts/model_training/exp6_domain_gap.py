#!/usr/bin/env python3
"""
实验6: t-SNE/UMAP可视化
验证训练集与final_dataset之间的Domain Gap
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
import gc

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️ 需要安装 sklearn 和 matplotlib")

TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
FINAL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_feature_extractor(device):
    """加载MobileNetV2作为特征提取器"""
    print("📦 加载MobileNetV2...")
    model = models.mobilenet_v2(weights=None)
    num_classes = len([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # 加载训练好的权重
    model_path = MODEL_DIR / "mobilenetv2_best.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("  ✅ 已加载训练权重")
    else:
        print("  ⚠️ 未找到权重，使用随机初始化")
    
    # 移除分类头，保留特征提取部分
    feature_extractor = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    ).to(device).eval()
    
    return feature_extractor


def extract_features(feature_extractor, device, dataset_dir, class_to_idx, sample_per_class=5):
    """提取特征"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    features = []
    labels = []
    sources = []  # 0=training, 1=final
    
    # 使用所有角色，但只关注几个重点角色
    focus_chars = ['Furina', 'Kafka', 'Paimon', 'Nahida', 'Klee', 'Sayu']
    
    for char_dir in dataset_dir.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            
            # 跳过不在训练集中的角色
            if char_name not in class_to_idx:
                continue
            
            # 如果是重点角色，多取一些样本
            samples = sample_per_class * 2 if char_name in focus_chars else sample_per_class
            
            imgs = list(char_dir.glob('*.jpg'))[:samples] + list(char_dir.glob('*.png'))[:samples]
            
            for img_path in imgs[:samples]:
                try:
                    img = Image.open(img_path).convert('RGB')
                    tensor = transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        feat = feature_extractor(tensor).squeeze().cpu().numpy()
                    
                    features.append(feat)
                    labels.append(class_to_idx[char_name])
                    sources.append(0 if dataset_dir == TRAIN_DIR else 1)
                    
                    del img, tensor, feat
                    gc.collect()
                except Exception as e:
                    continue
    
    print(f"  从 {dataset_dir.name} 提取了 {len(features)} 个特征")
    return np.array(features), np.array(labels), np.array(sources)


def plot_tsne(features, labels, sources, class_to_idx, title="t-SNE Visualization"):
    """绘制t-SNE图"""
    if not PLOT_AVAILABLE:
        print("⚠️ matplotlib/sklearn未安装，跳过绘图")
        return None
    
    print("\n📊 执行t-SNE降维...")
    # 先用PCA降维到50维加速
    pca = PCA(n_components=min(50, features.shape[1], len(features)-1))
    features_pca = pca.fit_transform(features)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features_pca)
    
    plt.figure(figsize=(12, 10))
    
    # 颜色映射
    unique_classes = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
    
    # 创建反向映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 绘制训练集（圆形）和测试集（方形）
    for i, label in enumerate(unique_classes):
        mask = (labels == label) & (sources == 0)
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=[colors[i]], marker='o', s=80, alpha=0.7,
                    label=f"{idx_to_class[label]} (train)")
        
        mask = (labels == label) & (sources == 1)
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=[colors[i]], marker='s', s=80, alpha=0.7,
                    label=f"{idx_to_class[label]} (final)")
    
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_path = MODEL_DIR.parent / "logs" / "tsne_domain_gap.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图片已保存: {output_path}")
    
    return features_2d


def analyze_domain_gap(features_train, features_final, labels_train, labels_final, class_to_idx):
    """分析域差距"""
    print("\n" + "=" * 60)
    print("📊 Domain Gap 分析")
    print("=" * 60)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 计算同一角色在两个数据集之间的距离
    focus_chars = ['Furina', 'Kafka', 'Paimon', 'Nahida', 'Klee', 'Sayu']
    
    for char_name in focus_chars:
        if char_name not in class_to_idx:
            continue
        
        char_idx = class_to_idx[char_name]
        
        # 获取该角色在两个数据集的特征
        train_mask = labels_train == char_idx
        final_mask = labels_final == char_idx
        
        if train_mask.sum() == 0 or final_mask.sum() == 0:
            continue
        
        train_feats = features_train[train_mask]
        final_feats = features_final[final_mask]
        
        # 计算距离
        train_center = np.mean(train_feats, axis=0)
        final_center = np.mean(final_feats, axis=0)
        
        inter_dist = np.linalg.norm(train_center - final_center)
        
        # 计算类内距离
        train_intra = np.mean([np.linalg.norm(f - train_center) for f in train_feats])
        final_intra = np.mean([np.linalg.norm(f - final_center) for f in final_feats])
        
        print(f"\n{char_name}:")
        print(f"  训练集样本数: {len(train_feats)}")
        print(f"  final样本数: {len(final_feats)}")
        print(f"  类内距离(train): {train_intra:.4f}")
        print(f"  类内距离(final): {final_intra:.4f}")
        print(f"  跨数据集距离: {inter_dist:.4f}")
        
        # 判断是否存在域差距
        if inter_dist > max(train_intra, final_intra) * 2:
            print(f"  ⚠️ 存在明显Domain Gap")
        else:
            print(f"  ✅ 域差距较小")


def main():
    print("=" * 60)
    print("🔬 实验6: Domain Gap 可视化")
    print("=" * 60)
    
    device = get_device()
    print(f"📱 设备: {device}")
    
    # 获取类别
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"📊 类别数: {len(classes)}")
    
    # 加载特征提取器
    feature_extractor = load_feature_extractor(device)
    
    # 提取训练集特征
    print("\n📥 提取训练集特征...")
    train_features, train_labels, train_sources = extract_features(
        feature_extractor, device, TRAIN_DIR, class_to_idx, sample_per_class=5
    )
    
    # 提取final_dataset特征
    print("\n📥 提取final_dataset特征...")
    final_features, final_labels, final_sources = extract_features(
        feature_extractor, device, FINAL_DIR, class_to_idx, sample_per_class=5
    )
    
    # 检查是否有final数据
    if len(final_features) == 0:
        print("❌ final_dataset中没有匹配的角色")
        return
    
    # 合并特征用于可视化
    all_features = np.vstack([train_features, final_features])
    all_labels = np.concatenate([train_labels, final_labels])
    all_sources = np.concatenate([train_sources, final_sources])
    
    print(f"\n📊 总特征数: {len(all_features)}")
    
    # 绘制t-SNE
    plot_tsne(all_features, all_labels, all_sources, class_to_idx, 
              title="MobileNetV2 Embedding t-SNE\nCircle=Train, Square=Final")
    
    # 分析域差距
    analyze_domain_gap(train_features, final_features, train_labels, final_labels, class_to_idx)
    
    print("\n" + "=" * 60)
    print("✅ 实验6完成")
    print("=" * 60)


if __name__ == "__main__":
    main()