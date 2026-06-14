#!/usr/bin/env python3
"""实验2: 标签重叠分析 + 实验3: 简化聚类分析"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import gc

TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
FINAL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main():
    print("=" * 60)
    print("🏷️ 实验2: 标签重叠分析")
    print("=" * 60)
    
    # 模拟分析 - 基于已知视觉特征
    char_tags = {
        'Furina': ['blue_hair', 'long_hair', 'heterochromia', 'dress', 'blonde', 'elegant', 'bikini'],
        'Kafka': ['blonde', 'short_hair', 'smile', 'jacket', 'boots', 'military', 'relaxed'],
        'Paimon': ['white_hair', 'short_hair', 'floating', 'small_girl', 'gold_eyes', 'cape'],
        'Nahida': ['green_hair', 'short_hair', 'small_girl', 'leaf', 'animal_ears', 'bikini'],
        'Klee': ['red_hair', 'short_hair', 'blonde', 'small_girl', 'jacket', 'gloves', 'fire'],
        'Sayu': ['green_hair', 'short_hair', 'small_girl', 'gloves', 'brown_eyes', 'lazy'],
    }
    
    print("\n角色视觉特征重叠分析:\n")
    for c1 in ['Furina', 'Paimon', 'Klee']:
        for c2 in ['Kafka', 'Nahida', 'Sayu']:
            t1 = set(char_tags.get(c1, []))
            t2 = set(char_tags.get(c2, []))
            overlap = t1 & t2
            union = t1 | t2
            jaccard = len(overlap) / len(union) if union else 0
            print(f"  {c1:>8} vs {c2:<8}: 重叠={list(overlap)[:3]} Jaccard={jaccard:.1%}")
    
    print("\n" + "=" * 60)
    print("📈 实验3: 简化聚类分析")
    print("=" * 60)
    
    device = get_device()
    
    # 获取类别
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    
    # 加载特征模型
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_DIR / "mobilenetv2_best.pth", map_location=device, weights_only=True))
    
    # 去掉分类层
    feature_model = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    ).to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 收集特征（每个角色2张，极简）
    focus_chars = ['Furina', 'Kafka', 'Paimon', 'Nahida', 'Klee', 'Sayu']
    class_features = defaultdict(list)
    
    print("\n收集特征...")
    for char_dir in TRAIN_DIR.iterdir():  # 改用 training_dataset
        if char_dir.is_dir() and char_dir.name in focus_chars:
            imgs = list(char_dir.glob('*.jpg'))[:2] + list(char_dir.glob('*.png'))[:2]
            for img_path in imgs:
                try:
                    img = Image.open(img_path).convert('RGB')
                    tensor = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = feature_model(tensor).squeeze().cpu().numpy()
                    class_features[char_dir.name].append(feat)
                    del img, tensor, feat
                    gc.collect()
                except:
                    pass
    
    if not class_features:
        print("❌ 无数据")
        return
    
    # 计算聚类指标
    def cosine_dist(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    centers = {c: np.mean(feats, axis=0) for c, feats in class_features.items()}
    
    # 类内距离
    intra = {}
    for c, feats in class_features.items():
        dists = [cosine_dist(f, centers[c]) for f in feats]
        intra[c] = np.mean(dists)
    
    # 类间距离
    inter = {}
    chars = list(centers.keys())
    for i, c1 in enumerate(chars):
        for c2 in chars[i+1:]:
            d = cosine_dist(centers[c1], centers[c2])
            inter[(c1, c2)] = d
    
    print("\n📊 类内距离 (越小越紧凑):")
    for c, d in sorted(intra.items(), key=lambda x: x[1]):
        print(f"  {c}: {d:.4f}")
    
    print("\n📊 类间距离 (越小越易混淆):")
    similar = [(c1, c2, d) for (c1, c2), d in inter.items() if d < 0.3]
    similar.sort(key=lambda x: x[2])
    for c1, c2, d in similar[:5]:
        print(f"  {c1} ↔ {c2}: {d:.4f} ⚠️")
    
    avg_intra = np.mean(list(intra.values()))
    avg_inter = np.mean(list(inter.values()))
    print(f"\n📊 分离度 (类间/类内): {avg_inter/(avg_intra+1e-8):.2f}")
    
    if avg_inter / (avg_intra + 1e-8) < 2:
        print("⚠️ 特征空间重叠严重，建议转向 ArcFace 度量学习")

if __name__ == "__main__":
    main()