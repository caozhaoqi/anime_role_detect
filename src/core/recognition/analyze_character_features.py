#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色特征分析脚本
- 角色内聚度(Intra Similarity)：衡量同一角色特征的一致性
- 角色间相似度(Inter Similarity)：衡量角色之间的区分度
- 混淆角色可视化：展示容易混淆的角色对
- t-SNE降维可视化：直观展示角色簇分布
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


from src.core.recognition.character_retriever import CharacterRetriever

# 可视化相关导入
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  matplotlib或sklearn未安装，将跳过可视化功能")


def compute_intra_similarity(feature_store):
    """
    计算角色内聚度（Intra Similarity）
    衡量同一角色内部特征向量之间的相似度
    
    Returns:
        dict: {角色名: {mean: 平均相似度, std: 标准差, min: 最小相似度}}
    """
    print("\n" + "=" * 70)
    print("📊 计算角色内聚度 (Intra Similarity)")
    print("=" * 70)
    
    results = {}
    characters = feature_store.list_characters()
    
    for char_name in characters:
        features = feature_store._get_character_features(char_name)
        if features is None or len(features) < 2:
            continue
        
        # 计算所有特征对之间的余弦相似度
        similarities = []
        n = len(features)
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(features[i], features[j])
                similarities.append(sim)
        
        if similarities:
            results[char_name] = {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'count': len(similarities),
                'feature_count': n
            }
    
    # 按平均相似度排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"{'角色名':<12} {'特征数':<6} {'平均相似度':<12} {'标准差':<8} {'最小相似度':<12}")
    print("-" * 60)
    
    for char_name, stats in sorted_results[:10]:
        print(f"{char_name:<12} {stats['feature_count']:<6} {stats['mean']:<12.4f} {stats['std']:<8.4f} {stats['min']:<12.4f}")
    
    print(f"\n📈 内聚度最高的角色（特征一致性好）:")
    for char_name, stats in sorted_results[:5]:
        print(f"   ✅ {char_name}: {stats['mean']:.4f}")
    
    print(f"\n📉 内聚度最低的角色（特征一致性差）:")
    for char_name, stats in sorted_results[-5:]:
        print(f"   ❌ {char_name}: {stats['mean']:.4f}")
    
    return results


def compute_inter_similarity(feature_store):
    """
    计算角色间相似度（Inter Similarity）
    衡量不同角色之间的相似度
    
    Returns:
        list: [(角色A, 角色B, 平均相似度), ...]
    """
    print("\n" + "=" * 70)
    print("📊 计算角色间相似度 (Inter Similarity)")
    print("=" * 70)
    
    characters = feature_store.list_characters()
    results = []
    
    for i, char_a in enumerate(characters):
        features_a = feature_store._get_character_features(char_a)
        if features_a is None:
            continue
        
        for j in range(i + 1, len(characters)):
            char_b = characters[j]
            features_b = feature_store._get_character_features(char_b)
            if features_b is None:
                continue
            
            # 计算两个角色所有特征对之间的相似度
            similarities = []
            for feat_a in features_a:
                for feat_b in features_b:
                    sim = np.dot(feat_a, feat_b)
                    similarities.append(sim)
            
            if similarities:
                avg_sim = float(np.mean(similarities))
                results.append((char_a, char_b, avg_sim))
    
    # 按相似度降序排序
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'角色A':<12} {'角色B':<12} {'平均相似度':<12}")
    print("-" * 40)
    
    for char_a, char_b, sim in results[:20]:
        print(f"{char_a:<12} {char_b:<12} {sim:<12.4f}")
    
    print(f"\n⚠️  最相似的角色对（最容易混淆）:")
    for char_a, char_b, sim in results[:10]:
        print(f"   {char_a} ↔ {char_b}: {sim:.4f}")
    
    return results


def analyze_confusion_patterns(feature_store, top_n=20):
    """
    分析混淆模式，找出最容易被误判的角色对
    """
    print("\n" + "=" * 70)
    print("🔍 分析混淆模式 (Confusion Patterns)")
    print("=" * 70)
    
    # 获取所有角色的特征
    characters = feature_store.list_characters()
    all_features = []
    all_labels = []
    
    for char_name in characters:
        features = feature_store._get_character_features(char_name)
        if features is not None:
            all_features.extend(features)
            all_labels.extend([char_name] * len(features))
    
    all_features = np.array(all_features)
    n_samples = len(all_features)
    
    # 对每个样本，找出最相似的非自身样本
    confusion_counts = defaultdict(lambda: defaultdict(int))
    
    for i in range(n_samples):
        query = all_features[i].reshape(1, -1)
        query_label = all_labels[i]
        
        # 计算与所有其他样本的相似度
        similarities = np.dot(all_features, query.T).flatten()
        
        # 找到相似度最高的top-k（排除自身）
        top_indices = np.argsort(similarities)[::-1][1:11]  # 排除自身，取top10
        
        for idx in top_indices:
            other_label = all_labels[idx]
            if other_label != query_label:
                confusion_counts[query_label][other_label] += 1
    
    # 转换为扁平列表并排序
    confusion_list = []
    for true_char, predictions in confusion_counts.items():
        for pred_char, count in predictions.items():
            confusion_list.append((true_char, pred_char, count))
    
    confusion_list.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'真实角色':<12} {'预测角色':<12} {'误判次数':<10}")
    print("-" * 40)
    
    for true_char, pred_char, count in confusion_list[:top_n]:
        print(f"{true_char:<12} {pred_char:<12} {count:<10}")
    
    # 统计每个角色被误判的总次数
    total_confusions = {}
    for true_char, predictions in confusion_counts.items():
        total_confusions[true_char] = sum(predictions.values())
    
    sorted_confusions = sorted(total_confusions.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 最容易被误判的角色:")
    for char_name, count in sorted_confusions[:10]:
        print(f"   ❌ {char_name}: {count}次被误判")
    
    return confusion_list


def visualize_character_clusters(feature_store, output_dir="visualization"):
    """
    使用 t-SNE 降维可视化角色簇分布
    
    Args:
        feature_store: FeatureStore实例
        output_dir: 输出目录
    """
    if not VISUALIZATION_AVAILABLE:
        print("⚠️  可视化功能不可用，跳过")
        return
    
    print("\n" + "=" * 70)
    print("🎨 生成 t-SNE 可视化")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有角色的特征
    characters = feature_store.list_characters()
    all_features = []
    all_labels = []
    
    for char_name in characters:
        features = feature_store._get_character_features(char_name)
        if features is not None:
            all_features.extend(features)
            all_labels.extend([char_name] * len(features))
    
    all_features = np.array(all_features)
    n_samples = len(all_features)
    
    print(f"样本数: {n_samples}, 角色数: {len(characters)}")
    
    # 先使用 PCA 降维到 50 维（加速 t-SNE）
    pca_dim = min(50, all_features.shape[1] - 1)
    pca = PCA(n_components=pca_dim)
    features_pca = pca.fit_transform(all_features)
    print(f"PCA 降维到 {pca_dim} 维")
    
    # 动态计算 perplexity（避免小样本报错）
    perplexity = max(2, min(30, n_samples - 1))
    print(f"t-SNE perplexity: {perplexity}")
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, verbose=1)
    features_tsne = tsne.fit_transform(features_pca)
    
    # 绘制散点图
    plt.figure(figsize=(16, 12))
    
    # 为每个角色分配颜色
    unique_labels = sorted(set(all_labels))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(all_labels) == label
        plt.scatter(
            features_tsne[mask, 0],
            features_tsne[mask, 1],
            c=[colors(i)],
            label=label,
            alpha=0.7,
            s=60
        )
    
    plt.title('Character Feature Clusters (t-SNE)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'tsne_clusters.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ t-SNE 可视化已保存: {output_path}")
    
    # 绘制混淆角色对的放大图
    plot_confused_pairs(features_tsne, all_labels, output_dir)
    
    plt.close()


def plot_confused_pairs(features_tsne, labels, output_dir):
    """
    绘制最容易混淆的角色对的可视化
    
    Args:
        features_tsne: t-SNE 降维后的特征
        labels: 标签列表
        output_dir: 输出目录
    """
    # 找出混淆最严重的角色对（基于距离）
    unique_labels = sorted(set(labels))
    label_to_indices = defaultdict(list)
    
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    # 计算角色对之间的平均距离
    dist_pairs = []
    
    for i, label_a in enumerate(unique_labels):
        for j in range(i + 1, len(unique_labels)):
            label_b = unique_labels[j]
            indices_a = label_to_indices[label_a]
            indices_b = label_to_indices[label_b]
            
            if len(indices_a) == 0 or len(indices_b) == 0:
                continue
            
            # 计算两个角色特征之间的平均距离
            dists = []
            for idx_a in indices_a:
                for idx_b in indices_b:
                    dist = np.linalg.norm(features_tsne[idx_a] - features_tsne[idx_b])
                    dists.append(dist)
            
            avg_dist = np.mean(dists)
            dist_pairs.append((label_a, label_b, avg_dist))
    
    # 按距离排序（距离越小越容易混淆）
    dist_pairs.sort(key=lambda x: x[2])
    
    print(f"\n最近的角色对（基于t-SNE距离）:")
    for label_a, label_b, dist in dist_pairs[:10]:
        print(f"   {label_a} ↔ {label_b}: {dist:.4f}")
    
    # 绘制最近的5对角色
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (label_a, label_b, _) in enumerate(dist_pairs[:3]):
        ax = axes[i]
        
        mask_a = np.array(labels) == label_a
        mask_b = np.array(labels) == label_b
        
        ax.scatter(features_tsne[mask_a, 0], features_tsne[mask_a, 1], c='red', label=label_a, alpha=0.7)
        ax.scatter(features_tsne[mask_b, 0], features_tsne[mask_b, 1], c='blue', label=label_b, alpha=0.7)
        
        ax.set_title(f'{label_a} vs {label_b}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confused_pairs.png')
    plt.savefig(output_path, dpi=150)
    print(f"✅ 混淆角色对可视化已保存: {output_path}")


def generate_character_report(feature_store):
    """
    生成完整的角色分析报告
    """
    print("\n" + "=" * 70)
    print("📋 角色特征分析报告")
    print("=" * 70)
    
    stats = feature_store.get_stats()
    print(f"特征库统计:")
    print(f"   角色数: {stats['total_characters']}")
    print(f"   特征数: {stats['total_features']}")
    print(f"   平均每角色特征数: {stats['total_features'] / stats['total_characters']:.1f}")
    
    # 计算内聚度
    intra_results = compute_intra_similarity(feature_store)
    
    # 计算角色间相似度
    inter_results = compute_inter_similarity(feature_store)
    
    # 分析混淆模式
    confusion_results = analyze_confusion_patterns(feature_store)
    
    # 生成汇总统计
    print("\n" + "=" * 70)
    print("📊 汇总统计")
    print("=" * 70)
    
    # 内聚度统计
    if intra_results:
        intra_means = [r['mean'] for r in intra_results.values()]
        print(f"角色内聚度:")
        print(f"   平均值: {np.mean(intra_means):.4f}")
        print(f"   标准差: {np.std(intra_means):.4f}")
        print(f"   范围: [{np.min(intra_means):.4f}, {np.max(intra_means):.4f}]")
    
    # 角色间相似度统计
    if inter_results:
        inter_sims = [r[2] for r in inter_results]
        print(f"\n角色间相似度:")
        print(f"   平均值: {np.mean(inter_sims):.4f}")
        print(f"   标准差: {np.std(inter_sims):.4f}")
        print(f"   最高相似度: {np.max(inter_sims):.4f}")
        print(f"   最低相似度: {np.min(inter_sims):.4f}")
    
    # 混淆统计
    if confusion_results:
        total_confusions = sum(r[2] for r in confusion_results)
        print(f"\n混淆模式:")
        print(f"   总误判对: {len(confusion_results)}")
        print(f"   总误判次数: {total_confusions}")
        print(f"   最严重混淆: {confusion_results[0][0]} → {confusion_results[0][1]} ({confusion_results[0][2]}次)")
    
    # 生成可视化
    visualize_character_clusters(feature_store)
    
    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="角色特征分析")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP模型名称")
    parser.add_argument("--index_path", type=str, default="data/feature_store/character_index.faiss", help="索引路径")
    parser.add_argument("--metadata_path", type=str, default="data/feature_store/character_metadata.json", help="元数据路径")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🎮 角色特征分析工具")
    print("=" * 70)
    print(f"模型: {args.clip_model}")
    print(f"特征库: {args.index_path}")
    print("=" * 70)
    
    # 创建并初始化检索器
    retriever = CharacterRetriever(
        clip_model_name=args.clip_model,
        feature_store_path=args.index_path,
        metadata_path=args.metadata_path,
        use_huggingface=True,
    )
    
    print("\n📥 初始化检索器...")
    retriever.initialize()
    
    # 生成分析报告
    generate_character_report(retriever.feature_store)


if __name__ == "__main__":
    sys.exit(main())
