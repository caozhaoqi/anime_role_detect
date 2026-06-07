#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.core.recognition.character_retriever import CharacterRetriever

def build_feature_store(args):
    """构建特征库"""
    retriever = CharacterRetriever(
        clip_model_name=args.clip_model,
        feature_store_path="data/feature_store/character_index.faiss",
        metadata_path="data/feature_store/character_metadata.json",
        use_huggingface=True,
    )
    retriever.initialize()
    
    # 删除旧索引
    for f in ["data/feature_store/character_index.faiss", "data/feature_store/character_metadata.json"]:
        if Path(f).exists():
            os.remove(f)
    
    # 确定原型模式
    use_prototype_val = args.multi_prototype if args.multi_prototype > 1 else (1 if args.use_prototype else False)
    
    # 注册角色
    dataset_path = Path("data/final_dataset")
    for i, character_dir in enumerate(dataset_path.iterdir()):
        if character_dir.is_dir():
            retriever.register_character_from_dir(
                character_name=character_dir.name,
                directory=str(character_dir),
                max_samples=10,
                use_prototype=use_prototype_val,
            )
    
    retriever.save()
    stats = retriever.get_stats()["feature_store"]
    print(f"✅ 特征库构建完成")
    print(f"   角色数: {stats['total_characters']}")
    print(f"   特征数: {stats['total_features']}")
    if args.multi_prototype > 1:
        print(f"   平均每角色原型数: {stats['total_features'] / stats['total_characters']:.1f}")
    return retriever

def evaluate_accuracy(retriever, args):
    """评估准确率"""
    print("\n" + "=" * 70)
    print("🧪 准确率测试")
    print("=" * 70)
    
    dataset_path = Path("data/final_dataset")
    total = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for character_dir in dataset_path.iterdir():
        if not character_dir.is_dir():
            continue
        
        character_name = character_dir.name
        image_files = list(character_dir.glob("*.png")) + list(character_dir.glob("*.jpg")) + list(character_dir.glob("*.jpeg"))
        
        for img_path in image_files[:5]:  # 每个角色测试5张
            total += 1
            result = retriever.identify(str(img_path), top_k=10)
            predictions = [r['character'] for r in result]
            
            # Top-K准确率
            if character_name == predictions[0]:
                top1_correct += 1
            if character_name in predictions[:3]:
                top3_correct += 1
            if character_name in predictions[:5]:
                top5_correct += 1
            if character_name in predictions[:10]:
                top10_correct += 1
            
            # 混淆矩阵
            confusion_matrix[character_name][predictions[0]] += 1
    
    print(f"测试样本数: {total}")
    print(f"Top-1 准确率: {top1_correct/total*100:.2f}% ({top1_correct}/{total})")
    print(f"Top-3 准确率: {top3_correct/total*100:.2f}% ({top3_correct}/{total})")
    print(f"Top-5 准确率: {top5_correct/total*100:.2f}% ({top5_correct}/{total})")
    print(f"Top-10 准确率: {top10_correct/total*100:.2f}% ({top10_correct}/{total})")
    
    # 输出混淆矩阵前20
    print("\n混淆矩阵（前20对）:")
    print(f"{'真实':<12} {'预测':<12} {'次数':<6}")
    print("-" * 30)
    error_pairs = []
    for true_char, preds in confusion_matrix.items():
        for pred_char, count in preds.items():
            if true_char != pred_char:
                error_pairs.append((true_char, pred_char, count))
    error_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_char, pred_char, count in error_pairs[:20]:
        print(f"{true_char:<12} {pred_char:<12} {count:<6}")

def main():
    parser = argparse.ArgumentParser(description="构建特征库并测试")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP模型")
    parser.add_argument("--use_prototype", action="store_true", help="单原型模式")
    parser.add_argument("--multi_prototype", type=int, default=0, help="多原型模式(K值)")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🎮 动漫角色识别测试")
    print("=" * 70)
    print(f"模型: {args.clip_model}")
    if args.multi_prototype > 1:
        print(f"模式: Multi-Prototype (K={args.multi_prototype})")
    elif args.use_prototype:
        print("模式: 单原型")
    else:
        print("模式: 每图入库")
    print("=" * 70)
    
    retriever = build_feature_store(args)
    evaluate_accuracy(retriever, args)

if __name__ == "__main__":
    main()
