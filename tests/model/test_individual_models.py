#!/usr/bin/env python3
"""
测试单个模型的分类准确度
分别测试CLIP、EfficientNet和DeepDanbooru模型的性能
"""
import os
import json
import argparse
from src.core.classification.general_classification import GeneralClassification

def load_test_data(test_dir):
    """加载测试数据
    
    Args:
        test_dir: 测试数据目录
        
    Returns:
        list: 测试数据列表，每个元素为 (image_path, true_label)
    """
    test_data = []
    
    # 遍历测试目录下的所有子目录
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue
        
        # 遍历子目录下的所有图片文件
        for image_file in os.listdir(label_dir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                image_path = os.path.join(label_dir, image_file)
                test_data.append((image_path, label))
    
    return test_data

def test_clip_model(test_data, classifier):
    """测试CLIP模型
    
    Args:
        test_data: 测试数据
        classifier: 分类器实例
        
    Returns:
        dict: 测试结果
    """
    print("\n=== 测试CLIP模型 ===")
    correct = 0
    total = len(test_data)
    results = []
    
    for i, (image_path, true_label) in enumerate(test_data):
        print(f"测试 {i+1}/{total}: {os.path.basename(image_path)}")
        try:
            # 使用CLIP模型（use_model=False）
            predicted_label, similarity, _ = classifier.classify_image(image_path, use_model=False)
            
            # 检查预测是否正确
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            
            results.append({
                'image': os.path.basename(image_path),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'similarity': float(similarity),
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"测试失败: {e}")
            results.append({
                'image': os.path.basename(image_path),
                'true_label': true_label,
                'predicted_label': 'error',
                'similarity': 0.0,
                'correct': False
            })
    
    accuracy = correct / total if total > 0 else 0
    print(f"CLIP模型准确度: {accuracy:.4f} ({correct}/{total})")
    
    return {
        'model': 'CLIP',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }

def test_efficientnet_model(test_data, classifier):
    """测试EfficientNet模型
    
    Args:
        test_data: 测试数据
        classifier: 分类器实例
        
    Returns:
        dict: 测试结果
    """
    print("\n=== 测试EfficientNet模型 ===")
    correct = 0
    total = len(test_data)
    results = []
    
    for i, (image_path, true_label) in enumerate(test_data):
        print(f"测试 {i+1}/{total}: {os.path.basename(image_path)}")
        try:
            # 使用EfficientNet模型（use_model=True）
            predicted_label, similarity, _ = classifier.classify_image(image_path, use_model=True)
            
            # 检查预测是否正确
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            
            results.append({
                'image': os.path.basename(image_path),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'similarity': float(similarity),
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"测试失败: {e}")
            results.append({
                'image': os.path.basename(image_path),
                'true_label': true_label,
                'predicted_label': 'error',
                'correct': False
            })
    
    accuracy = correct / total if total > 0 else 0
    print(f"EfficientNet模型准确度: {accuracy:.4f} ({correct}/{total})")
    
    return {
        'model': 'EfficientNet',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }

def test_ensemble_model(test_data, classifier):
    """测试集成模型
    
    Args:
        test_data: 测试数据
        classifier: 分类器实例
        
    Returns:
        dict: 测试结果
    """
    print("\n=== 测试集成模型 ===")
    correct = 0
    total = len(test_data)
    results = []
    
    for i, (image_path, true_label) in enumerate(test_data):
        print(f"测试 {i+1}/{total}: {os.path.basename(image_path)}")
        try:
            # 使用集成模型
            predicted_label, similarity, _ = classifier.classify_image_with_deepdanbooru(image_path)
            
            # 检查预测是否正确
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            
            results.append({
                'image': os.path.basename(image_path),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'similarity': float(similarity),
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"测试失败: {e}")
            results.append({
                'image': os.path.basename(image_path),
                'true_label': true_label,
                'predicted_label': 'error',
                'correct': False
            })
    
    accuracy = correct / total if total > 0 else 0
    print(f"集成模型准确度: {accuracy:.4f} ({correct}/{total})")
    
    return {
        'model': 'Ensemble',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试单个模型的分类准确度')
    parser.add_argument('--test_dir', type=str, default='data/augmented_dataset', help='测试数据目录')
    parser.add_argument('--index_path', type=str, default='role_index', help='索引路径')
    args = parser.parse_args()
    
    # 加载测试数据
    print(f"加载测试数据: {args.test_dir}")
    test_data = load_test_data(args.test_dir)
    print(f"加载了 {len(test_data)} 个测试样本")
    
    if not test_data:
        print("没有找到测试数据，退出")
        return
    
    # 初始化分类器
    print("初始化分类器...")
    classifier = GeneralClassification(index_path=args.index_path)
    classifier.initialize()
    
    # 测试各个模型
    clip_results = test_clip_model(test_data, classifier)
    efficientnet_results = test_efficientnet_model(test_data, classifier)
    ensemble_results = test_ensemble_model(test_data, classifier)
    
    # 汇总结果
    all_results = {
        'clip': clip_results,
        'efficientnet': efficientnet_results,
        'ensemble': ensemble_results,
        'summary': {
            'total_test_samples': len(test_data),
            'best_model': max([clip_results, efficientnet_results, ensemble_results], key=lambda x: x['accuracy'])['model'],
            'best_accuracy': max([clip_results, efficientnet_results, ensemble_results], key=lambda x: x['accuracy'])['accuracy']
        }
    }
    
    # 保存结果
    output_file = 'model_accuracy_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试结果已保存到: {output_file}")
    
    # 打印摘要
    print("\n=== 测试结果摘要 ===")
    print(f"CLIP模型: {clip_results['accuracy']:.4f}")
    print(f"EfficientNet模型: {efficientnet_results['accuracy']:.4f}")
    print(f"集成模型: {ensemble_results['accuracy']:.4f}")
    print(f"最佳模型: {all_results['summary']['best_model']} ({all_results['summary']['best_accuracy']:.4f})")

if __name__ == '__main__':
    main()
