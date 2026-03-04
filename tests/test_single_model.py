#!/usr/bin/env python3
"""
测试单个模型的分类准确度
使用少量样本测试不同模型的性能
"""
import os
import json
import argparse
from src.core.classification.general_classification import GeneralClassification

def load_sample_data(test_dir, max_samples=10):
    """加载测试数据
    
    Args:
        test_dir: 测试数据目录
        max_samples: 每个角色最大样本数
        
    Returns:
        list: 测试数据列表，每个元素为 (image_path, true_label)
    """
    test_data = []
    
    # 检查目录是否存在
    if not os.path.exists(test_dir):
        print(f"测试目录不存在: {test_dir}")
        # 使用tests/img目录下的图片作为测试数据
        tests_img_dir = os.path.join(os.getcwd(), "", "img")
        if os.path.exists(tests_img_dir):
            print(f"使用tests/img目录: {tests_img_dir} 中的图片作为测试数据")
            
            # 遍历tests/img目录下的所有图片文件
            count = 0
            for image_file in os.listdir(tests_img_dir):
                if count >= max_samples:
                    break
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    image_path = os.path.join(tests_img_dir, image_file)
                    test_data.append((image_path, "test"))
                    count += 1
            
            return test_data
        else:
            # 检查是否有其他测试图片
            test_image = os.path.join(os.getcwd(), "", "img", "微信图片_20260204115846_481_347.jpg")
            if os.path.exists(test_image):
                print(f"使用测试图片: {test_image}")
                test_data.append((test_image, "test"))
                return test_data
            else:
                print("没有找到测试图片，退出")
                return test_data
    
    # 遍历测试目录下的所有子目录
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue
        
        # 遍历子目录下的所有图片文件
        count = 0
        for image_file in os.listdir(label_dir):
            if count >= max_samples:
                break
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                image_path = os.path.join(label_dir, image_file)
                test_data.append((image_path, label))
                count += 1
    
    return test_data

def test_model(test_data, classifier, use_model=False):
    """测试模型
    
    Args:
        test_data: 测试数据
        classifier: 分类器实例
        use_model: 是否使用EfficientNet模型
        
    Returns:
        dict: 测试结果
    """
    model_name = "EfficientNet" if use_model else "CLIP"
    print(f"\n=== 测试{model_name}模型 ===")
    correct = 0
    total = len(test_data)
    results = []
    
    for i, (image_path, true_label) in enumerate(test_data):
        print(f"测试 {i+1}/{total}: {os.path.basename(image_path)}")
        try:
            # 分类图像
            predicted_label, similarity, _ = classifier.classify_image(image_path, use_model=use_model)
            
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
            
            print(f"  真实标签: {true_label}")
            print(f"  预测标签: {predicted_label}")
            print(f"  相似度: {similarity:.4f}")
            print(f"  结果: {'正确' if is_correct else '错误'}")
            print()
            
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
    print(f"{model_name}模型准确度: {accuracy:.4f} ({correct}/{total})")
    
    return {
        'model': model_name,
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
            
            print(f"  真实标签: {true_label}")
            print(f"  预测标签: {predicted_label}")
            print(f"  相似度: {similarity:.4f}")
            print(f"  结果: {'正确' if is_correct else '错误'}")
            print()
            
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
    parser.add_argument('--test_dir', type=str, default='data/all_characters', help='测试数据目录')
    parser.add_argument('--index_path', type=str, default='role_index', help='索引路径')
    parser.add_argument('--max_samples', type=int, default=5, help='每个角色最大样本数')
    args = parser.parse_args()
    
    # 加载测试数据
    print(f"加载测试数据: {args.test_dir}")
    test_data = load_sample_data(args.test_dir, args.max_samples)
    print(f"加载了 {len(test_data)} 个测试样本")
    
    if not test_data:
        print("没有找到测试数据，退出")
        return
    
    # 初始化分类器
    print("初始化分类器...")
    classifier = GeneralClassification(index_path=args.index_path)
    classifier.initialize()
    
    # 测试各个模型
    clip_results = test_model(test_data, classifier, use_model=False)
    efficientnet_results = test_model(test_data, classifier, use_model=True)
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
    output_file = 'model_accuracy_results_short.json'
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
