#!/usr/bin/env python3
"""
蔚蓝档案角色分类测试脚本
使用优化后采集的角色图片进行分类测试
"""
import os
import sys
import shutil
import random
import numpy as np
from PIL import Image

# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification

def basic_process(image_path):
    """基础预处理"""
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"预处理失败: {e}")
        return None

def shuffle_images(downloaded_dir, output_dir):
    """打散图片"""
    print("=== 打散图片 ===")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 收集所有图片
    all_images = []
    
    # 获取所有角色目录
    role_dirs = [d for d in os.listdir(downloaded_dir) if os.path.isdir(os.path.join(downloaded_dir, d))]
    
    for role in role_dirs:
        role_dir = os.path.join(downloaded_dir, role)
        image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in image_files:
            img_path = os.path.join(role_dir, img_file)
            all_images.append((img_path, role))
    
    print(f"共收集到 {len(all_images)} 张图片")
    
    # 打乱图片顺序
    random.shuffle(all_images)
    print("✓ 图片已打乱")
    
    # 复制打乱后的图片到输出目录
    for i, (img_path, role) in enumerate(all_images):
        # 生成新文件名
        ext = os.path.splitext(img_path)[1]
        new_filename = f"shuffled_{i+1}_{role.replace(' ', '_')}{ext}"
        new_path = os.path.join(output_dir, new_filename)
        
        # 复制图片
        shutil.copy2(img_path, new_path)
        
        if (i + 1) % 10 == 0:
            print(f"已复制 {i+1}/{len(all_images)} 张图片")
    
    print(f"✓ 所有图片已复制到 {output_dir}")
    return all_images

def classify_images(shuffled_images, output_dir):
    """分类图片"""
    print("\n=== 分类图片 ===")
    
    # 初始化系统
    print("初始化系统模块...")
    try:
        extractor = FeatureExtraction()
        classifier = Classification()
        print("✓ 系统模块初始化成功")
    except Exception as e:
        print(f"✗ 系统模块初始化失败: {e}")
        return False, {}
    
    # 构建索引
    print("\n构建特征索引...")
    
    # 为每个角色选择一张图片构建索引
    role_features = {}
    
    for img_path, role in shuffled_images:
        if role not in role_features:
            try:
                # 预处理
                normalized_img = basic_process(img_path)
                if normalized_img is None:
                    continue
                
                # 提取特征
                feature = extractor.extract_features(normalized_img)
                role_features[role] = feature
                print(f"✓ 为角色 {role} 提取特征")
            except Exception as e:
                print(f"✗ 处理角色 {role} 失败: {e}")
    
    if len(role_features) < 2:
        print("✗ 无法构建索引：角色数量不足")
        return False, {}
    
    # 构建索引
    features = list(role_features.values())
    roles = list(role_features.keys())
    
    features_np = np.array(features).astype(np.float32)
    classifier.build_index(features_np, roles)
    print(f"✓ 索引构建成功，包含 {len(roles)} 个角色")
    
    # 分类所有图片
    print("\n分类所有图片...")
    
    results = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "unknown": 0,
        "role_wise": {}
    }
    
    # 初始化角色级别的结果
    for role in roles:
        results["role_wise"][role] = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "unknown": 0
        }
    
    # 创建分类结果目录
    classification_dir = os.path.join(output_dir, "classification_results")
    if not os.path.exists(classification_dir):
        os.makedirs(classification_dir)
    
    for img_path, true_role in shuffled_images:
        results["total"] += 1
        
        # 初始化角色级别的计数
        if true_role not in results["role_wise"]:
            results["role_wise"][true_role] = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "unknown": 0
            }
        results["role_wise"][true_role]["total"] += 1
        
        try:
            # 预处理
            normalized_img = basic_process(img_path)
            if normalized_img is None:
                results["unknown"] += 1
                results["role_wise"][true_role]["unknown"] += 1
                continue
            
            # 提取特征
            feature = extractor.extract_features(normalized_img)
            
            # 分类
            predicted_role, similarity = classifier.classify(feature)
            
            # 记录结果
            if predicted_role == "unknown" or similarity < 0.7:
                results["unknown"] += 1
                results["role_wise"][true_role]["unknown"] += 1
                status = "unknown"
            elif predicted_role == true_role:
                results["correct"] += 1
                results["role_wise"][true_role]["correct"] += 1
                status = "correct"
            else:
                results["incorrect"] += 1
                results["role_wise"][true_role]["incorrect"] += 1
                status = "incorrect"
            
            # 复制到对应结果目录
            status_dir = os.path.join(classification_dir, status)
            if not os.path.exists(status_dir):
                os.makedirs(status_dir)
            
            # 复制图片
            img_filename = os.path.basename(img_path)
            dest_path = os.path.join(status_dir, img_filename)
            shutil.copy2(img_path, dest_path)
            
            # 打印进度
            if results["total"] % 10 == 0:
                print(f"已分类 {results['total']}/{len(shuffled_images)} 张图片")
                print(f"  正确: {results['correct']}, 错误: {results['incorrect']}, 未知: {results['unknown']}")
                
        except Exception as e:
            print(f"✗ 分类图片 {os.path.basename(img_path)} 失败: {e}")
            results["unknown"] += 1
            results["role_wise"][true_role]["unknown"] += 1
            continue
    
    return True, results

def generate_report(results, output_dir):
    """生成详细的分类报告"""
    print("\n=== 生成分类报告 ===")
    
    # 计算准确率
    if results["total"] > 0:
        accuracy = (results["correct"] / results["total"]) * 100
    else:
        accuracy = 0
    
    # 生成报告内容
    report_content = """
蔚蓝档案角色分类测试报告
=======================

1. 总体分类结果
----------------
"""
    
    report_content += f"总测试图片数: {results['total']}\n"
    report_content += f"正确分类数: {results['correct']}\n"
    report_content += f"错误分类数: {results['incorrect']}\n"
    report_content += f"无法分类数: {results['unknown']}\n"
    report_content += f"总体准确率: {accuracy:.2f}%\n"
    
    report_content += "\n2. 角色级别分类结果\n"
    report_content += "------------------\n"
    
    for role, role_results in results["role_wise"].items():
        if role_results["total"] > 0:
            role_accuracy = (role_results["correct"] / role_results["total"]) * 100
        else:
            role_accuracy = 0
        
        report_content += f"\n角色: {role}\n"
        report_content += f"  测试图片数: {role_results['total']}\n"
        report_content += f"  正确分类数: {role_results['correct']}\n"
        report_content += f"  错误分类数: {role_results['incorrect']}\n"
        report_content += f"  无法分类数: {role_results['unknown']}\n"
        report_content += f"  准确率: {role_accuracy:.2f}%\n"
    
    report_content += "\n3. 分析与建议\n"
    report_content += "--------------\n"
    
    if accuracy >= 70:
        report_content += "✓ 分类性能良好！系统能够较好地区分不同角色。\n"
    elif accuracy >= 50:
        report_content += "⚠️  分类性能一般，存在一定的误分类情况。\n"
    else:
        report_content += "✗ 分类性能较差，需要进一步优化。\n"
    
    report_content += "\n建议：\n"
    report_content += "- 增加每个角色的训练样本数量\n"
    report_content += "- 尝试使用更精确的角色标签\n"
    report_content += "- 考虑调整特征提取和分类算法参数\n"
    report_content += "- 对相似角色进行更细致的特征分析\n"
    
    # 保存报告
    report_path = os.path.join(output_dir, "blue_archive_optimized_classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ 分类报告已保存到 {report_path}")
    
    # 打印关键结果
    print("\n=== 关键分类结果 ===")
    print(f"总体准确率: {accuracy:.2f}%")
    print(f"正确分类: {results['correct']}/{results['total']}")
    print(f"错误分类: {results['incorrect']}/{results['total']}")
    print(f"无法分类: {results['unknown']}/{results['total']}")
    
    print("\n角色级别准确率:")
    for role, role_results in results["role_wise"].items():
        if role_results["total"] > 0:
            role_accuracy = (role_results["correct"] / role_results["total"]) * 100
            print(f"  {role}: {role_accuracy:.2f}%")
    
    return report_path

def main():
    """主函数"""
    print("=== 蔚蓝档案角色分类测试 ===")
    
    # 配置
    base_dir = "data/blue_archive_optimized_v2"
    downloaded_dir = base_dir  # 直接使用base_dir，因为角色目录已经在这里
    shuffle_dir = os.path.join(base_dir, "shuffled")
    
    # 检查下载目录是否存在
    if not os.path.exists(downloaded_dir):
        print(f"❌ 下载目录不存在: {downloaded_dir}")
        print("请先运行 collect_blue_archive.py 脚本下载角色图片")
        return 1
    
    # 1. 打散图片
    print("\n[步骤 1] 打散图片...")
    shuffled_images = shuffle_images(downloaded_dir, shuffle_dir)
    
    if not shuffled_images:
        print("❌ 没有图片可打散，终止流程")
        return 1
    
    # 2. 分类图片
    print("\n[步骤 2] 分类图片...")
    success, results = classify_images(shuffled_images, base_dir)
    
    if success:
        # 3. 生成报告
        print("\n[步骤 3] 生成分类报告...")
        report_path = generate_report(results, base_dir)
        
        print("\n🎉 蔚蓝档案角色分类测试完成！")
        print(f"\n结果目录:")
        print(f"- 下载的图片: {downloaded_dir}")
        print(f"- 打散的图片: {shuffle_dir}")
        print(f"- 分类结果: {os.path.join(base_dir, 'classification_results')}")
        print(f"- 分类报告: {report_path}")
        return 0
    else:
        print("\n❌ 分类失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
