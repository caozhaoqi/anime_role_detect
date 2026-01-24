#!/usr/bin/env python3
"""
完整流程脚本：下载角色图片 → 打散 → 分类
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

from tests.collect_test_data import collect_single_character_data
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

def download_character_images(roles, output_base_dir, image_limit=5):
    """下载角色图片"""
    print("=== 下载角色图片 ===")
    
    # 确保输出目录存在
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    downloaded_roles = {}
    
    for role in roles:
        output_dir = os.path.join(output_base_dir, role)
        
        # 清理旧目录
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        print(f"\n下载角色 '{role}' 的图片...")
        count = collect_single_character_data(role, image_limit, output_dir)
        
        if count > 0:
            downloaded_roles[role] = output_dir
            print(f"✓ 成功下载 {count} 张图片")
        else:
            print(f"✗ 无法下载图片")
    
    print(f"\n下载完成，成功下载 {len(downloaded_roles)} 个角色的图片")
    return downloaded_roles

def shuffle_images(downloaded_roles, output_dir):
    """打散图片"""
    print("\n=== 打散图片 ===")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 收集所有图片
    all_images = []
    for role, role_dir in downloaded_roles.items():
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
        return False
    
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
        return False
    
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
        "unknown": 0
    }
    
    # 创建分类结果目录
    classification_dir = os.path.join(output_dir, "classification_results")
    if not os.path.exists(classification_dir):
        os.makedirs(classification_dir)
    
    for img_path, true_role in shuffled_images:
        results["total"] += 1
        
        try:
            # 预处理
            normalized_img = basic_process(img_path)
            if normalized_img is None:
                results["unknown"] += 1
                continue
            
            # 提取特征
            feature = extractor.extract_features(normalized_img)
            
            # 分类
            predicted_role, similarity = classifier.classify(feature)
            
            # 记录结果
            if predicted_role == "unknown" or similarity < 0.7:
                results["unknown"] += 1
                status = "unknown"
            elif predicted_role == true_role:
                results["correct"] += 1
                status = "correct"
            else:
                results["incorrect"] += 1
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
            continue
    
    # 计算准确率
    if results["total"] > 0:
        accuracy = (results["correct"] / results["total"]) * 100
        print(f"\n=== 分类结果 ===")
        print(f"总图片数: {results['total']}")
        print(f"正确分类: {results['correct']}")
        print(f"错误分类: {results['incorrect']}")
        print(f"无法分类: {results['unknown']}")
        print(f"准确率: {accuracy:.2f}%")
        
        # 保存结果报告
        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("分类报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"总图片数: {results['total']}\n")
            f.write(f"正确分类: {results['correct']}\n")
            f.write(f"错误分类: {results['incorrect']}\n")
            f.write(f"无法分类: {results['unknown']}\n")
            f.write(f"准确率: {accuracy:.2f}%\n")
            f.write("=" * 50 + "\n")
        
        print(f"✓ 分类报告已保存到 {report_path}")
        
    return True

def main():
    """主函数"""
    print("=== 开始完整流程 ===")
    
    # 配置
    roles = [
        "genshin_impact lumine",    # 原神 荧
        "genshin_impact aether",     # 原神 空
        "genshin_impact jean",       # 原神 琴
        "genshin_impact venti",      # 原神 温迪
        "wuthering_waves anby",      # 鸣潮 anby
        "wuthering_waves bianca"     # 鸣潮 bianca
    ]
    
    base_dir = "data/process_and_classify"
    download_dir = os.path.join(base_dir, "downloaded")
    shuffle_dir = os.path.join(base_dir, "shuffled")
    
    # 1. 下载角色图片
    print("\n[步骤 1] 下载角色图片...")
    downloaded_roles = download_character_images(roles, download_dir, image_limit=3)
    
    if not downloaded_roles:
        print("❌ 无法下载任何角色图片，终止流程")
        return 1
    
    # 2. 打散图片
    print("\n[步骤 2] 打散图片...")
    shuffled_images = shuffle_images(downloaded_roles, shuffle_dir)
    
    if not shuffled_images:
        print("❌ 没有图片可打散，终止流程")
        return 1
    
    # 3. 分类图片
    print("\n[步骤 3] 分类图片...")
    success = classify_images(shuffled_images, base_dir)
    
    if success:
        print("\n🎉 完整流程执行成功！")
        print(f"\n结果目录:")
        print(f"- 下载的图片: {download_dir}")
        print(f"- 打散的图片: {shuffle_dir}")
        print(f"- 分类结果: {os.path.join(base_dir, 'classification_results')}")
        print(f"- 分类报告: {os.path.join(base_dir, 'classification_report.txt')}")
        return 0
    else:
        print("\n❌ 分类失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
