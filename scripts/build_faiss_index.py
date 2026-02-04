#!/usr/bin/env python3
"""
FAISS索引构建脚本
负责从数据目录提取特征并构建向量索引，保存为文件供Web应用使用。
"""
import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.general_classification import GeneralClassification

def build_index(data_dir, output_path):
    """
    构建索引并保存
    :param data_dir: 数据目录路径
    :param output_path: 输出索引文件路径（不含扩展名）
    """
    print(f"=== 开始构建索引 ===")
    print(f"数据目录: {data_dir}")
    print(f"输出路径: {output_path}")
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return False
        
    # 初始化分类器
    classifier = GeneralClassification()
    
    # 构建索引
    success = classifier.build_index_from_directory(data_dir)
    
    if success:
        # 保存索引
        save_success = classifier.save_index(output_path)
        if save_success:
            print(f"=== 索引构建并保存成功! ===")
            print(f"索引文件: {output_path}.faiss")
            print(f"映射文件: {output_path}_mapping.json")
            return True
        else:
            print("错误: 保存索引失败")
            return False
    else:
        print("错误: 构建索引失败")
        return False

def main():
    parser = argparse.ArgumentParser(description="构建FAISS向量索引")
    parser.add_argument("--data_dir", default="data/all_characters", help="包含角色图片的根目录")
    parser.add_argument("--output_path", default="role_index", help="输出索引文件的路径（不含扩展名）")
    
    args = parser.parse_args()
    
    # 优先尝试增强数据目录
    data_dir = args.data_dir
    if data_dir == "data/all_characters" and os.path.exists("data/augmented_characters"):
        print("发现增强数据目录，优先使用: data/augmented_characters")
        data_dir = "data/augmented_characters"
    
    build_index(data_dir, args.output_path)

if __name__ == "__main__":
    main()
