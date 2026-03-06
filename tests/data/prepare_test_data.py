#!/usr/bin/env python3
"""
测试数据准备脚本
"""
import os
import shutil
import argparse


def prepare_test_data(dataset_dir, test_dir):
    """从数据集准备测试数据"""
    # 检查数据集目录
    if not os.path.exists(dataset_dir):
        print(f"数据集目录不存在: {dataset_dir}")
        return
    
    # 检查测试目录
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # 获取角色目录
    role_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not role_dirs:
        print(f"数据集目录中没有角色子目录: {dataset_dir}")
        return
    
    print(f"发现 {len(role_dirs)} 个角色")
    
    # 为每个角色准备测试数据
    for role in role_dirs:
        role_dir = os.path.join(dataset_dir, role)
        test_role_dir = os.path.join(test_dir, role)
        
        # 创建测试角色目录
        if not os.path.exists(test_role_dir):
            os.makedirs(test_role_dir)
        
        # 获取角色图片
        image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not image_files:
            print(f"角色 {role} 没有图片")
            continue
        
        # 选择前10张图片作为测试数据
        test_images = image_files[:10]
        print(f"为角色 {role} 准备 {len(test_images)} 张测试图片")
        
        # 复制图片到测试目录
        for img_file in test_images:
            src_path = os.path.join(role_dir, img_file)
            dst_path = os.path.join(test_role_dir, img_file)
            shutil.copy2(src_path, dst_path)
    
    print(f"测试数据准备完成，保存到: {test_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试数据准备脚本")
    parser.add_argument("--dataset_dir", default="dataset", help="数据集目录")
    parser.add_argument("--test_dir", default="tests/test_images/single_character", help="测试数据目录")
    
    args = parser.parse_args()
    prepare_test_data(args.dataset_dir, args.test_dir)


if __name__ == "__main__":
    main()
