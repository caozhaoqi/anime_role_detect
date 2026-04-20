#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据提取到的特征对角色进行进一步分类
"""

import os

def classify_by_features(input_file):
    """
    根据提取到的特征对角色进行进一步分类
    
    Args:
        input_file: 分类结果文件路径
    """
    # 特征分类映射
    feature_categories = {
        "万年萝莉": [],
        "幼女": [],
        "小学生": [],
        "御姐": [],
        "巨乳": [],
        "熟女": [],
        "身高偏矮": []
    }
    
    # 读取分类结果文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"共读取到 {len(lines)} 行数据")
    
    # 解析文件内容
    current_category = None
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 打印每一行的内容
        print(f"第 {i+1} 行: {line}")
        
        # 跳过空行和统计信息行
        if not line or line.startswith("分类结果统计:") or line.startswith("详细分类结果:"):
            continue
        
        # 检查是否是分类标题
        if line.endswith(":"):
            current_category = line[:-1]
            print(f"当前分类: {current_category}")
            continue
        
        # 解析角色信息
        if current_category and " - 特征: " in line:
            # 提取角色名和特征
            parts = line.split(" - 特征: ")
            if len(parts) == 2:
                character_name = parts[0].strip()
                features_str = parts[1]
                
                # 解析特征列表
                if features_str == "无":
                    features = []
                else:
                    # 移除 [] 并分割特征
                    features_str = features_str.strip('[]')
                    # 处理特征列表，注意处理包含空格的特征
                    features = []
                    # 处理单引号包围的特征
                    if "', '" in features_str:
                        for f in features_str.split("', '"):
                            feature = f.strip().strip("'").strip('"')
                            if feature:
                                features.append(feature)
                    else:
                        # 尝试其他分割方式
                        feature = features_str.strip().strip("'").strip('"')
                        if feature:
                            features.append(feature)
                
                # 打印调试信息
                print(f"角色: {character_name}, 特征: {features}")
                
                # 根据特征分类
                for feature in features:
                    # 处理反向特征
                    if feature.endswith('(-)'):
                        feature_name = feature[:-3]
                        if feature_name in feature_categories:
                            feature_categories[feature_name].append(character_name)
                            print(f"添加到反向特征 {feature_name}: {character_name}")
                    else:
                        if feature in feature_categories:
                            feature_categories[feature].append(character_name)
                            print(f"添加到特征 {feature}: {character_name}")
                        # 特殊处理：检查特征是否包含关键词
                        for category in feature_categories:
                            if category in feature:
                                feature_categories[category].append(character_name)
                                print(f"添加到特征 {category}: {character_name}")
    
    # 输出特征分类结果
    output_dir = os.path.dirname(input_file)
    feature_output_dir = os.path.join(output_dir, "features")
    if not os.path.exists(feature_output_dir):
        os.makedirs(feature_output_dir)
    
    # 输出总特征分类结果
    with open(os.path.join(feature_output_dir, "feature_classification_results.txt"), "w", encoding="utf-8") as f:
        f.write("特征分类结果:\n")
        for feature, characters in feature_categories.items():
            f.write(f"\n{feature}: {len(characters)} 个角色\n")
            for character in characters:
                f.write(f"  {character}\n")
    
    # 按特征输出到不同文件
    for feature, characters in feature_categories.items():
        if characters:
            with open(os.path.join(feature_output_dir, f"{feature}.txt"), "w", encoding="utf-8") as f:
                for character in characters:
                    f.write(f"{character}\n")
    
    print(f"特征分类结果已输出到 {feature_output_dir} 目录")
    
    # 打印特征分类统计
    print("\n特征分类统计:")
    for feature, characters in feature_categories.items():
        print(f"{feature}: {len(characters)} 个角色")

if __name__ == "__main__":
    input_file = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lolis/classified/classification_results.txt"
    print(f"读取文件: {input_file}")
    print(f"文件是否存在: {os.path.exists(input_file)}")
    if os.path.exists(input_file):
        print(f"文件大小: {os.path.getsize(input_file)} 字节")
    classify_by_features(input_file)
