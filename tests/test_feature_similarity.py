import os
import sys
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath('.'))

from src.core.feature_extraction.feature_extraction import FeatureExtraction

def calculate_similarity(feature1, feature2):
    """计算两个特征向量的余弦相似度"""
    return np.dot(feature1, feature2)

def main():
    # 初始化特征提取器
    extractor = FeatureExtraction()
    
    # 加载日奈_1.jpg
    hina_path = 'data/train/日奈/日奈_1.jpg'
    hina_img = Image.open(hina_path)
    hina_feature = extractor.extract_features(hina_img)
    print(f"日奈_1.jpg特征维度: {hina_feature.shape}")
    
    # 加载伊织_1.jpg
    izumi_path = 'data/train/伊织/伊织_1.jpg'
    izumi_img = Image.open(izumi_path)
    izumi_feature = extractor.extract_features(izumi_img)
    print(f"伊织_1.jpg特征维度: {izumi_feature.shape}")
    
    # 计算相似度
    similarity = calculate_similarity(hina_feature, izumi_feature)
    print(f"日奈_1.jpg与伊织_1.jpg的相似度: {similarity:.4f}")
    
    # 加载其他日奈图片进行对比
    hina2_path = 'data/train/日奈/日奈_2.jpg'
    hina2_img = Image.open(hina2_path)
    hina2_feature = extractor.extract_features(hina2_img)
    similarity2 = calculate_similarity(hina_feature, hina2_feature)
    print(f"日奈_1.jpg与日奈_2.jpg的相似度: {similarity2:.4f}")

if __name__ == "__main__":
    main()