import os
import numpy as np
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath('.'))

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification

def main():
    # 数据目录
    data_dir = 'data/downloaded_images'
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    # 获取角色目录列表
    role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"发现 {len(role_dirs)} 个角色目录")
    
    # 打印所有角色目录
    print("所有角色目录:")
    for i, role_dir in enumerate(role_dirs):
        print(f"{i+1}. {role_dir}")
    
    # 检查是否包含主要角色目录
    main_roles = ['日奈', '普拉娜', '阿罗娜', '伊织', '亚子']
    print("\n检查主要角色目录:")
    for role in main_roles:
        if role in role_dirs:
            role_path = os.path.join(data_dir, role)
            image_files = [f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"✓ {role} 目录存在，包含 {len(image_files)} 张图片")
        else:
            print(f"✗ {role} 目录不存在")
    
    print("\n开始处理图像...")
    
    # 初始化模块
    preprocessor = Preprocessing()
    extractor = FeatureExtraction()
    classifier = Classification()
    
    # 提取所有角色的特征向量
    all_features = []
    all_role_names = []
    
    # 处理所有角色目录
    for role_name in role_dirs:
        role_dir = os.path.join(data_dir, role_name)
        # 只处理.jpg, .jpeg, .png, .bmp, .webp格式的文件
        image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        # 跳过没有图片的目录
        if len(image_files) == 0:
            print(f"跳过角色 '{role_name}'，没有图片")
            continue
        
        print(f"处理角色 '{role_name}'，发现 {len(image_files)} 张图片")
        
        for img_file in image_files:
            img_path = os.path.join(role_dir, img_file)
            
            try:
                # 预处理图像
                normalized_img, _ = preprocessor.process(img_path)
                
                # 提取特征
                feature = extractor.extract_features(normalized_img)
                
                # 添加到列表
                all_features.append(feature)
                all_role_names.append(role_name)
                
                # 每处理10张图片打印一次进度
                if len(all_features) % 10 == 0:
                    print(f"已处理 {len(all_features)} 张图片")
                
            except Exception as e:
                print(f"处理图像 {img_path} 失败: {e}")
                continue
    
    if not all_features:
        print("没有成功提取任何特征向量")
        return
    
    # 转换为numpy数组
    features_array = np.array(all_features, dtype=np.float32)
    
    print(f"成功提取 {len(features_array)} 个特征向量")
    
    # 构建索引
    classifier.build_index(features_array, all_role_names)
    
    # 保存索引
    index_path = 'role_index'
    classifier.save_index(index_path)
    print(f"索引已保存到 {index_path}.faiss 和 {index_path}_mapping.json")
    
    # 检查索引文件
    if os.path.exists('role_index.faiss') and os.path.exists('role_index_mapping.json'):
        print("索引文件创建成功!")
    else:
        print("索引文件创建失败!")

if __name__ == "__main__":
    main()
