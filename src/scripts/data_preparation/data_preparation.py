import os
import numpy as np
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification
from PIL import Image

class DataPreparation:
    def __init__(self, data_dir="dataset", index_path="role_index"):
        """初始化数据准备模块"""
        self.data_dir = data_dir
        self.index_path = index_path
        self.preprocessor = Preprocessing()
        self.extractor = FeatureExtraction()
        self.classifier = Classification()
    
    def organize_dataset(self):
        """组织数据集目录结构"""
        # 检查数据集目录是否存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"创建数据集目录: {self.data_dir}")
        
        # 建议的目录结构
        print("建议的数据集目录结构:")
        print(f"{self.data_dir}/")
        print("  角色1/")
        print("    1.jpg")
        print("    2.jpg")
        print("    ...")
        print("  角色2/")
        print("    1.jpg")
        print("    2.jpg")
        print("    ...")
        print("  ...")
    
    def build_index_from_dataset(self):
        """从数据集构建向量索引"""
        # 检查数据集目录是否存在
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据集目录不存在: {self.data_dir}")
        
        # 获取角色目录列表
        role_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        if not role_dirs:
            raise ValueError(f"数据集目录中没有角色子目录: {self.data_dir}")
        
        print(f"发现 {len(role_dirs)} 个角色目录")
        
        # 提取所有角色的特征向量
        all_features = []
        all_role_names = []
        
        for role_name in role_dirs:
            role_dir = os.path.join(self.data_dir, role_name)
            image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"处理角色 '{role_name}'，发现 {len(image_files)} 张图片")
            
            for img_file in image_files:
                img_path = os.path.join(role_dir, img_file)
                
                try:
                    # 预处理图像
                    normalized_img, _ = self.preprocessor.process(img_path)
                    
                    # 提取特征
                    feature = self.extractor.extract_features(normalized_img)
                    
                    # 添加到列表
                    all_features.append(feature)
                    all_role_names.append(role_name)
                    
                except Exception as e:
                    print(f"处理图像 {img_path} 失败: {e}")
                    continue
        
        if not all_features:
            raise ValueError("没有成功提取任何特征向量")
        
        # 转换为numpy数组
        features_array = np.array(all_features, dtype=np.float32)
        
        print(f"成功提取 {len(features_array)} 个特征向量")
        
        # 构建索引
        self.classifier.build_index(features_array, all_role_names)
        
        # 保存索引
        self.classifier.save_index(self.index_path)
        print(f"索引已保存到 {self.index_path}.faiss 和 {self.index_path}_mapping.json")
        
        return features_array, all_role_names
    
    def add_new_role(self, role_name, image_dir):
        """添加新角色到数据集和索引"""
        # 检查角色目录是否存在，不存在则创建
        role_dir = os.path.join(self.data_dir, role_name)
        if not os.path.exists(role_dir):
            os.makedirs(role_dir)
            print(f"创建角色目录: {role_dir}")
        
        # 复制图像到角色目录（实际应用中可能需要实现复制逻辑）
        print(f"请将 {role_name} 的图片放入目录: {role_dir}")
        print("然后运行 build_index_from_dataset() 重新构建索引")
    
    def validate_dataset(self):
        """验证数据集的质量"""
        # 检查数据集目录是否存在
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据集目录不存在: {self.data_dir}")
        
        # 获取角色目录列表
        role_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        if not role_dirs:
            raise ValueError(f"数据集目录中没有角色子目录: {self.data_dir}")
        
        print("数据集验证结果:")
        print(f"总角色数: {len(role_dirs)}")
        
        total_images = 0
        for role_name in role_dirs:
            role_dir = os.path.join(self.data_dir, role_name)
            image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            num_images = len(image_files)
            total_images += num_images
            
            print(f"角色 '{role_name}': {num_images} 张图片")
        
        print(f"总图片数: {total_images}")
        
        # 检查每个角色的图片数量是否合理
        for role_name in role_dirs:
            role_dir = os.path.join(self.data_dir, role_name)
            image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            num_images = len(image_files)
            
            if num_images < 20:
                print(f"警告: 角色 '{role_name}' 的图片数量较少 ({num_images} 张)，建议至少收集 20 张不同角度的照片")
            elif num_images > 50:
                print(f"提示: 角色 '{role_name}' 的图片数量较多 ({num_images} 张)，可以考虑筛选质量较好的图片")

if __name__ == "__main__":
    # 测试数据准备模块
    data_prep = DataPreparation()
    
    # 组织数据集目录结构
    data_prep.organize_dataset()
    
    # 验证数据集
    try:
        data_prep.validate_dataset()
    except Exception as e:
        print(f"数据集验证失败: {e}")
    
    # 构建索引（实际应用中需要先准备好数据集）
    try:
        features, role_names = data_prep.build_index_from_dataset()
        print("索引构建成功!")
    except Exception as e:
        print(f"索引构建失败: {e}")
