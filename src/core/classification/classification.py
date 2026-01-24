import faiss
import numpy as np
import json
import os

class Classification:
    # 全局索引缓存
    _index_cache = {}
    
    def __init__(self, index_path=None, threshold=0.7):
        """初始化分类模块"""
        self.threshold = threshold
        self.index = None
        self.role_mapping = []  # 存储向量索引到角色名称的映射
        
        # 如果提供了索引路径，加载索引
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
    
    def build_index(self, features, role_names):
        """构建向量索引"""
        # 获取特征维度
        dim = features.shape[1]
        
        # 创建Faiss索引（使用余弦相似度）
        self.index = faiss.IndexFlatIP(dim)
        
        # 添加特征向量到索引
        self.index.add(features)
        
        # 存储角色名称映射
        self.role_mapping = role_names
    
    def save_index(self, index_path):
        """保存索引到文件"""
        if self.index is None:
            raise ValueError("索引尚未构建")
        
        # 保存Faiss索引
        faiss.write_index(self.index, f"{index_path}.faiss")
        
        # 保存角色映射
        with open(f"{index_path}_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.role_mapping, f, ensure_ascii=False, indent=2)
    
    def load_index(self, index_path):
        """从文件加载索引"""
        # 检查缓存中是否已有该索引
        if index_path in self.__class__._index_cache:
            cached_index, cached_mapping = self.__class__._index_cache[index_path]
            self.index = cached_index
            self.role_mapping = cached_mapping
            return
        
        # 加载Faiss索引
        self.index = faiss.read_index(f"{index_path}.faiss")
        
        # 加载角色映射
        with open(f"{index_path}_mapping.json", "r", encoding="utf-8") as f:
            self.role_mapping = json.load(f)
        
        # 缓存索引和映射
        self.__class__._index_cache[index_path] = (self.index, self.role_mapping)
    
    def classify(self, feature, top_k=5):
        """分类单个特征向量"""
        if self.index is None:
            raise ValueError("索引尚未构建")
        
        # 确保特征向量是二维的
        if len(feature.shape) == 1:
            feature = feature.reshape(1, -1)
        
        # 搜索最相似的向量
        distances, indices = self.index.search(feature, top_k)
        
        # 处理结果
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            distance = distances[0][i]
            
            # 确保索引在有效范围内
            if idx < len(self.role_mapping):
                role_name = self.role_mapping[idx]
                results.append({
                    "role": role_name,
                    "similarity": float(distance)
                })
        
        # 根据相似度阈值判断是否为已知角色
        if results and results[0]["similarity"] >= self.threshold:
            return results[0]["role"], results[0]["similarity"]
        else:
            return "unknown", results[0]["similarity"] if results else 0.0
    
    def batch_classify(self, features, top_k=5):
        """批量分类特征向量"""
        if self.index is None:
            raise ValueError("索引尚未构建")
        
        # 搜索最相似的向量
        distances, indices = self.index.search(features, top_k)
        
        # 处理结果
        batch_results = []
        for i in range(features.shape[0]):
            results = []
            for j in range(top_k):
                idx = indices[i][j]
                distance = distances[i][j]
                
                # 确保索引在有效范围内
                if idx < len(self.role_mapping):
                    role_name = self.role_mapping[idx]
                    results.append({
                        "role": role_name,
                        "similarity": float(distance)
                    })
            
            # 根据相似度阈值判断是否为已知角色
            if results and results[0]["similarity"] >= self.threshold:
                batch_results.append((results[0]["role"], results[0]["similarity"]))
            else:
                batch_results.append(("unknown", results[0]["similarity"] if results else 0.0))
        
        return batch_results
    
    def classify_multiple_characters(self, characters):
        """分类多个角色"""
        if self.index is None:
            raise ValueError("索引尚未构建")
        
        try:
            # 提取所有角色的特征
            features = np.array([char['feature'] for char in characters], dtype=np.float32)
            
            # 批量分类
            results = self.batch_classify(features)
            
            # 将分类结果与角色信息关联
            for i, char in enumerate(characters):
                char['role'], char['similarity'] = results[i]
            
            return characters
        except Exception as e:
            print(f"多角色分类失败: {e}")
            return []
    
    def update_index(self, new_features, new_role_names):
        """更新索引，添加新的特征向量和角色"""
        if self.index is None:
            # 如果索引不存在，构建新索引
            self.build_index(new_features, new_role_names)
        else:
            # 添加新的特征向量
            self.index.add(new_features)
            
            # 添加新的角色名称映射
            self.role_mapping.extend(new_role_names)
    
    def incremental_learning(self, image_path, correct_role):
        """增量学习：根据用户提供的正确角色更新索引"""
        from src.core.preprocessing.preprocessing import Preprocessing
        from src.core.feature_extraction.feature_extraction import FeatureExtraction
        
        # 初始化预处理和特征提取模块
        preprocessor = Preprocessing()
        extractor = FeatureExtraction()
        
        try:
            # 预处理图像
            normalized_img, _ = preprocessor.process(image_path)
            
            # 提取特征
            feature = extractor.extract_features(normalized_img)
            feature = feature.reshape(1, -1)
            
            # 更新索引
            self.update_index(feature, [correct_role])
            
            print(f"增量学习成功: 图像 {image_path} 已添加到角色 {correct_role} 的索引中")
            return True
        except Exception as e:
            print(f"增量学习失败: {e}")
            return False

if __name__ == "__main__":
    # 测试分类模块
    classifier = Classification(threshold=0.7)
    
    # 生成测试数据
    # 假设有2个角色，每个角色有5个特征向量
    dim = 768  # CLIP特征维度
    num_roles = 2
    num_samples_per_role = 5
    
    # 生成随机特征向量（实际应用中应该使用真实的特征向量）
    features = np.random.randn(num_roles * num_samples_per_role, dim).astype(np.float32)
    # 归一化特征向量
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # 生成角色名称映射
    role_names = []
    for i in range(num_roles):
        role_name = f"角色{i+1}"
        role_names.extend([role_name] * num_samples_per_role)
    
    # 构建索引
    classifier.build_index(features, role_names)
    
    # 保存索引
    index_path = "test_index"
    classifier.save_index(index_path)
    print("索引已保存")
    
    # 加载索引
    new_classifier = Classification()
    new_classifier.load_index(index_path)
    print("索引已加载")
    
    # 测试分类
    test_feature = np.random.randn(dim).astype(np.float32)
    test_feature = test_feature / np.linalg.norm(test_feature)
    
    role, similarity = new_classifier.classify(test_feature)
    print(f"分类结果: 角色={role}, 相似度={similarity:.4f}")
