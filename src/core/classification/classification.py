import faiss
import numpy as np
import json
import os
import hashlib

# 使用全局日志系统
from src.core.logging.global_logger import get_logger, log_system, log_error
logger = get_logger("classification")

class Classification:
    # 全局索引缓存
    _index_cache = {}
    # 结果缓存
    _result_cache = {}
    # 缓存大小限制 - 减少缓存大小以降低内存使用
    _cache_size = 100
    # 索引缓存大小限制
    _index_cache_size = 2
    
    def __init__(self, index_path=None, threshold=0.4):
        """初始化分类模块"""
        self.threshold = threshold
        self.index = None
        self.role_mapping = []  # 存储向量索引到角色名称的映射
        
        logger.info(f"初始化分类模块，阈值: {threshold}, index_path: {index_path}")
        
        # 如果提供了索引路径，加载索引
        if index_path:
            # 检查是否是绝对路径
            if os.path.isabs(index_path):
                # 如果是绝对路径，直接使用
                faiss_path = index_path
                logger.info(f"使用绝对路径: {faiss_path}")
            else:
                # 如果是相对路径，构建绝对路径
                # 获取项目根目录
                current_file = os.path.abspath(__file__)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
                # 再次向上一级，因为我们需要的是项目根目录，不是src目录
                project_root = os.path.dirname(project_root)
                logger.info(f"项目根目录: {project_root}")
                
                # 构建绝对路径
                faiss_path = os.path.join(project_root, index_path)
                logger.info(f"构建绝对路径: {faiss_path}")
            
            mapping_path = faiss_path.replace(".faiss", "_mapping.json")
            
            logger.info(f"检查索引文件是否存在: {faiss_path} -> {os.path.exists(faiss_path)}")
            logger.info(f"检查映射文件是否存在: {mapping_path} -> {os.path.exists(mapping_path)}")
            
            # 检查文件是否存在，处理.faiss后缀
            if not faiss_path.endswith(".faiss"):
                faiss_path_with_ext = f"{faiss_path}.faiss"
            else:
                faiss_path_with_ext = faiss_path
            
            if not mapping_path.endswith("_mapping.json"):
                mapping_path_with_ext = f"{mapping_path}_mapping.json"
            else:
                mapping_path_with_ext = mapping_path
            
            logger.info(f"检查索引文件是否存在: {faiss_path_with_ext} -> {os.path.exists(faiss_path_with_ext)}")
            logger.info(f"检查映射文件是否存在: {mapping_path_with_ext} -> {os.path.exists(mapping_path_with_ext)}")
            
            if os.path.exists(faiss_path_with_ext) and os.path.exists(mapping_path_with_ext):
                logger.info(f"加载索引: {faiss_path_with_ext}")
                self.load_index(faiss_path_with_ext)
            else:
                logger.error(f"索引文件或映射文件不存在")
    
    def build_index(self, features, role_names):
        """构建向量索引"""
        # 获取特征维度
        dim = features.shape[1]
        
        logger.info(f"开始构建索引，特征维度: {dim}, 特征数量: {features.shape[0]}, 角色数量: {len(role_names)}")
        
        # 创建Faiss索引（使用余弦相似度）
        self.index = faiss.IndexFlatIP(dim)
        
        # 添加特征向量到索引
        self.index.add(features)
        
        # 存储角色名称映射
        self.role_mapping = role_names
        
        logger.info(f"索引构建完成，包含 {len(role_names)} 个角色")
    
    def save_index(self, index_path):
        """保存索引到文件"""
        if self.index is None:
            raise ValueError("索引尚未构建")
        
        logger.info(f"开始保存索引到: {index_path}")
        
        # 保存Faiss索引
        faiss.write_index(self.index, f"{index_path}.faiss")
        
        # 保存角色映射
        with open(f"{index_path}_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.role_mapping, f, ensure_ascii=False, indent=2)
        
        logger.info(f"索引保存完成")
    
    def load_index(self, index_path):
        """从文件加载索引"""
        # 检查缓存中是否已有该索引
        if index_path in self.__class__._index_cache:
            logger.info(f"从缓存加载索引: {index_path}")
            cached_index, cached_mapping = self.__class__._index_cache[index_path]
            self.index = cached_index
            self.role_mapping = cached_mapping
            logger.info(f"缓存加载完成，角色数量: {len(cached_mapping)}")
            return
        
        # 检查索引缓存大小
        if len(self.__class__._index_cache) >= self.__class__._index_cache_size:
            # 移除最早的缓存项
            oldest_key = next(iter(self.__class__._index_cache))
            del self.__class__._index_cache[oldest_key]
            logger.info(f"索引缓存已满，移除最早的索引: {oldest_key}")
        
        logger.info(f"开始加载索引: {index_path}")

        # 确保路径有 .faiss 扩展名
        faiss_path = index_path if index_path.endswith('.faiss') else f"{index_path}.faiss"
        mapping_path = f"{index_path}_mapping.json" if not index_path.endswith('_mapping.json') else index_path

        # 加载Faiss索引
        self.index = faiss.read_index(faiss_path)

        # 加载角色映射
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.role_mapping = json.load(f)
        
        # 缓存索引和映射
        self.__class__._index_cache[faiss_path] = (self.index, self.role_mapping)

        logger.info(f"索引加载完成，角色数量: {len(self.role_mapping)}")
    
    def _get_feature_hash(self, feature):
        """计算特征向量的哈希值作为缓存键"""
        # 确保特征向量是一维的
        if len(feature.shape) > 1:
            feature = feature.flatten()
        # 转换为字节并计算哈希
        feature_bytes = feature.tobytes()
        return hashlib.md5(feature_bytes).hexdigest()
    
    def _cache_result(self, feature, result):
        """缓存分类结果"""
        # 计算缓存键
        cache_key = self._get_feature_hash(feature)
        # 检查缓存大小
        if len(self.__class__._result_cache) >= self.__class__._cache_size:
            # 移除最早的缓存项
            oldest_key = next(iter(self.__class__._result_cache))
            del self.__class__._result_cache[oldest_key]
        # 添加到缓存
        self.__class__._result_cache[cache_key] = result
    
    def _get_cached_result(self, feature):
        """获取缓存的分类结果"""
        cache_key = self._get_feature_hash(feature)
        return self.__class__._result_cache.get(cache_key)
    
    def classify(self, feature, top_k=5, tags=None):
        """分类单个特征向量
        
        Args:
            feature: 特征向量
            top_k: 返回前k个最相似的结果
            tags: 图片标签，用于辅助分类
            
        Returns:
            (角色名称, 相似度)
        """
        if self.index is None:
            logger.error("索引尚未构建")
            raise ValueError("索引尚未构建")
        
        # 尝试从缓存获取结果
        cached_result = self._get_cached_result(feature)
        if cached_result:
            logger.debug("从缓存获取分类结果")
            return cached_result
        
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
        
        # 如果没有结果，返回unknown
        if not results:
            result = ("unknown", 0.0)
            self._cache_result(feature, result)
            return result
        
        # 1. 计算每个角色的平均相似度
        role_similarities = {}
        role_counts = {}
        
        for result in results:
            role = result["role"]
            similarity = result["similarity"]
            
            if role not in role_similarities:
                role_similarities[role] = 0
                role_counts[role] = 0
            
            role_similarities[role] += similarity
            role_counts[role] += 1
        
        # 计算平均相似度
        role_avg_similarities = {}
        for role, total_similarity in role_similarities.items():
            role_avg_similarities[role] = total_similarity / role_counts[role]
        
        # 2. 找出平均相似度最高的角色
        sorted_roles = sorted(role_avg_similarities.items(), key=lambda x: x[1], reverse=True)
        
        best_role = sorted_roles[0][0]
        best_similarity = sorted_roles[0][1]
        
        # 3. 利用标签信息辅助分类
        if tags:
            # 转换标签为小写（处理字典格式的标签）
            tags_lower = []
            for tag_item in tags:
                if isinstance(tag_item, dict) and 'tag' in tag_item:
                    tags_lower.append(tag_item['tag'].lower())
                elif isinstance(tag_item, str):
                    tags_lower.append(tag_item.lower())
            
            # 检查标签中是否包含角色相关信息
            for role, similarity in sorted_roles:
                # 拼音到中文的映射
                pinyin_to_chinese = {
                    "ri4nai4": "日奈",
                    "a1luo2na4": "阿罗娜"
                }
                
                # 获取中文角色名
                chinese_role = pinyin_to_chinese.get(role, role)
                
                # 检查日奈
                if (role == "日奈" or role == "ri4nai4") and any('hina' in tag or '日奈' in tag for tag in tags_lower):
                    result = ("日奈", similarity)
                    self._cache_result(feature, result)
                    return result
                # 检查伊织
                elif role == "伊织" and any('izumi' in tag or '伊织' in tag for tag in tags_lower):
                    result = ("伊织", similarity)
                    self._cache_result(feature, result)
                    return result
                # 检查阿罗娜
                elif (role == "阿罗娜" or role == "a1luo2na4") and any('arona' in tag or '阿罗娜' in tag for tag in tags_lower):
                    result = ("阿罗娜", similarity)
                    self._cache_result(feature, result)
                    return result
                # 检查普拉娜
                elif role == "普拉娜" and any('prana' in tag or '普拉娜' in tag for tag in tags_lower):
                    result = ("普拉娜", similarity)
                    self._cache_result(feature, result)
                    return result
        
        # 4. 调整阈值，使其更倾向于选择平均相似度高的角色
        threshold = self.threshold  # 使用构造函数中设置的阈值
        
        # 5. 拼音到中文的映射
        pinyin_to_chinese = {
            "ri4nai4": "日奈",
            "a1luo2na4": "阿罗娜"
        }
        
        # 转换拼音为中文
        if best_role in pinyin_to_chinese:
            best_role = pinyin_to_chinese[best_role]
        
        # 总是返回最相似的角色，即使相似度低于阈值
        result = (best_role, best_similarity)
        self._cache_result(feature, result)
        return result
    
    def batch_classify(self, features, top_k=5):
        """批量分类特征向量
        
        Args:
            features: 特征向量批次
            top_k: 返回前k个最相似的结果
            
        Returns:
            分类结果列表 [(角色名称, 相似度)]
        """
        if self.index is None:
            raise ValueError("索引尚未构建")
        
        logger.info(f"开始批量分类，特征数量: {features.shape[0]}, top_k: {top_k}")
        
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
            
            # 如果没有结果，返回unknown
            if not results:
                batch_results.append(("unknown", 0.0))
                continue
            
            # 1. 动态阈值调整
            # 计算相似度的平均值和标准差
            similarities = [r["similarity"] for r in results]
            avg_similarity = sum(similarities) / len(similarities)
            std_similarity = (sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)) ** 0.5
            
            # 根据相似度分布动态调整阈值
            dynamic_threshold = max(self.threshold - 0.1, min(self.threshold + 0.1, avg_similarity - std_similarity * 0.5))
            
            # 2. Top-k投票机制
            role_counts = {}
            role_scores = {}
            
            for j, result in enumerate(results):
                role = result["role"]
                similarity = result["similarity"]
                
                # 权重随排名递减
                weight = (top_k - j) / top_k
                
                if role not in role_counts:
                    role_counts[role] = 0
                    role_scores[role] = 0
                
                # 只有相似度高于动态阈值的结果才参与投票
                if similarity >= dynamic_threshold:
                    role_counts[role] += 1
                    role_scores[role] += similarity * weight
            
            # 找出得票最多的角色
            if role_counts:
                # 首先按票数排序，票数相同则按得分排序
                sorted_roles = sorted(role_counts.items(), key=lambda x: (x[1], role_scores[x[0]]), reverse=True)
                best_role = sorted_roles[0][0]
                # 避免除以零
                if role_counts[best_role] > 0:
                    best_similarity = role_scores[best_role] / role_counts[best_role]  # 平均加权相似度
                else:
                    best_similarity = 0.0
                
                # 检查最佳角色的相似度是否足够高
                if best_similarity >= max(self.threshold - 0.1, 0.5):
                    batch_results.append((best_role, best_similarity))
                    continue
            
            # 3. 如果投票机制失败，回退到原始的top-1结果
            if results[0]["similarity"] >= self.threshold - 0.1:
                batch_results.append((results[0]["role"], results[0]["similarity"]))
            else:
                batch_results.append(("unknown", results[0]["similarity"] if results else 0.0))
        
        logger.info(f"批量分类完成，处理了 {len(batch_results)} 个特征向量")
        return batch_results
    
    def classify_multiple_characters(self, characters):
        """分类多个角色"""
        if self.index is None:
            raise ValueError("索引尚未构建")
        
        logger.info(f"开始多角色分类，角色数量: {len(characters)}")
        
        try:
            # 提取所有角色的特征
            features = np.array([char['feature'] for char in characters], dtype=np.float32)
            
            # 批量分类
            results = self.batch_classify(features)
            
            # 将分类结果与角色信息关联
            for i, char in enumerate(characters):
                char['role'], char['similarity'] = results[i]
            
            logger.info(f"多角色分类完成")
            return characters
        except Exception as e:
            logger.error(f"多角色分类失败: {e}")
            return []
    
    def update_index(self, new_features, new_role_names):
        """更新索引，添加新的特征向量和角色"""
        if self.index is None:
            # 如果索引不存在，构建新索引
            logger.info(f"索引不存在，构建新索引，特征数量: {new_features.shape[0]}")
            self.build_index(new_features, new_role_names)
        else:
            logger.info(f"开始更新索引，添加特征数量: {new_features.shape[0]}, 角色数量: {len(new_role_names)}")
            # 添加新的特征向量
            self.index.add(new_features)
            
            # 添加新的角色名称映射
            self.role_mapping.extend(new_role_names)
            
            logger.info(f"索引更新完成，当前角色数量: {len(self.role_mapping)}")
    
    def incremental_learning(self, image_path, correct_role):
        """增量学习：根据用户提供的正确角色更新索引"""
        from src.core.preprocessing.preprocessing import Preprocessing
        from src.core.feature_extraction.feature_extraction import FeatureExtraction
        
        logger.info(f"开始增量学习，图像路径: {image_path}, 正确角色: {correct_role}")
        
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
            
            logger.info(f"增量学习成功: 图像 {image_path} 已添加到角色 {correct_role} 的索引中")
            return True
        except Exception as e:
            logger.error(f"增量学习失败: {e}")
            return False
    
    @classmethod
    def use_coreml(cls, config_path="./coreml_models/end_to_end_config.json"):
        """使用Core ML分类器
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Core ML分类器实例
        """
        try:
            # 动态导入Core ML分类器
            from .coreml_classification import CoreMLClassification
            logger.info("使用Core ML分类器")
            return CoreMLClassification(config_path)
        except ImportError as e:
            logger.warning(f"Core ML分类器导入失败: {e}")
            logger.info("回退到默认分类器")
            return cls()

if __name__ == "__main__":
    # 测试分类模块
    classifier = Classification(threshold=0.7)
    
    # 生成测试数据
    # 假设有2个角色，每个角色有5个特征向量
    dim = 512  # CLIP ViT-B/32特征维度
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
