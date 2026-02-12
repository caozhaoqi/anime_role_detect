#!/usr/bin/env python3
"""
通用分类模块
封装核心分类逻辑，提供简洁的API接口
支持网页应用和其他脚本调用
"""
import os
import sys
import numpy as np
from PIL import Image

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 计算项目根目录的绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 使用全局日志系统
from src.core.logging.global_logger import get_logger, log_system, log_inference, log_error
logger = get_logger("general_classification")

# 导入日志记录模块
try:
    from src.core.log_fusion.log_recorder import record_classification_log
except Exception as e:
    logger.warning(f"导入日志记录模块失败: {e}")
    record_classification_log = None


class GeneralClassification:
    """通用分类器类"""
    
    # 全局缓存
    _instance_cache = None
    
    def __new__(cls, threshold=0.7, index_path=None, model=None):
        """单例模式，避免重复初始化"""
        if cls._instance_cache is None:
            cls._instance_cache = super(GeneralClassification, cls).__new__(cls)
        return cls._instance_cache
    
    def __init__(self, threshold=0.7, index_path=None, model=None):
        """初始化通用分类器"""
        if hasattr(self, 'is_initialized') and self.is_initialized:
            return
        
        self.threshold = threshold
        self.index_path = index_path
        self.model = model  # 新增：模型参数
        self.extractor = None
        self.classifier = None
        self.preprocessor = None
        self.model_inference = None # 新增：模型推理器
        self.is_initialized = False
        
    def initialize(self):
        """初始化所有模块"""
        if self.is_initialized:
            return True
        
        try:
            # 1. 配置管理
            import threading
            import time
            
            # 延迟导入，避免启动时加载所有模块
            from src.core.preprocessing.preprocessing import Preprocessing
            from src.core.feature_extraction.feature_extraction import FeatureExtraction
            from src.core.classification.classification import Classification
            
            # 初始化状态跟踪
            init_results = {}
            lock = threading.Lock()
            
            # 2. 并行初始化函数
            def init_module(name, init_func):
                try:
                    logger.info(f"初始化{name}模块...")
                    start_time = time.time()
                    result = init_func()
                    elapsed = time.time() - start_time
                    logger.info(f"{name}模块初始化完成，耗时: {elapsed:.2f}秒")
                    with lock:
                        init_results[name] = (True, result)
                except Exception as e:
                    error_msg = f"{name}模块初始化失败: {e}"
                    logger.error(error_msg)
                    with lock:
                        init_results[name] = (False, error_msg)
            
            # 3. 创建初始化线程
            threads = []
            
            # 预处理模块初始化
            t1 = threading.Thread(target=init_module, args=("预处理", lambda: Preprocessing()))
            threads.append(t1)
            
            # 特征提取模块初始化
            t2 = threading.Thread(target=init_module, args=("特征提取", lambda: FeatureExtraction()))
            threads.append(t2)
            
            # 分类模块初始化
            t3 = threading.Thread(target=init_module, args=("分类", lambda: Classification(threshold=self.threshold)))
            threads.append(t3)
            
            # 启动所有线程
            for t in threads:
                t.start()
            
            # 等待所有线程完成
            for t in threads:
                t.join()
            
            # 4. 处理初始化结果
            if init_results.get("预处理", (False,))[0]:
                self.preprocessor = init_results["预处理"][1]
            else:
                logger.warning("预处理模块初始化失败，使用默认实现")
                self.preprocessor = Preprocessing()
            
            if init_results.get("特征提取", (False,))[0]:
                self.extractor = init_results["特征提取"][1]
            else:
                logger.warning("特征提取模块初始化失败，使用默认实现")
                self.extractor = FeatureExtraction()
            
            if init_results.get("分类", (False,))[0]:
                self.classifier = init_results["分类"][1]
            else:
                logger.warning("分类模块初始化失败，使用默认实现")
                self.classifier = Classification(threshold=self.threshold)
            
            # 5. 尝试初始化EfficientNet推理器（单独处理，非致命）
            try:
                logger.info("初始化EfficientNet推理模型...")
                from src.core.classification.efficientnet_inference import EfficientNetInference
                # 如果指定了模型，则构建模型路径
                model_path = None
                if self.model:
                    # 构建模型路径
                    base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
                    model_path = os.path.join(base_model_dir, self.model, 'model_best.pth')
                    logger.info(f"使用指定模型: {model_path}")
                self.model_inference = EfficientNetInference(model_path=model_path)
                logger.info("EfficientNet推理模型初始化成功")
            except Exception as e:
                logger.warning(f"EfficientNet模型初始化失败 (非致命): {e}")
                self.model_inference = None
            
            # 6. 尝试初始化DeepDanbooru推理器（单独处理，非致命）
            try:
                logger.info("初始化DeepDanbooru推理模型...")
                from src.core.classification.deepdanbooru_inference import DeepDanbooruInference
                self.deepdanbooru_inference = DeepDanbooruInference()
                logger.info("DeepDanbooru推理模型初始化成功")
            except Exception as e:
                logger.warning(f"DeepDanbooru模型初始化失败 (非致命): {e}")
                self.deepdanbooru_inference = None
            
            # 7. 如果指定了索引路径，尝试加载
            if self.index_path:
                index_files = [
                    f"{self.index_path}.faiss",
                    f"{self.index_path}_mapping.json"
                ]
                if all(os.path.exists(f) for f in index_files):
                    logger.info(f"加载索引文件: {self.index_path}")
                    try:
                        self.classifier.load_index(self.index_path)
                        logger.info(f"索引加载成功，包含 {len(self.classifier.role_mapping)} 个角色")
                    except Exception as e:
                        logger.error(f"索引加载失败: {e}")
                else:
                    logger.warning(f"索引文件不存在: {self.index_path}")
            
            # 7. 验证初始化状态
            required_modules = [self.preprocessor, self.extractor, self.classifier]
            if all(module is not None for module in required_modules):
                self.is_initialized = True
                logger.info("所有核心模块初始化成功!")
                return True
            else:
                logger.error("核心模块初始化不完整")
                self.is_initialized = False
                return False
                
        except Exception as e:
            logger.error(f"模块初始化失败: {e}")
            self.is_initialized = False
            return False
    
    def build_index_from_directory(self, data_dir):
        """从目录构建特征索引"""
        if not self.initialize():
            return False
        
        try:
            logger.info(f"从目录构建索引: {data_dir}")
            
            # 收集所有角色目录，过滤掉非角色目录
            exclude_dirs = ['shuffled', 'classification_results', 'downloaded', '.DS_Store']
            role_dirs = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d)) and d not in exclude_dirs]
            
            if not role_dirs:
                logger.error("目录中没有角色子目录")
                return False
            
            logger.info(f"找到 {len(role_dirs)} 个角色目录")
            
            # 为每个角色提取特征
            features = []
            role_names = []
            
            for role in role_dirs:
                role_dir = os.path.join(data_dir, role)
                
                # 获取角色目录中的图片
                image_files = [f for f in os.listdir(role_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                if not image_files:
                    logger.warning(f"角色 {role} 目录中没有图片")
                    continue
                
                logger.info(f"处理角色: {role}, 共 {len(image_files)} 张图片")
                
                # 遍历该角色下的所有图片，提取特征
                for img_file in image_files:
                    image_path = os.path.join(role_dir, img_file)
                    try:
                        # 预处理
                        normalized_img, _ = self.preprocessor.process(image_path)
                        
                        # 提取特征
                        feature = self.extractor.extract_features(normalized_img)
                        features.append(feature)
                        role_names.append(role)
                        
                    except Exception as e:
                        logger.error(f"处理图片 {img_file} 失败: {e}")
                        continue
                
                logger.info(f"完成角色 {role} 的特征提取")
            
            if not features:
                logger.error("无法提取任何特征")
                return False
            
            # 构建索引
            features_np = np.array(features).astype(np.float32)
            self.classifier.build_index(features_np, role_names)
            
            logger.info(f"索引构建成功，包含 {len(features)} 个向量，覆盖 {len(set(role_names))} 个角色")
            return True
        except Exception as e:
            logger.error(f"构建索引失败: {e}")
            return False
    
    def classify_image(self, image_path, use_model=False):
        """分类单个图像
        :param image_path: 图片路径
        :param use_model: 是否使用EfficientNet模型进行推理
        """
        if not self.initialize():
            logger.error("分类器未初始化，无法进行分类")
            return None, 0.0, None
            
        try:
            logger.info(f"开始分类图像: {image_path}, use_model={use_model}")
            # 预处理 (YOLO检测)
            # 无论使用哪种方法，先用YOLO裁剪出人物主体总是好的
            normalized_img, boxes = self.preprocessor.process(image_path)
            
            # 如果指定使用模型且模型已加载
            if use_model and self.model_inference:
                # 注意：EfficientNetInference 内部会再次读取图片并进行自己的预处理
                # 这里我们传入原始路径，或者我们可以优化让它接受 PIL Image
                # 目前为了简单，直接传路径
                logger.info("使用 EfficientNet 模型进行推理")
                # 使用标签辅助推理
                role, similarity, _ = self.model_inference.predict_with_tags(image_path)
                feature = None # 模型推理不产生CLIP特征向量
            else:
                # 使用默认的 CLIP + FAISS 索引
                if self.classifier.index is None:
                    raise ValueError("索引尚未构建或加载。请先运行 scripts/build_faiss_index.py 构建索引。")
                
                # 提取特征
                logger.info("使用 CLIP 模型提取特征")
                feature = self.extractor.extract_features(normalized_img)
                
                # 分类
                logger.info("使用 FAISS 索引进行分类")
                role, similarity = self.classifier.classify(feature)
            
            # 记录分类日志
            if record_classification_log is not None:
                record_classification_log(
                    image_path=image_path,
                    role=role,
                    similarity=similarity,
                    feature=feature if feature is not None else [],
                    boxes=boxes
                )
            
            logger.info(f"分类完成，角色: {role}, 相似度: {similarity:.4f}")
            return role, similarity, boxes
        except Exception as e:
            logger.error(f"分类图像失败: {e}")
            raise e # 抛出异常以便上层捕获
    
    def classify_image_ensemble(self, image_path, clip_weight=0.7, model_weight=0.3, confidence_threshold=0.6):
        """使用集成方法分类单个图像（整合两个模型的结果）
        :param image_path: 图片路径
        :param clip_weight: CLIP模型的权重
        :param model_weight: MobileNetV2模型的权重
        :param confidence_threshold: 置信度阈值
        """
        if not self.initialize():
            return None, 0.0, None
            
        try:
            # 1. 使用CLIP模型分类
            clip_role, clip_similarity, boxes = self.classify_image(image_path, use_model=False)
            
            # 2. 使用MobileNetV2模型分类
            model_role, model_similarity, _ = self.classify_image(image_path, use_model=True)
            
            # 3. 整合两个模型的结果
            # 初始化结果字典
            results = {}
            
            # 添加CLIP模型的结果
            if clip_role and clip_similarity >= confidence_threshold:
                weighted_score = clip_similarity * clip_weight
                results[clip_role] = results.get(clip_role, 0) + weighted_score
            
            # 添加MobileNetV2模型的结果
            if model_role and model_similarity >= confidence_threshold:
                weighted_score = model_similarity * model_weight
                results[model_role] = results.get(model_role, 0) + weighted_score
            
            # 4. 选择最终结果
            if results:
                # 按加权分数排序
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                final_role, final_score = sorted_results[0]
                
                # 计算标准化的最终相似度
                total_weight = clip_weight + model_weight
                normalized_score = final_score / total_weight
                
                # 5. 记录集成分类日志
                if record_classification_log is not None:
                    # 提取CLIP特征用于日志
                    normalized_img, _ = self.preprocessor.process(image_path)
                    feature = self.extractor.extract_features(normalized_img)
                    
                    record_classification_log(
                        image_path=image_path,
                        role=final_role,
                        similarity=normalized_score,
                        feature=feature if feature is not None else [],
                        boxes=boxes,
                        metadata={
                            "ensemble": True,
                            "clip_role": clip_role,
                            "clip_similarity": clip_similarity,
                            "model_role": model_role,
                            "model_similarity": model_similarity,
                            "weights": {"clip": clip_weight, "model": model_weight}
                        }
                    )
                
                return final_role, normalized_score, boxes
            else:
                # 如果两个模型的置信度都低于阈值，使用CLIP模型的结果
                if record_classification_log is not None:
                    normalized_img, _ = self.preprocessor.process(image_path)
                    feature = self.extractor.extract_features(normalized_img)
                    
                    record_classification_log(
                        image_path=image_path,
                        role=clip_role,
                        similarity=clip_similarity,
                        feature=feature if feature is not None else [],
                        boxes=boxes,
                        metadata={
                            "ensemble": True,
                            "fallback_to_clip": True,
                            "clip_similarity": clip_similarity,
                            "model_similarity": model_similarity
                        }
                    )
                
                return clip_role, clip_similarity, boxes
                
        except Exception as e:
            logger.error(f"集成分类失败: {e}")
            # 失败时回退到CLIP模型
            logger.warning("回退到CLIP模型进行分类")
            return self.classify_image(image_path, use_model=False)
    
    def classify_image_with_deepdanbooru(self, image_path, clip_weight=0.4, model_weight=0.4, deepdanbooru_weight=0.2, confidence_threshold=0.6):
        """使用集成方法分类单个图像（整合CLIP、专用模型和DeepDanbooru的结果）
        :param image_path: 图片路径
        :param clip_weight: CLIP模型的权重
        :param model_weight: 专用模型的权重
        :param deepdanbooru_weight: DeepDanbooru模型的权重
        :param confidence_threshold: 置信度阈值
        """
        if not self.initialize():
            return None, 0.0, None
            
        try:
            logger.info(f"开始集成DeepDanbooru的分类: {image_path}")
            
            # 1. 使用CLIP模型分类
            clip_role, clip_similarity, boxes = self.classify_image(image_path, use_model=False)
            logger.info(f"CLIP模型结果: {clip_role}, 相似度: {clip_similarity:.4f}")
            
            # 2. 使用专用模型分类
            model_role, model_similarity, _ = self.classify_image(image_path, use_model=True)
            logger.info(f"专用模型结果: {model_role}, 相似度: {model_similarity:.4f}")
            
            # 3. 使用DeepDanbooru模型分类
            deepdanbooru_tags = []
            if self.deepdanbooru_inference:
                try:
                    deepdanbooru_tags = self.deepdanbooru_inference.predict(image_path)
                    logger.info(f"DeepDanbooru模型识别到 {len(deepdanbooru_tags)} 个标签")
                except Exception as e:
                    logger.error(f"DeepDanbooru推理失败: {e}")
            else:
                logger.warning("DeepDanbooru推理器未初始化")
            
            # 4. 整合三个模型的结果
            # 初始化结果字典
            results = {}
            
            # 添加CLIP模型的结果
            if clip_role and clip_similarity >= confidence_threshold:
                weighted_score = clip_similarity * clip_weight
                results[clip_role] = results.get(clip_role, 0) + weighted_score
            
            # 添加专用模型的结果
            if model_role and model_similarity >= confidence_threshold:
                weighted_score = model_similarity * model_weight
                results[model_role] = results.get(model_role, 0) + weighted_score
            
            # 添加DeepDanbooru模型的结果（如果有相关标签）
            # 这里需要将DeepDanbooru的标签映射到角色名称
            # 由于DeepDanbooru主要识别标签而非角色，我们需要一个映射表
            # 这里使用简单的关键词匹配作为示例
            if deepdanbooru_tags:
                # 简单的角色-标签映射表
                role_tag_mapping = {
                    'Blue Archive_Hoshino': ['hoshino', 'blue archive', 'hoshino (blue archive)'],
                    'Blue Archive_Shiroko': ['shiroko', 'blue archive', 'shiroko (blue archive)'],
                    'Blue Archive_Arona': ['arona', 'blue archive', 'arona (blue archive)'],
                    'Blue Archive_Miyako': ['miyako', 'blue archive', 'miyako (blue archive)'],
                    'Blue Archive_Hina': ['hina', 'blue archive', 'hina (blue archive)'],
                    'Blue Archive_Yuuka': ['yuuka', 'blue archive', 'yuuka (blue archive)'],
                    'BangDream_MyGo_Aimi Kanazawa': ['aimi kanazawa', 'kanazawa aimi', 'mygo', 'bang dream'],
                    'BangDream_MyGo_Ritsuki': ['ritsuki', 'mygo', 'bang dream'],
                    'BangDream_MyGo_Touko Takamatsu': ['touko takamatsu', 'takamatsu touko', 'mygo', 'bang dream'],
                    'BangDream_MyGo_Soyo Nagasaki': ['soyo nagasaki', 'nagasaki soyo', 'mygo', 'bang dream'],
                    'BangDream_MyGo_Sakiko Tamagawa': ['sakiko tamagawa', 'tamagawa sakiko', 'mygo', 'bang dream'],
                    'BangDream_MyGo_Mutsumi Wakaba': ['mutsumi wakaba', 'wakaba mutsumi', 'mygo', 'bang dream'],
                }
                
                # 提取DeepDanbooru标签
                deepdanbooru_tag_names = [tag['tag'] for tag in deepdanbooru_tags]
                
                # 匹配角色
                for role, tags in role_tag_mapping.items():
                    # 计算匹配分数
                    match_score = 0
                    for tag in tags:
                        if any(tag in deepdanbooru_tag for deepdanbooru_tag in deepdanbooru_tag_names):
                            match_score += 1
                    
                    # 如果有匹配，计算权重分数
                    if match_score > 0:
                        # 归一化匹配分数
                        normalized_score = match_score / len(tags)
                        # 找到最高置信度的匹配标签
                        max_score = 0
                        for tag in tags:
                            for deepdanbooru_tag in deepdanbooru_tags:
                                if tag in deepdanbooru_tag['tag']:
                                    max_score = max(max_score, deepdanbooru_tag['score'])
                        
                        # 综合分数
                        if max_score > 0:
                            combined_score = (normalized_score + max_score) / 2
                            weighted_score = combined_score * deepdanbooru_weight
                            results[role] = results.get(role, 0) + weighted_score
                            logger.info(f"DeepDanbooru匹配 {role}, 分数: {combined_score:.4f}")
            
            # 5. 选择最终结果
            if results:
                # 按加权分数排序
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                final_role, final_score = sorted_results[0]
                
                # 计算标准化的最终相似度
                total_weight = clip_weight + model_weight + deepdanbooru_weight
                normalized_score = final_score / total_weight
                
                logger.info(f"最终集成结果: {final_role}, 相似度: {normalized_score:.4f}")
                
                # 6. 记录集成分类日志
                if record_classification_log is not None:
                    # 提取CLIP特征用于日志
                    normalized_img, _ = self.preprocessor.process(image_path)
                    feature = self.extractor.extract_features(normalized_img)
                    
                    record_classification_log(
                        image_path=image_path,
                        role=final_role,
                        similarity=normalized_score,
                        feature=feature if feature is not None else [],
                        boxes=boxes,
                        metadata={
                            "ensemble": True,
                            "with_deepdanbooru": True,
                            "clip_role": clip_role,
                            "clip_similarity": clip_similarity,
                            "model_role": model_role,
                            "model_similarity": model_similarity,
                            "deepdanbooru_tags": [tag['tag'] for tag in deepdanbooru_tags[:5]],
                            "weights": {
                                "clip": clip_weight,
                                "model": model_weight,
                                "deepdanbooru": deepdanbooru_weight
                            }
                        }
                    )
                
                return final_role, normalized_score, boxes
            else:
                # 如果所有模型的置信度都低于阈值，使用CLIP模型的结果
                logger.warning("所有模型置信度都低于阈值，使用CLIP模型结果")
                if record_classification_log is not None:
                    normalized_img, _ = self.preprocessor.process(image_path)
                    feature = self.extractor.extract_features(normalized_img)
                    
                    record_classification_log(
                        image_path=image_path,
                        role=clip_role,
                        similarity=clip_similarity,
                        feature=feature if feature is not None else [],
                        boxes=boxes,
                        metadata={
                            "ensemble": True,
                            "with_deepdanbooru": True,
                            "fallback_to_clip": True,
                            "clip_similarity": clip_similarity,
                            "model_similarity": model_similarity,
                            "deepdanbooru_tags": [tag['tag'] for tag in deepdanbooru_tags[:5]]
                        }
                    )
                
                return clip_role, clip_similarity, boxes
                
        except Exception as e:
            logger.error(f"集成DeepDanbooru分类失败: {e}")
            # 失败时回退到CLIP模型
            logger.warning("回退到CLIP模型进行分类")
            return self.classify_image(image_path, use_model=False)
    
    def classify_pil_image(self, pil_image, use_model=False):
        """分类PIL图像对象"""
        if not self.initialize():
            return None, 0.0
        
        try:
            # 保存为临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 保存图像
            pil_image.save(temp_path)
            
            # 调用分类方法
            role, similarity, _ = self.classify_image(temp_path, use_model=use_model)
            
            # 删除临时文件
            os.unlink(temp_path)
            
            return role, similarity
        except Exception as e:
            logger.error(f"分类PIL图像失败: {e}")
            return None, 0.0
    
    def batch_classify(self, image_paths, use_model=False):
        """批量分类图像"""
        if not self.initialize():
            return []
        
        results = []
        for image_path in image_paths:
            try:
                role, similarity, boxes = self.classify_image(image_path, use_model=use_model)
                results.append({
                    'image_path': image_path,
                    'role': role,
                    'similarity': similarity,
                    'boxes': boxes
                })
            except Exception as e:
                logger.error(f"处理图片 {image_path} 失败: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results
    
    def get_available_roles(self):
        """获取可用角色列表"""
        if not self.initialize():
            return []
        
        if hasattr(self.classifier, 'role_mapping'):
            return self.classifier.role_mapping
        return []
    
    def save_index(self, index_path):
        """保存索引"""
        if not self.initialize():
            return False
        
        try:
            self.classifier.save_index(index_path)
            logger.info(f"索引已保存到: {index_path}")
            return True
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            return False
    
    def load_index(self, index_path):
        """加载索引"""
        if not self.initialize():
            return False
        
        try:
            self.classifier.load_index(index_path)
            logger.info(f"索引已从: {index_path} 加载")
            logger.info(f"包含角色: {self.classifier.role_mapping}")
            return True
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False

# 全局分类器实例
_global_classifier = None

def get_classifier(index_path="role_index", model=None):
    """获取全局分类器实例"""
    global _global_classifier
    # 构造索引文件的绝对路径
    abs_index_path = os.path.join(PROJECT_ROOT, index_path)
    if _global_classifier is None:
        _global_classifier = GeneralClassification(index_path=abs_index_path, model=model)
    else:
        # 如果已存在实例但指定了不同的模型，则重新初始化
        if model and getattr(_global_classifier, 'model', None) != model:
            _global_classifier = GeneralClassification(index_path=abs_index_path, model=model)
    return _global_classifier

def classify_image(image_path, use_model=False):
    """快速分类图像的便捷函数"""
    classifier = get_classifier()
    return classifier.classify_image(image_path, use_model=use_model)

def classify_pil_image(pil_image, use_model=False):
    """快速分类PIL图像的便捷函数"""
    classifier = get_classifier()
    return classifier.classify_pil_image(pil_image, use_model=use_model)

def classify_image_ensemble(image_path, clip_weight=0.7, model_weight=0.3, confidence_threshold=0.6):
    """快速使用集成方法分类图像的便捷函数"""
    classifier = get_classifier()
    return classifier.classify_image_ensemble(image_path, clip_weight, model_weight, confidence_threshold)

def build_index_from_directory(data_dir):
    """快速从目录构建索引的便捷函数"""
    classifier = get_classifier()
    return classifier.build_index_from_directory(data_dir)

if __name__ == "__main__":
    """测试通用分类模块"""
    # 测试初始化
    print("测试模块初始化...")
    classifier = GeneralClassification()
    classifier.initialize()
    
    # 测试从目录构建索引
    test_dir = "data/blue_archive_optimized"
    if os.path.exists(test_dir):
        print(f"\n测试从目录构建索引: {test_dir}")
        classifier.build_index_from_directory(test_dir)
        
        # 测试分类
        test_image = os.path.join(test_dir, "蔚蓝档案_星野", 
                                 [f for f in os.listdir(os.path.join(test_dir, "蔚蓝档案_星野")) 
                                  if f.endswith('.jpg')][0])
        
        print(f"\n测试分类图像 (索引模式): {test_image}")
        role, similarity, boxes = classifier.classify_image(test_image, use_model=False)
        print(f"分类结果: 角色={role}, 相似度={similarity:.4f}, 检测框={boxes}")
        
        print(f"\n测试分类图像 (模型模式): {test_image}")
        role, similarity, boxes = classifier.classify_image(test_image, use_model=True)
        print(f"分类结果: 角色={role}, 相似度={similarity:.4f}, 检测框={boxes}")
    else:
        print(f"测试目录不存在: {test_dir}")
