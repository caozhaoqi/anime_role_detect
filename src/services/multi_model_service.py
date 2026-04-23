import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple

from src.core.logging.global_logger import get_logger
from src.services.processor.model_processor import process_with_local_model, process_with_model_service

logger = get_logger("multi_model_service")

class MultiModelService:
    """多模型集成服务"""
    
    _instance: Optional['MultiModelService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return
        
        self.initialized = True
        
        # 模型配置
        self.model_configs = {
            "resnet18_loli8": {
                "type": "local",
                "weight": 0.3
            },
            "mobilenet_v2": {
                "type": "local",
                "weight": 0.25
            },
            "vit_b_16": {
                "type": "service",
                "weight": 0.45
            }
        }
        
        # 结果融合策略
        self.fusion_strategy = "weighted_average"  # weighted_average, majority_vote, max_confidence
        
        logger.info("多模型集成服务初始化完成")
    
    async def process_with_multiple_models(self, file, content, multi_role=False) -> Dict[str, Any]:
        """
        使用多个模型处理图像
        
        Args:
            file: 上传的文件
            content: 文件内容
            multi_role: 是否使用多角色检测
        
        Returns:
            dict: 融合后的处理结果
        """
        try:
            # 并行执行多个模型
            tasks = []
            for model_name, config in self.model_configs.items():
                if config["type"] == "local":
                    task = process_with_local_model(file, content, model_name)
                else:
                    task = process_with_model_service(file, content, model_name, multi_role)
                tasks.append((model_name, config["weight"], task))
            
            # 等待所有任务完成
            results = []
            for model_name, weight, task in tasks:
                try:
                    result = await task
                    results.append((model_name, weight, result))
                    logger.info(f"模型 {model_name} 处理完成")
                except Exception as e:
                    logger.error(f"模型 {model_name} 处理失败: {e}")
            
            # 融合结果
            if not results:
                logger.error("所有模型处理失败")
                return {"error": "所有模型处理失败"}
            
            if multi_role:
                fused_result = self._fuse_multi_role_results(results)
            else:
                fused_result = self._fuse_single_role_results(results)
            
            return fused_result
        except Exception as e:
            logger.error(f"多模型处理失败: {e}")
            raise
    
    def _fuse_single_role_results(self, results: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        融合单角色检测结果
        
        Args:
            results: 模型处理结果列表，每个元素为 (model_name, weight, result)
        
        Returns:
            dict: 融合后的结果
        """
        # 提取有效结果
        valid_results = []
        for model_name, weight, result in results:
            if "error" not in result:
                valid_results.append((model_name, weight, result))
        
        if not valid_results:
            return {"error": "所有模型返回错误"}
        
        # 根据融合策略处理
        if self.fusion_strategy == "weighted_average":
            return self._weighted_average_fusion(valid_results)
        elif self.fusion_strategy == "majority_vote":
            return self._majority_vote_fusion(valid_results)
        elif self.fusion_strategy == "max_confidence":
            return self._max_confidence_fusion(valid_results)
        else:
            return self._weighted_average_fusion(valid_results)
    
    def _fuse_multi_role_results(self, results: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        融合多角色检测结果
        
        Args:
            results: 模型处理结果列表，每个元素为 (model_name, weight, result)
        
        Returns:
            dict: 融合后的结果
        """
        # 提取有效结果
        valid_results = []
        for model_name, weight, result in results:
            if "error" not in result and "roles" in result:
                valid_results.append((model_name, weight, result))
        
        if not valid_results:
            return {"error": "所有模型返回错误"}
        
        # 合并角色检测结果
        all_roles = []
        for model_name, weight, result in valid_results:
            roles = result.get("roles", [])
            for role in roles:
                # 为每个角色添加模型信息和权重
                role["model"] = model_name
                role["model_weight"] = weight
                all_roles.append(role)
        
        # 按角色名称分组
        role_groups = {}
        for role in all_roles:
            role_name = role.get("role", "unknown")
            if role_name not in role_groups:
                role_groups[role_name] = []
            role_groups[role_name].append(role)
        
        # 对每个角色组进行融合
        fused_roles = []
        for role_name, role_list in role_groups.items():
            # 计算加权平均相似度和置信度
            total_weight = sum(role.get("model_weight", 1.0) for role in role_list)
            weighted_similarity = sum(role.get("similarity", 0.0) * role.get("model_weight", 1.0) for role in role_list) / total_weight
            weighted_confidence = sum(role.get("confidence", 0.0) * role.get("model_weight", 1.0) for role in role_list) / total_weight
            
            # 合并标签
            all_tags = set()
            for role in role_list:
                all_tags.update(role.get("tags", []))
            
            # 取第一个角色的边界框（可以根据需要改进）
            bbox = role_list[0].get("bbox", {})
            
            fused_role = {
                "role": role_name,
                "similarity": weighted_similarity,
                "confidence": weighted_confidence,
                "tags": list(all_tags),
                "bbox": bbox,
                "models": [role.get("model") for role in role_list]
            }
            fused_roles.append(fused_role)
        
        # 按相似度排序
        fused_roles.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        
        # 构建融合结果
        fused_result = {
            "roles": fused_roles,
            "count": len(fused_roles),
            "text_detections": valid_results[0][2].get("text_detections", []),
            "keypoints": valid_results[0][2].get("keypoints", []),
            "ai_predicted_role": valid_results[0][2].get("ai_predicted_role", "unknown"),
            "nsfw": valid_results[0][2].get("nsfw", {"is_nsfw": False, "details": {}})
        }
        
        return fused_result
    
    def _weighted_average_fusion(self, results: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        加权平均融合策略
        
        Args:
            results: 模型处理结果列表
        
        Returns:
            dict: 融合后的结果
        """
        # 计算加权平均相似度
        total_weight = sum(weight for _, weight, _ in results)
        weighted_similarity = sum(result.get("similarity", 0.0) * weight for _, weight, result in results) / total_weight
        
        # 合并属性和标签
        all_attributes = set()
        all_tags = set()
        for _, _, result in results:
            all_attributes.update(result.get("attributes", []))
            all_tags.update(result.get("tags", []))
        
        # 构建融合结果
        fused_result = {
            "role": self._get_most_common_role(results),
            "similarity": weighted_similarity,
            "possible_roles": [],
            "attributes": list(all_attributes),
            "tags": list(all_tags),
            "text_detections": results[0][2].get("text_detections", []),
            "keypoints": results[0][2].get("keypoints", []),
            "ai_predicted_role": results[0][2].get("ai_predicted_role", "unknown"),
            "nsfw": results[0][2].get("nsfw", {"is_nsfw": False, "details": {}}),
            "models": [model_name for model_name, _, _ in results]
        }
        
        return fused_result
    
    def _majority_vote_fusion(self, results: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        多数投票融合策略
        
        Args:
            results: 模型处理结果列表
        
        Returns:
            dict: 融合后的结果
        """
        # 统计角色出现次数
        role_counts = {}
        for _, _, result in results:
            role = result.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # 选择出现次数最多的角色
        most_common_role = max(role_counts, key=role_counts.get)
        
        # 计算该角色的平均相似度
        similarities = [result.get("similarity", 0.0) for _, _, result in results if result.get("role") == most_common_role]
        average_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # 合并属性和标签
        all_attributes = set()
        all_tags = set()
        for _, _, result in results:
            all_attributes.update(result.get("attributes", []))
            all_tags.update(result.get("tags", []))
        
        # 构建融合结果
        fused_result = {
            "role": most_common_role,
            "similarity": average_similarity,
            "possible_roles": [],
            "attributes": list(all_attributes),
            "tags": list(all_tags),
            "text_detections": results[0][2].get("text_detections", []),
            "keypoints": results[0][2].get("keypoints", []),
            "ai_predicted_role": results[0][2].get("ai_predicted_role", "unknown"),
            "nsfw": results[0][2].get("nsfw", {"is_nsfw": False, "details": {}}),
            "models": [model_name for model_name, _, _ in results]
        }
        
        return fused_result
    
    def _max_confidence_fusion(self, results: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        最大置信度融合策略
        
        Args:
            results: 模型处理结果列表
        
        Returns:
            dict: 融合后的结果
        """
        # 选择相似度最高的结果
        best_result = max(results, key=lambda x: x[2].get("similarity", 0.0))
        _, _, best_result_data = best_result
        
        # 合并其他模型的属性和标签
        all_attributes = set(best_result_data.get("attributes", []))
        all_tags = set(best_result_data.get("tags", []))
        for _, _, result in results:
            if result != best_result_data:
                all_attributes.update(result.get("attributes", []))
                all_tags.update(result.get("tags", []))
        
        # 构建融合结果
        fused_result = best_result_data.copy()
        fused_result["attributes"] = list(all_attributes)
        fused_result["tags"] = list(all_tags)
        fused_result["models"] = [model_name for model_name, _, _ in results]
        
        return fused_result
    
    def _get_most_common_role(self, results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """
        获取最常见的角色
        
        Args:
            results: 模型处理结果列表
        
        Returns:
            str: 最常见的角色
        """
        role_counts = {}
        for _, _, result in results:
            role = result.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return max(role_counts, key=role_counts.get) if role_counts else "unknown"
    
    def add_model(self, model_name: str, model_type: str, weight: float):
        """
        添加模型
        
        Args:
            model_name: 模型名称
            model_type: 模型类型 (local 或 service)
            weight: 模型权重
        """
        self.model_configs[model_name] = {
            "type": model_type,
            "weight": weight
        }
        logger.info(f"添加模型: {model_name}, 类型: {model_type}, 权重: {weight}")
    
    def remove_model(self, model_name: str):
        """
        移除模型
        
        Args:
            model_name: 模型名称
        """
        if model_name in self.model_configs:
            del self.model_configs[model_name]
            logger.info(f"移除模型: {model_name}")
    
    def set_fusion_strategy(self, strategy: str):
        """
        设置融合策略
        
        Args:
            strategy: 融合策略 (weighted_average, majority_vote, max_confidence)
        """
        valid_strategies = ["weighted_average", "majority_vote", "max_confidence"]
        if strategy in valid_strategies:
            self.fusion_strategy = strategy
            logger.info(f"设置融合策略: {strategy}")
        else:
            logger.error(f"无效的融合策略: {strategy}, 有效值: {valid_strategies}")
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        获取模型配置
        
        Returns:
            dict: 模型配置
        """
        return self.model_configs
    
    def get_fusion_strategy(self) -> str:
        """
        获取融合策略
        
        Returns:
            str: 融合策略
        """
        return self.fusion_strategy

# 全局多模型服务实例
_multi_model_service: Optional[MultiModelService] = None

def get_multi_model_service() -> MultiModelService:
    """获取多模型服务实例"""
    global _multi_model_service
    if _multi_model_service is None:
        _multi_model_service = MultiModelService()
    return _multi_model_service

def init_multi_model_service():
    """初始化多模型服务"""
    global _multi_model_service
    if _multi_model_service is None:
        _multi_model_service = MultiModelService()
        logger.info("多模型服务初始化完成")
    return _multi_model_service

def process_with_multiple_models(file, content, multi_role=False) -> Dict[str, Any]:
    """
    使用多个模型处理图像
    
    Args:
        file: 上传的文件
        content: 文件内容
        multi_role: 是否使用多角色检测
    
    Returns:
        dict: 融合后的处理结果
    """
    return asyncio.run(get_multi_model_service().process_with_multiple_models(file, content, multi_role))

def add_model(model_name: str, model_type: str, weight: float):
    """
    添加模型
    
    Args:
        model_name: 模型名称
        model_type: 模型类型 (local 或 service)
        weight: 模型权重
    """
    get_multi_model_service().add_model(model_name, model_type, weight)

def remove_model(model_name: str):
    """
    移除模型
    
    Args:
        model_name: 模型名称
    """
    get_multi_model_service().remove_model(model_name)

def set_fusion_strategy(strategy: str):
    """
    设置融合策略
    
    Args:
        strategy: 融合策略 (weighted_average, majority_vote, max_confidence)
    """
    get_multi_model_service().set_fusion_strategy(strategy)

def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    获取模型配置
    
    Returns:
        dict: 模型配置
    """
    return get_multi_model_service().get_model_configs()

def get_fusion_strategy() -> str:
    """
    获取融合策略
    
    Returns:
        str: 融合策略
    """
    return get_multi_model_service().get_fusion_strategy()
