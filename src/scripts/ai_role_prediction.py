#!/usr/bin/env python3
"""
AI 角色预测脚本
参考 JD Agent 项目的多智能体协同方法，根据识别出的标签推测角色名
"""

import os
import sys
import json
import requests

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 配置日志
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ai_role_prediction")

# 全局单例实例
_ai_role_predictor_instance = None


class AIRolePredictor:
    """
    AI 角色预测器
    参考 JD Agent 项目的多智能体协同方法
    """

    def __new__(cls):
        """创建单例实例"""
        global _ai_role_predictor_instance
        if _ai_role_predictor_instance is None:
            _ai_role_predictor_instance = super(AIRolePredictor, cls).__new__(cls)
            _ai_role_predictor_instance._initialized = False
        return _ai_role_predictor_instance

    def __init__(self):
        """初始化 AI 角色预测器"""
        if not getattr(self, "_initialized", False):
            logger.info("初始化 AI 角色预测器")
            # 从环境变量获取 API 配置
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
            self.api_base = os.environ.get("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
            self.model_name = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

            # 尝试从项目根目录的.env文件读取配置
            # 使用绝对路径
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            project_env_path = os.path.join(project_root, ".env")
            logger.info(f"尝试读取 .env 文件: {project_env_path}")
            if os.path.exists(project_env_path):
                logger.info(f"从 {project_env_path} 读取 API 配置")
                with open(project_env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            key, value = line.split("=", 1)
                            if key == "OPENAI_API_KEY":
                                self.api_key = value.strip()
                                logger.info("成功读取 OPENAI_API_KEY")
                            elif key == "OPENAI_API_BASE":
                                self.api_base = value.strip()
                                logger.info(f"成功读取 OPENAI_API_BASE: {self.api_base}")
                            elif key == "MODEL_NAME":
                                self.model_name = value.strip()
                                logger.info(f"成功读取 MODEL_NAME: {self.model_name}")
            else:
                logger.warning(f".env 文件不存在: {project_env_path}")

            if not self.api_key:
                logger.warning("未设置 OPENAI_API_KEY 环境变量，将使用模拟响应")

            self.api_url = f"{self.api_base}/chat/completions"
            logger.info(f"使用 API 配置 - 基础URL: {self.api_base}, 模型: {self.model_name}")

            # 添加预测缓存
            self.prediction_cache = {}
            self._initialized = True

    def predict_role(self, tags):
        """
        根据标签预测角色名

        Args:
            tags: 标签列表（可以是字符串列表或字典列表）

        Returns:
            预测的角色名
        """
        # 处理标签格式
        processed_tags = []
        for tag_item in tags:
            if isinstance(tag_item, dict) and "tag" in tag_item:
                processed_tags.append(tag_item["tag"])
            elif isinstance(tag_item, str):
                processed_tags.append(tag_item)

        # 生成缓存键
        cache_key = tuple(sorted(processed_tags[:10]))  # 使用前10个标签作为缓存键

        # 检查缓存
        if cache_key in self.prediction_cache:
            logger.info("从缓存获取角色预测结果")
            return self.prediction_cache[cache_key]

        logger.info(f"开始预测角色，标签数量: {len(processed_tags)}")
        logger.info(f"前10个标签: {processed_tags[:10]}")

        # 构建提示
        prompt = self._build_prompt(processed_tags)

        # 调用真实的 AI API
        try:
            predicted_role = self._call_openai_api(prompt)
        except Exception as e:
            logger.error(f"API 调用失败: {e}")
            # 如果 API 调用失败，使用模拟响应
            predicted_role = self._simulate_ai_response(prompt, processed_tags)

        # 缓存结果
        self.prediction_cache[cache_key] = predicted_role
        logger.info(f"预测角色: {predicted_role}")
        return predicted_role

    def _build_prompt(self, tags):
        """
        构建提示

        Args:
            tags: 标签列表

        Returns:
            提示字符串
        """
        prompt = f"""你是一个动漫角色识别专家，根据以下标签推测可能的角色名：

标签列表：
{', '.join(tags[:20])}  # 只使用前20个标签，避免提示过长

请根据这些标签，推测最可能的动漫角色名。要求：
1. 只输出角色名，不要输出其他内容
2. 基于标签的语义关联进行推测
3. 考虑常见的动漫角色特征
4. 如果无法确定，请输出'未知角色'
"""
        return prompt

    def _call_openai_api(self, prompt):
        """
        调用 AI API

        Args:
            prompt: 提示字符串

        Returns:
            AI 响应的角色名
        """
        if not self.api_key:
            raise ValueError("未设置 OPENAI_API_KEY 环境变量")

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个动漫角色识别专家"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 50,
            "temperature": 0.7,
        }

        logger.info(f"调用 AI API - 模型: {self.model_name}")
        # 添加超时时间，避免无限期等待
        response = requests.post(
            self.api_url, headers=headers, json=data, timeout=30
        )  # 增加超时时间到30秒
        response.raise_for_status()

        result = response.json()
        predicted_role = result["choices"][0]["message"]["content"].strip()

        # 确保返回的是有效的角色名
        if not predicted_role or predicted_role == "未知角色":
            return "未知角色"

        return predicted_role

    def _simulate_ai_response(self, prompt, tags):
        """
        模拟 AI 响应

        Args:
            prompt: 提示字符串
            tags: 标签列表

        Returns:
            模拟的 AI 响应
        """
        # 基于标签的简单规则匹配
        tags_lower = [tag.lower() for tag in tags]
        logger.info(f"处理后的标签: {tags_lower[:10]}")

        # 检查是否包含特定角色的特征标签
        if "klee" in tags_lower or "可莉" in tags_lower:
            logger.info("匹配到角色: 可莉")
            return "可莉"
        elif "arataki itto" in tags_lower or "荒泷一斗" in tags_lower:
            logger.info("匹配到角色: 荒泷一斗")
            return "荒泷一斗"
        elif "raiden shogun" in tags_lower or "雷电将军" in tags_lower:
            logger.info("匹配到角色: 雷电将军")
            return "雷电将军"
        elif "hu tao" in tags_lower or "胡桃" in tags_lower:
            logger.info("匹配到角色: 胡桃")
            return "胡桃"
        elif any("hina" in tag for tag in tags_lower):
            logger.info("匹配到角色: 日奈 (通过 hina 标签)")
            return "日奈"
        elif "日奈" in tags_lower:
            logger.info("匹配到角色: 日奈 (通过 日奈 标签)")
            return "日奈"
        elif "genshin" in tags_lower or "原神" in tags_lower:
            logger.info("匹配到角色类型: 原神角色")
            return "原神角色"
        elif "honkai" in tags_lower or "崩坏" in tags_lower:
            logger.info("匹配到角色类型: 崩坏角色")
            return "崩坏角色"
        elif "anime" in tags_lower or "动漫" in tags_lower:
            logger.info("匹配到角色类型: 动漫角色")
            return "动漫角色"
        else:
            # 根据一些常见标签组合推测
            if "1girl" in tags_lower and "breasts" in tags_lower and "swimsuit" in tags_lower:
                logger.info("匹配到角色类型: 泳装女孩")
                return "泳装女孩"
            elif "1boy" in tags_lower and "sword" in tags_lower:
                logger.info("匹配到角色类型: 剑士")
                return "剑士"
            elif "cat" in tags_lower and "girl" in tags_lower:
                logger.info("匹配到角色类型: 猫耳女孩")
                return "猫耳女孩"
            else:
                logger.info("未匹配到任何角色")
                return "未知角色"


def main():
    """主函数"""
    # 示例标签（从之前的 API 响应中提取）
    tags = [
        "1girl",
        "breasts",
        "animal_print",
        "lactation",
        "cow_print",
        "nipples",
        "questionable",
        "swimsuit",
        "doughnut",
        "huge_breasts",
        "bell",
        "neck_bell",
        "hairclip",
        "bikini",
        "hair_ornament",
        "english_text",
        "mouth_hold",
        "colored_skin",
        "collar",
        "cow_print_bikini",
    ]

    # 初始化预测器
    predictor = AIRolePredictor()

    # 预测角色
    predicted_role = predictor.predict_role(tags)

    # 输出结果
    print(f"AI 识别结果: {predicted_role}")


if __name__ == "__main__":
    main()
