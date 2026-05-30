#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大模型调用接口
参考ai_role_prediction.py，提供通用的大模型调用功能
"""

import os
import sys
import json
import requests
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llm_interface")

# 全局单例实例
_llm_client_instance = None


class LLMClient:
    """
    大模型客户端
    参考ai_role_prediction.py，提供通用的大模型调用功能
    """

    def __new__(cls):
        """创建单例实例"""
        global _llm_client_instance
        if _llm_client_instance is None:
            _llm_client_instance = super(LLMClient, cls).__new__(cls)
            _llm_client_instance._initialized = False
        return _llm_client_instance

    def __init__(self):
        """初始化大模型客户端"""
        if not getattr(self, "_initialized", False):
            logger.info("初始化大模型客户端")
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

            # 添加响应缓存
            self.response_cache = {}
            self._initialized = True

    def call_llm(self, prompt, system_prompt="你是一个智能助手", max_tokens=200, temperature=0.7):
        """
        调用大模型

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            max_tokens: 最大生成长度
            temperature: 温度参数

        Returns:
            大模型的响应
        """
        # 生成缓存键
        cache_key = f"{system_prompt[:100]}_{prompt[:200]}_{max_tokens}_{temperature}"

        # 检查缓存
        if cache_key in self.response_cache:
            logger.info("从缓存获取大模型响应")
            return self.response_cache[cache_key]

        logger.info("调用大模型 API")

        # 构建请求数据
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # 调用真实的 AI API
        try:
            response = self._call_api(data)
            # 缓存结果
            self.response_cache[cache_key] = response
            logger.info("大模型响应获取成功")
            return response
        except Exception as e:
            logger.error(f"API 调用失败: {e}")
            # 禁用模拟响应，确保只使用真实的大模型API调用
            raise

    def _call_api(self, data):
        """
        调用 API

        Args:
            data: 请求数据

        Returns:
            API 响应
        """
        if not self.api_key:
            raise ValueError("未设置 OPENAI_API_KEY 环境变量")

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        logger.info(f"调用 AI API - 模型: {self.model_name}")
        # 添加超时时间
        response = requests.post(
            self.api_url, headers=headers, json=data, timeout=30
        )  # 增加超时时间到30秒
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        return content

    def _simulate_response(self, prompt, system_prompt):
        """
        模拟大模型响应

        Args:
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            模拟的响应
        """
        logger.info("使用模拟响应")

        # 基于提示内容的简单规则匹配
        prompt_lower = prompt.lower()

        # 角色相关查询
        if "角色" in prompt or "character" in prompt_lower:
            if "原神" in prompt or "genshin" in prompt_lower:
                return "原神中的角色包括：琴、安柏、丽莎、芭芭拉、可莉、诺艾尔、菲谢尔、砂糖、莫娜、迪奥娜等。"
            elif "星穹铁道" in prompt or "star rail" in prompt_lower:
                return "星穹铁道中的角色包括：艾丝妲、三月七、希露瓦、黑塔、银狼、希儿、卡芙卡、素裳、姬子、布洛妮娅等。"
            elif "崩坏3" in prompt or "honkai 3" in prompt_lower:
                return "崩坏3中的角色包括：布洛妮娅、符华、希儿、格蕾修、丽塔、爱莉希雅、琪亚娜、雷电芽衣等。"
            else:
                return "常见的动漫角色包括：漩涡鸣人、宇智波佐助、孙悟空、路飞、柯南、初音未来、樱木花道等。"

        # 分类相关查询
        elif "分类" in prompt or "classify" in prompt_lower:
            if "萝莉" in prompt or "loli" in prompt_lower:
                return "萝莉是指年幼或外表年幼的女性角色，通常具有可爱的特征，如娇小的体型、天真的性格等。"
            elif "御姐" in prompt or "onee-san" in prompt_lower:
                return "御姐是指成熟、自信的女性角色，通常具有优雅的气质和较强的能力。"
            else:
                return "动漫角色可以根据多种维度分类，如年龄、性格、职业、外表特征等。"

        # 其他查询
        else:
            return "这是一个模拟的大模型响应。在实际使用中，这里会返回真实的大模型生成内容。"

    def get_game_characters(self, game_name):
        """
        获取指定游戏的角色列表

        Args:
            game_name: 游戏名称

        Returns:
            角色列表
        """
        # 多种提示词策略
        prompts = [
            f"列出{game_name}游戏中的所有可玩角色，每行一个，不要编号",
            f"{game_name}游戏有哪些主要角色？每行一个角色名",
            f"{game_name}游戏的完整角色列表，按出场顺序排列，每行一个",
            f"{game_name}游戏中所有已发布的角色，每行一个角色名",
        ]

        all_characters = set()

        for i, prompt in enumerate(prompts):
            logger.info(f"尝试提示词 {i+1}/{len(prompts)}: {prompt[:50]}...")

            # 多次重试
            for retry in range(3):
                try:
                    system_prompt = "你是一个游戏角色专家，熟悉各种游戏中的角色"
                    response = self.call_llm(
                        prompt, system_prompt, max_tokens=1000, temperature=0.7
                    )

                    # 解析角色列表
                    characters = self._parse_character_list(response, game_name)
                    if characters:
                        all_characters.update(characters)
                        logger.info(f"从提示词获取到 {len(characters)} 个角色")
                    break  # 成功后退出重试循环
                except Exception as e:
                    logger.warning(f"第 {retry+1} 次尝试失败: {e}")
                    time.sleep(2)  # 等待后重试

        # 去重并过滤
        characters = list(all_characters)
        # 过滤掉空字符串和无效角色名
        characters = [
            c
            for c in characters
            if c
            and len(c) > 1
            and not any(keyword in c for keyword in ["角色", "包括", "中的", "等", "游戏"])
        ]
        logger.info(f"从大模型获取到 {len(characters)} 个 {game_name} 角色")

        return characters

    def _parse_character_list(self, response, game_name):
        """
        解析角色列表

        Args:
            response: 大模型响应
            game_name: 游戏名称

        Returns:
            角色列表
        """
        characters = []

        # 按行分割
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("http") and not line.startswith("#"):
                # 移除前缀文本
                if "角色包括：" in line:
                    line = line.split("角色包括：")[1]
                elif "包括：" in line:
                    line = line.split("包括：")[1]
                elif "：" in line:
                    line = line.split("：")[1]

                # 移除编号和标点
                line = line.replace("、", ",").replace("，", ",")
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    # 移除数字编号
                    part = "".join([c for c in part if not c.isdigit()])
                    # 移除标点符号
                    part = part.strip(" .，。、：:等")
                    # 移除游戏名称
                    part = part.replace(game_name, "").strip()
                    if part and len(part) > 1:
                        characters.append(part)

        return characters

    def clear_cache(self):
        """
        清空缓存
        """
        self.response_cache.clear()
        logger.info("缓存已清空")

    def export_cache(self, output_file):
        """
        导出缓存

        Args:
            output_file: 输出文件路径
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.response_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"缓存已导出到: {output_file}")
        except Exception as e:
            logger.error(f"导出缓存失败: {e}")

    def import_cache(self, input_file):
        """
        导入缓存

        Args:
            input_file: 输入文件路径
        """
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.response_cache.update(data)
            logger.info(f"缓存已从 {input_file} 导入")
        except Exception as e:
            logger.error(f"导入缓存失败: {e}")


def main():
    """
    主函数
    """
    # 初始化大模型客户端
    client = LLMClient()

    # 测试获取游戏角色
    games = ["原神", "星穹铁道", "崩坏3"]
    for game in games:
        print(f"\n获取 {game} 角色列表:")
        characters = client.get_game_characters(game)
        print(f"共获取到 {len(characters)} 个角色")
        print(f"前10个角色: {characters[:10]}")

    # 测试通用查询
    print("\n测试通用查询:")
    response = client.call_llm("什么是萝莉？")
    print(f"响应: {response}")


if __name__ == "__main__":
    main()
