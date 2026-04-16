#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类auto_spider_img目录中的萝莉角色
可通过多种方式搜索实现分类角色
"""

import os
import json
import re
import time
from pathlib import Path

# 大模型配置
USE_LLM = False  # 是否使用大模型
LLM_API_KEY = "sk-ifyrynvsnvchcjoarqnrduygkeukpmoezvdmmhmfxdzlkpib"  # 大模型API密钥（默认使用JD_agent配置）
LLM_API_BASE = "https://cloud.siliconflow.cn/v1"  # API基础URL（修正为正确的地址）
LLM_MODEL = "deepseek-ai/DeepSeek-V2.5"  # 使用的大模型（修正为正确的模型名称）
LLM_TIMEOUT = 60  # 大模型超时时间（秒），增加超时时间以避免连接超时
LLM_MAX_RETRIES = 3  # 大模型最大重试次数
LLM_RETRY_DELAY = 2  # 重试延迟时间（秒）

# 尝试导入大模型库
try:
    from openai import OpenAI
except ImportError:
    print("警告: 未安装openai库，大模型功能将不可用")
    USE_LLM = False

# 加载大模型配置
def load_llm_config():
    """加载大模型配置"""
    global LLM_API_KEY, LLM_API_BASE, LLM_MODEL
    
    # 首先检查JD_agent项目中的.env文件
    jd_agent_env_path = "/Users/caozhaoqi/PycharmProjects/JD_agent/.env"
    if os.path.exists(jd_agent_env_path):
        print(f"从 {jd_agent_env_path} 读取 API 配置")
        with open(jd_agent_env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    if key == 'OPENAI_API_KEY':
                        LLM_API_KEY = value
                    elif key == 'OPENAI_API_BASE':
                        LLM_API_BASE = value
                    elif key == 'MODEL_NAME':
                        LLM_MODEL = value
    
    # 从环境变量获取配置（如果JD_agent配置不存在）
    if not LLM_API_KEY:
        LLM_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    if not LLM_API_BASE:
        LLM_API_BASE = os.environ.get('OPENAI_API_BASE', 'https://cloud.siliconflow.cn/v1')
    if not LLM_MODEL:
        LLM_MODEL = os.environ.get('MODEL_NAME', 'deepseek-ai/DeepSeek-V2.5')
    
    # 尝试从项目根目录的.env文件读取配置
    project_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if os.path.exists(project_env_path):
        print(f"从 {project_env_path} 读取 API 配置")
        with open(project_env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    if key == 'OPENAI_API_KEY':
                        LLM_API_KEY = value
                    elif key == 'OPENAI_API_BASE':
                        LLM_API_BASE = value
                    elif key == 'MODEL_NAME':
                        LLM_MODEL = value
    
    if not LLM_API_KEY:
        print("警告: 未设置 OPENAI_API_KEY 环境变量，大模型功能将不可用")
        return False
    
    print(f"使用大模型配置 - 基础URL: {LLM_API_BASE}, 模型: {LLM_MODEL}")
    return True

# 初始化加载配置
load_llm_config()

# 配置参数
AUTO_SPIDER_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img"
LOLIS_OUTPUT_DIR = os.path.join(AUTO_SPIDER_DIR, "lolis")

# 萝莉角色特征定义（参考网络标准）
LOLI_KEYWORDS = {
    # 关键词匹配（参考网络标准）
    "keywords": [
        "萝莉", "loli", "幼女", "小学生", "child", "kid", "young", "small", "tiny",
        "little girl", "young girl", "childish", "juvenile", "preteen", "tween"
    ],
    # 角色名匹配（已知的萝莉角色，参考网络资料）
    "known_lolis": [
        # 原神
        "可莉", "Klee", "迪奥娜", "Diona", "早柚", "Sayu", "七七", "Qi Qi", "瑶瑶", "Yaoyao",
        "多莉", "Dori", "柯莱", "Collei", "纳西妲", "Nahida", "派蒙", "Paimon",
        # 星穹铁道
        "虎克", "Hook", "玲妮", "Lynx", "霍霍", "huohuo", "希格雯", "Sigewinne",
        # 崩坏3
        "德丽莎", "Delisha", "苏莎娜", "Susana", "莉莉娅", "Lilibia", "萝莎莉娅", "Luoshaliaya",
        # 其他
        "康娜", "Kanna", "血小板", "血小板", "面码", "Menma", "雏田", "Hinata",
        "小樱", "Sakura", "小埋", "Umaru", "妮可", "Nico", "真白", " Mashiro"
    ],
    # 角色名模式匹配（包含特定词缀的角色，参考网络命名习惯）
    "patterns": [
        r"小\w+",  # 小X
        r"\w+酱",  # X酱
        r"\w+ちゃん",  # Xちゃん
        r"\w+kun",  # Xkun
        r"\w+chan",  # Xchan
        r"\w+baby",  # Xbaby
        r"\w+child",  # Xchild
        r"\w+kid",  # Xkid
        r"\w+幼女",  # X幼女
        r"\w+萝莉"   # X萝莉
    ]
}

# 搜索配置
SEARCH_METHODS = [
    "keyword_match",    # 关键词匹配
    "known_loli_match", # 已知萝莉匹配
    "pattern_match",    # 模式匹配
    "name_analysis",    # 名称分析
    "llm_analysis"      # 大模型分析
]

def load_keywords_from_file(file_path):
    """从文件中加载关键词"""
    keywords = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                keyword = line.strip()
                if keyword:
                    keywords.append(keyword)
    return keywords

def load_all_keywords():
    """加载所有关键词文件中的角色名"""
    all_keywords = set()
    
    # 遍历auto_spider_img目录中的所有txt文件
    for filename in os.listdir(AUTO_SPIDER_DIR):
        if filename.endswith('.txt'):
            file_path = os.path.join(AUTO_SPIDER_DIR, filename)
            keywords = load_keywords_from_file(file_path)
            all_keywords.update(keywords)
    
    # 过滤空字符串
    all_keywords = [kw for kw in all_keywords if kw]
    
    print(f"共加载 {len(all_keywords)} 个角色关键词")
    return all_keywords

def ask_llm(character):
    """询问大模型角色是否为萝莉
    
    Args:
        character: 角色名
        
    Returns:
        bool: 是否为萝莉角色
    """
    if not USE_LLM or 'OpenAI' not in globals() or not LLM_API_KEY:
        return False
    
    # 创建OpenAI客户端
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE
    )
    
    # 构建提示词
    prompt = f"""请判断以下动漫/游戏角色是否为萝莉（loli）：{character}
    
    萝莉的定义：
    - 年龄通常在12-14岁以下
    - 外貌可爱，身材娇小
    - 通常具有天真、活泼的性格
    - 与幼女（年龄更小）有所区别，但界限可能模糊
    
    请直接回答"是"或"否"，不要添加任何其他解释。"""
    
    # 尝试调用大模型
    for i in range(LLM_MAX_RETRIES):
        try:
            print(f"调用大模型 ({i+1}/{LLM_MAX_RETRIES})...")
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个动漫角色识别专家"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            # 解析响应
            answer = response.choices[0].message.content.strip()
            print(f"大模型响应: {answer}")
            return answer == "是"
        except Exception as e:
            print(f"大模型调用失败 ({i+1}/{LLM_MAX_RETRIES}): {e}")
            time.sleep(LLM_RETRY_DELAY)
    
    return False

def is_loli_character(character, method="all"):
    """判断角色是否为萝莉
    
    Args:
        character: 角色名
        method: 搜索方法，可选值："all", "keyword_match", "known_loli_match", "pattern_match", "name_analysis", "llm_analysis"
        
    Returns:
        bool: 是否为萝莉角色
        list: 使用的匹配方法
    """
    used_methods = []
    
    # 转换为小写用于匹配
    char_lower = character.lower()
    
    # 关键词匹配
    if method == "all" or method == "keyword_match":
        for keyword in LOLI_KEYWORDS["keywords"]:
            if keyword.lower() in char_lower:
                used_methods.append("keyword_match")
                return True, used_methods
    
    # 已知萝莉匹配
    if method == "all" or method == "known_loli_match":
        if character in LOLI_KEYWORDS["known_lolis"]:
            used_methods.append("known_loli_match")
            return True, used_methods
    
    # 模式匹配
    if method == "all" or method == "pattern_match":
        for pattern in LOLI_KEYWORDS["patterns"]:
            if re.search(pattern, character):
                used_methods.append("pattern_match")
                return True, used_methods
    
    # 名称分析
    if method == "all" or method == "name_analysis":
        # 分析角色名长度和特征（参考网络标准）
        if len(character) <= 3:
            # 短名称可能是萝莉
            used_methods.append("name_analysis")
            return True, used_methods
        
        # 检查是否包含表示年龄小的词汇
        young_indicators = ["小", "幼", "baby", "child", "kid", "young", "little"]
        for indicator in young_indicators:
            if indicator in char_lower:
                used_methods.append("name_analysis")
                return True, used_methods
    
    # 大模型分析
    if method == "all" or method == "llm_analysis":
        if ask_llm(character):
            used_methods.append("llm_analysis")
            return True, used_methods
    
    return False, used_methods

def classify_characters(characters, methods=None):
    """分类角色
    
    Args:
        characters: 角色列表
        methods: 使用的搜索方法列表
        
    Returns:
        dict: 分类结果，包含萝莉角色和非萝莉角色
    """
    if methods is None:
        methods = SEARCH_METHODS
    
    loli_characters = []
    non_loli_characters = []
    classification_details = {}
    
    for character in characters:
        is_loli = False
        used_methods = []
        
        for method in methods:
            result, method_used = is_loli_character(character, method)
            if result:
                is_loli = True
                used_methods.extend(method_used)
                break
        
        if is_loli:
            loli_characters.append(character)
            classification_details[character] = {
                "is_loli": True,
                "used_methods": used_methods
            }
        else:
            non_loli_characters.append(character)
            classification_details[character] = {
                "is_loli": False,
                "used_methods": []
            }
    
    return {
        "loli_characters": loli_characters,
        "non_loli_characters": non_loli_characters,
        "classification_details": classification_details
    }

def save_classification_result(result):
    """保存分类结果"""
    # 创建输出目录
    os.makedirs(LOLIS_OUTPUT_DIR, exist_ok=True)
    
    # 保存萝莉角色列表
    loli_file = os.path.join(LOLIS_OUTPUT_DIR, "loli_characters.txt")
    with open(loli_file, 'w', encoding='utf-8') as f:
        for character in result["loli_characters"]:
            f.write(f"{character}\n")
    
    # 保存非萝莉角色列表
    non_loli_file = os.path.join(LOLIS_OUTPUT_DIR, "non_loli_characters.txt")
    with open(non_loli_file, 'w', encoding='utf-8') as f:
        for character in result["non_loli_characters"]:
            f.write(f"{character}\n")
    
    # 保存分类详情
    details_file = os.path.join(LOLIS_OUTPUT_DIR, "classification_details.json")
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(result["classification_details"], f, ensure_ascii=False, indent=2)
    
    print(f"分类结果已保存到: {LOLIS_OUTPUT_DIR}")



def main():
    """主函数"""
    print("=" * 80)
    print("分类auto_spider_img目录中的萝莉角色")
    print("=" * 80)
    
    # 加载所有角色关键词
    characters = load_all_keywords()
    
    if not characters:
        print("没有找到角色关键词")
        return
    
    # 询问用户是否使用大模型
    use_llm_input = input("是否使用大模型进行更精确的分类？(y/n): ").strip().lower()
    global USE_LLM
    if use_llm_input == "y":
        # 重新加载配置
        config_loaded = load_llm_config()
        
        # 询问用户是否有API密钥
        api_key = input("请输入OpenAI API密钥（留空使用默认值）: ").strip()
        if api_key:
            global LLM_API_KEY
            LLM_API_KEY = api_key
            config_loaded = True
        
        if config_loaded and LLM_API_KEY:
            USE_LLM = True
            print("将使用大模型进行分类，这可能会花费一些时间...")
        else:
            USE_LLM = False
            print("大模型配置不可用，将使用传统方法进行分类...")
    else:
        USE_LLM = False
        print("将使用传统方法进行分类...")
    
    # 分类角色
    print("\n开始分类角色...")
    result = classify_characters(characters)
    
    # 输出分类结果
    print(f"\n分类结果:")
    print(f"萝莉角色数量: {len(result['loli_characters'])}")
    print(f"非萝莉角色数量: {len(result['non_loli_characters'])}")
    
    # 显示前20个萝莉角色
    print("\n前20个萝莉角色:")
    for i, character in enumerate(result['loli_characters'][:20], 1):
        methods = result['classification_details'][character]['used_methods']
        print(f"  {i}. {character} (匹配方法: {', '.join(methods)})")
    
    # 保存分类结果
    save_classification_result(result)
    
    # 输出分类方法
    classification_methods = "关键词匹配、已知萝莉匹配、模式匹配和名称分析"
    if USE_LLM:
        classification_methods += "和大模型分析"
    
    print("\n" + "=" * 80)
    print("分类完成")
    print("=" * 80)
    print(f"萝莉角色列表已保存到: {os.path.join(LOLIS_OUTPUT_DIR, 'loli_characters.txt')}")
    print(f"分类方法参考了网络标准，包括{classification_methods}")
    print("=" * 80)

if __name__ == "__main__":
    main()