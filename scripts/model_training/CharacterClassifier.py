import requests
import re
import os

class CharacterClassifier:
    def __init__(self):
        # 萌娘百科 API 地址
        self.moegirl_api_url = "https://zh.moegirl.org.cn/api.php"
        
        # 维基百科 API 地址
        self.wikipedia_api_url = "https://zh.wikipedia.org/w/api.php"
        self.wikipedia_en_api_url = "https://en.wikipedia.org/w/api.php"
        
        # 定义 NLP 关键词权重字典 (特征工程)
        self.keywords_weights = {
            "萝莉": 5.0,
            "合法萝莉": 5.0,
            "万年萝莉": 5.0,
            "幼女": 4.0,
            "体型娇小": 2.0,
            "娇小": 1.5,
            "小女孩": 2.0,
            "平胸": 1.0,
            "小学生": 2.0,
            "童颜": 1.5
        }
        
        # 反向特征（降低该分类概率的特征）
        self.negative_keywords = {
            "御姐": -4.0,
            "人妻": -3.0,
            "巨乳": -3.0,
            "熟女": -4.0,
            "成年女性": -3.0
        }

    def fetch_from_moegirl(self, character_name):
        """
        从萌娘百科获取角色数据
        """
        params = {
            "action": "query",
            "prop": "extracts|categories",
            "titles": character_name,
            "format": "json",
            "exintro": True,
            "explaintext": True,
            "cllimit": "max"
        }
        
        try:
            response = requests.get(self.moegirl_api_url, params=params, timeout=10)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            
            for page_id, page_info in pages.items():
                if page_id == "-1":
                    return None
                return page_info
        except Exception as e:
            print(f"萌娘百科请求错误: {e}")
            return None
    
    def fetch_from_wikipedia(self, character_name, lang="zh"):
        """
        从维基百科获取角色数据
        
        Args:
            character_name: 角色名称
            lang: 语言，默认为中文
        """
        api_url = self.wikipedia_api_url if lang == "zh" else self.wikipedia_en_api_url
        
        params = {
            "action": "query",
            "prop": "extracts",
            "titles": character_name,
            "format": "json",
            "exintro": True,
            "explaintext": True
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            data = response.json()
            
            # 检查API响应是否有效
            if "query" not in data:
                print(f"维基百科({lang})API响应无效: {data}")
                return None
            
            pages = data.get("query", {}).get("pages", {})
            
            for page_id, page_info in pages.items():
                if page_id == "-1":
                    return None
                # 添加分类信息（空列表）
                page_info["categories"] = []
                return page_info
        except Exception as e:
            print(f"维基百科({lang})请求错误: {e}")
            return None
    
    def fetch_character_data(self, character_name):
        """
        综合从多个数据源获取角色数据
        优先级：萌娘百科 -> 中文维基百科 -> 英文维基百科
        """
        # 1. 首先尝试萌娘百科
        page_info = self.fetch_from_moegirl(character_name)
        if page_info:
            print(f"从萌娘百科找到角色: {character_name}")
            return page_info
        
        # 2. 尝试中文维基百科
        page_info = self.fetch_from_wikipedia(character_name, lang="zh")
        if page_info:
            print(f"从中文维基百科找到角色: {character_name}")
            return page_info
        
        # 3. 尝试英文维基百科
        page_info = self.fetch_from_wikipedia(character_name, lang="en")
        if page_info:
            print(f"从英文维基百科找到角色: {character_name}")
            return page_info
        
        # 4. 所有数据源都未找到
        print(f"所有数据源都未找到角色: {character_name}")
        return None

    def analyze_height(self, text):
        """
        基于正则表达式的数值信息抽取：尝试抽取身高数据
        通常该类角色身高设定在 150cm 以下
        """
        height_pattern = r'身高[:：]?\s*约?(\d{2,3})\s*(?:cm|厘米)'
        match = re.search(height_pattern, text)
        if match:
            height = int(match.group(1))
            if height < 145:
                return 3.0  # 身高极矮，增加权重
            elif height < 150:
                return 1.5  # 身高偏矮，略微增加权重
            elif height > 165:
                return -3.0 # 身高较高，降低权重
        return 0.0

    def classify(self, character_name):
        """
        综合分析判断入口
        """
        print(f"正在分析角色: 【{character_name}】...")
        page_info = self.fetch_character_data(character_name)
        
        if not page_info:
            return "未在数据库中找到该角色，或角色名不准确。", [], "未找到角色"

        # 提取文本和分类标签（Metadata）
        extract_text = page_info.get("extract", "")
        categories = [cat["title"] for cat in page_info.get("categories", [])]
        
        # 整合所有文本信息用于分析
        full_text = extract_text + " ".join(categories)
        
        score = 0.0
        matched_tags = []

        # 1. 关键词特征匹配
        for keyword, weight in self.keywords_weights.items():
            if keyword in full_text:
                score += weight
                matched_tags.append(keyword)
                
        for keyword, weight in self.negative_keywords.items():
            if keyword in full_text:
                score += weight
                matched_tags.append(keyword + "(-)")

        # 2. 数值信息提取（身高）
        height_score = self.analyze_height(full_text)
        score += height_score
        if height_score > 0:
            matched_tags.append("身高偏矮")

        # 3. 综合判定逻辑 (阈值分类器)
        print(f"提取到的特征: {matched_tags if matched_tags else '无明显特征'}")
        print(f"综合特征得分: {score}")

        if score >= 4.0:
            return f"✅ 判定结果：【{character_name}】 属于 萝莉 分类。", matched_tags, "萝莉"
        elif score > 0:
            return f"❓ 判定结果：【{character_name}】 可能具有部分萝莉特征，但证据不足。", matched_tags, "可能具有部分萝莉特征"
        else:
            return f"❌ 判定结果：【{character_name}】 不属于 萝莉 分类。", matched_tags, "不属于萝莉"


def load_keywords_from_specific_file(file_path):
    """从指定文件中加载关键词"""
    keywords = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                keyword = line.strip()
                if keyword:
                    keywords.append(keyword)
    print(f"从 {file_path} 加载了 {len(keywords)} 个角色")
    return keywords
# 配置参数
AUTO_SPIDER_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img"
LOLIS_OUTPUT_DIR = os.path.join(AUTO_SPIDER_DIR, "lolis")


def load_all_characters_from_directory(directory_path):
    """从目录中加载所有txt文件中的角色
    
    Args:
        directory_path: 目录路径
        
    Returns:
        list: 角色列表
    """
    all_characters = []
    
    # 遍历目录中的所有txt文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            # 跳过lolis子目录中的文件
            if 'lolis' in filename:
                continue
                
            file_path = os.path.join(directory_path, filename)
            characters = load_keywords_from_file(file_path)
            all_characters.extend(characters)
    
    # 去重
    all_characters = list(set(all_characters))
    
    print(f"从 {directory_path} 加载了 {len(all_characters)} 个角色")
    return all_characters



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

# 测试脚本
if __name__ == "__main__":
    classifier = CharacterClassifier()
    
    # 加载loli_characters.txt文件中的角色
    file_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lolis/loli_characters.txt"
    characters = load_keywords_from_specific_file(file_path)
    
    if not characters:
        print("没有找到角色关键词")
        exit()
    
    # 分类结果统计
    results = {
        "萝莉": 0,
        "可能具有部分萝莉特征": 0,
        "不属于萝莉": 0,
        "未找到角色": 0
    }
    
    # 按分类存储角色
    classified_characters = {
        "萝莉": [],
        "可能具有部分萝莉特征": [],
        "不属于萝莉": [],
        "未找到角色": []
    }
    
    # 对每个角色进行分类
    for name in characters:
        result, tags, category = classifier.classify(name)
        print(result)
        print("-" * 40)
        
        # 统计结果
        results[category] += 1
        classified_characters[category].append((name, tags))
    
    # 输出统计结果
    print("=" * 80)
    print("分类结果统计:")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("=" * 80)
    
    # 将结果输出为txt文件
    output_dir = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lolis/classified"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 输出总结果
    with open(os.path.join(output_dir, "classification_results.txt"), "w", encoding="utf-8") as f:
        f.write("分类结果统计:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write("\n详细分类结果:\n")
        for category, chars in classified_characters.items():
            f.write(f"\n{category}:\n")
            for char_name, tags in chars:
                f.write(f"  {char_name} - 特征: {tags if tags else '无'}\n")
    
    # 按分类输出到不同文件
    for category, chars in classified_characters.items():
        if chars:
            with open(os.path.join(output_dir, f"{category}.txt"), "w", encoding="utf-8") as f:
                for char_name, tags in chars:
                    f.write(f"{char_name}\n")
    
    print(f"分类结果已输出到 {output_dir} 目录")