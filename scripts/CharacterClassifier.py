import requests
import re
import os

class CharacterClassifier:
    def __init__(self):
        # 萌娘百科 API 地址
        self.api_url = "https://zh.moegirl.org.cn/api.php"
        
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

    def fetch_character_data(self, character_name):
        """
        利用网络爬虫与API技术，获取角色的元数据和百科文本。
        """
        params = {
            "action": "query",
            "prop": "extracts|categories",
            "titles": character_name,
            "format": "json",
            "exintro": True,      # 只获取摘要部分
            "explaintext": True,  # 返回纯文本而非HTML
            "cllimit": "max"      # 获取最大数量的分类标签
        }
        
        try:
            response = requests.get(self.api_url, params=params, timeout=10)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            
            for page_id, page_info in pages.items():
                if page_id == "-1":
                    return None # 未找到该角色
                return page_info
        except Exception as e:
            print(f"网络请求错误: {e}")
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
            return "未在数据库中找到该角色，或角色名不准确。"

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
            return f"✅ 判定结果：【{character_name}】 属于 萝莉 分类。"
        elif score > 0:
            return f"❓ 判定结果：【{character_name}】 可能具有部分萝莉特征，但证据不足。"
        else:
            return f"❌ 判定结果：【{character_name}】 不属于 萝莉 分类。"


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

    # if choice == "1":
        # file_path = input("请输入文件路径（默认: /Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lolis/loli_characters.txt）: ").strip()
        # if not file_path:
        # file_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lolis/loli_characters.txt"
        # characters = load_keywords_from_specific_file(file_path)
    # elif choice == "2":
    test_characters = load_all_characters_from_directory(AUTO_SPIDER_DIR)
    # else:
    #     # 默认加载所有角色关键词
    #     characters = load_all_keywords()
    
    # if not characters:
    #     print("没有找到角色关键词")
    #     return
    
    
    # 你可以在这里替换为你想要测试的角色名
    # test_characters = ["可莉", "雷电将军", "伊莉雅丝菲尔·冯·爱因兹贝伦", "初音未来"]
    
    for name in test_characters:
        result = classifier.classify(name)
        print(result)
        print("-" * 40)