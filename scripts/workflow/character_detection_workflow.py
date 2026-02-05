#!/usr/bin/env python3
"""
角色检测工作流脚本
端到端实现：输入角色 → 采集数据 → 训练模型 → 检测角色 → 输出结果
"""
import os
import sys
import time
import argparse
import logging
import subprocess
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('character_detection_workflow')

class CharacterDetectionWorkflow:
    """角色检测工作流"""
    
    def __init__(self, output_dir='data/all_characters', model_dir='models', max_images=50):
        """初始化工作流
        
        Args:
            output_dir: 数据输出目录
            model_dir: 模型保存目录
            max_images: 每个角色的最大图片数量
        """
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.max_images = max_images
        
        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs('data/split_dataset/train', exist_ok=True)
        os.makedirs('data/split_dataset/val', exist_ok=True)
    
    def collect_data(self, characters):
        """采集角色数据
        
        Args:
            characters: 角色列表，每个元素是包含name和series的字典
        """
        logger.info(f"开始采集 {len(characters)} 个角色的数据")
        
        # 创建采集脚本
        crawl_script = '''#!/usr/bin/env python3
"""
临时数据采集脚本
"""
import os
import time
import random
import logging
import json
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('temp_crawler')

class TempImageCrawler:
    """临时图片采集器"""
    
    def __init__(self, output_dir, max_images):
        """初始化采集器
        
        Args:
            output_dir: 输出目录
            max_images: 每个角色的最大图片数量
        """
        self.output_dir = output_dir
        self.max_images = max_images
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        }
    
    def crawl_images(self, character_name, series):
        """采集指定角色的图片
        
        Args:
            character_name: 角色名称
            series: 系列名称
        """
        # 创建角色目录
        character_dir = os.path.join(self.output_dir, f'{series}_{character_name}')
        os.makedirs(character_dir, exist_ok=True)
        
        # 构建搜索查询
        search_query = f"{series} {character_name} 角色图片"
        logger.info(f"开始采集角色: {character_name} (系列: {series})")
        
        # 采集图片
        image_count = 0
        page = 1
        
        while image_count < self.max_images and page <= 5:
            logger.info(f"搜索第 {page} 页")
            
            # 使用Bing图片搜索
            search_url = f"https://www.bing.com/images/search?q={{}}&first={{}}".format(
                search_query.replace(' ', '+'),
                ((page-1)*35)+1
            )
            
            try:
                response = self.session.get(search_url, headers=self.headers, timeout=10)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"搜索失败: {{e}}")
                page += 1
                continue
            
            # 解析HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 查找图片元素
            image_elements = soup.find_all('img', {'class': 'mimg'})
            
            if not image_elements:
                logger.warning(f"第 {{page}} 页未找到图片")
                page += 1
                continue
            
            # 下载图片
            for img_element in image_elements:
                if image_count >= self.max_images:
                    break
                
                img_url = img_element.get('src') or img_element.get('data-src')
                if not img_url:
                    continue
                
                # 确保URL完整
                if not img_url.startswith('http'):
                    img_url = f"https://www.bing.com{{img_url}}"
                
                try:
                    # 下载图片
                    img_response = self.session.get(img_url, headers=self.headers, timeout=10)
                    img_response.raise_for_status()
                    
                    # 验证图片
                    image = Image.open(BytesIO(img_response.content))
                    image.verify()
                    
                    # 保存图片
                    img_path = os.path.join(character_dir, f"{series}_{character_name}_{image_count}_{int(time.time())}.jpg")
                    with open(img_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    image_count += 1
                    logger.info(f"已下载 {{image_count}}/{{self.max_images}} 张图片")
                    
                    # 随机延迟
                    time.sleep(random.uniform(0.5, 2.0))
                    
                except Exception as e:
                    logger.error(f"下载图片失败: {{e}}")
                    continue
            
            page += 1
        
        logger.info(f"角色 {{character_name}} 图片采集完成，共下载 {{image_count}} 张图片")
        return image_count

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='临时数据采集脚本')
    parser.add_argument('--characters', type=str, required=True, help='角色列表JSON字符串')
    parser.add_argument('--output_dir', type=str, default='data/all_characters', help='输出目录')
    parser.add_argument('--max_images', type=int, default=50, help='每个角色的最大图片数量')
    
    args = parser.parse_args()
    
    # 解析角色列表
    characters = json.loads(args.characters)
    
    # 创建采集器
    crawler = TempImageCrawler(args.output_dir, args.max_images)
    
    # 采集所有角色
    total_images = 0
    for character in characters:
        image_count = crawler.crawl_images(character['name'], character['series'])
        total_images += image_count
        
        # 每个角色之间的延迟
        time.sleep(random.uniform(2.0, 5.0))
    
    logger.info(f"所有角色图片采集完成，共下载 {{total_images}} 张图片")
    print(f"{{total_images}}")

if __name__ == "__main__":
    main()
'''
        
        # 保存临时采集脚本
        temp_crawl_script = 'scripts/workflow/temp_crawl_script.py'
        with open(temp_crawl_script, 'w') as f:
            f.write(crawl_script)
        
        # 执行采集脚本
        characters_json = json.dumps(characters)
        cmd = [
            sys.executable, temp_crawl_script,
            '--characters', characters_json,
            '--output_dir', self.output_dir,
            '--max_images', str(self.max_images)
        ]
        
        logger.info(f"执行采集脚本: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            total_images = int(result.stdout.strip())
            logger.info(f"采集完成，共下载 {total_images} 张图片")
        except subprocess.CalledProcessError as e:
            logger.error(f"采集脚本执行失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            return False
        finally:
            # 删除临时脚本
            if os.path.exists(temp_crawl_script):
                os.remove(temp_crawl_script)
        
        return True
    
    def split_dataset(self):
        """分割数据集"""
        logger.info("开始分割数据集...")
        
        # 执行分割脚本
        cmd = [
            sys.executable, 'scripts/data_processing/split_dataset.py'
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("数据集分割完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"分割脚本执行失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            return False
        
        return True
    
    def train_model(self, batch_size=16, num_epochs=50, learning_rate=5e-5, num_workers=4):
        """训练模型
        
        Args:
            batch_size: 批量大小
            num_epochs: 训练轮数
            learning_rate: 学习率
            num_workers: 数据加载线程数
        """
        logger.info("开始训练模型...")
        
        # 执行训练脚本
        cmd = [
            sys.executable, 'scripts/model_training/train_model_improved.py',
            '--batch_size', str(batch_size),
            '--num_epochs', str(num_epochs),
            '--learning_rate', str(learning_rate),
            '--num_workers', str(num_workers)
        ]
        
        logger.info(f"执行训练脚本: {' '.join(cmd)}")
        
        try:
            # 训练可能需要较长时间，使用非阻塞模式
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 实时输出训练日志
            while process.poll() is None:
                line = process.stdout.readline()
                if line:
                    logger.info(line.strip())
                
                error_line = process.stderr.readline()
                if error_line:
                    logger.error(error_line.strip())
                
                # 避免CPU占用过高
                time.sleep(0.1)
            
            # 读取剩余输出
            stdout, stderr = process.communicate()
            if stdout:
                logger.info(stdout.strip())
            if stderr:
                logger.error(stderr.strip())
            
            if process.returncode == 0:
                logger.info("模型训练完成")
                return True
            else:
                logger.error(f"训练脚本执行失败，返回码: {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"训练脚本执行失败: {e}")
            return False
    
    def detect_character(self, image_path, model_path=None, threshold=0.5, multiple=False, grid_size=3):
        """检测角色
        
        Args:
            image_path: 图像路径
            model_path: 模型路径
            threshold: 检测阈值
            multiple: 是否检测多个角色
            grid_size: 网格大小
        """
        logger.info(f"开始检测角色: {image_path}")
        
        # 如果没有指定模型路径，使用默认路径
        if not model_path:
            model_path = os.path.join(self.model_dir, 'character_classifier_best_improved.pth')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        # 执行检测脚本
        cmd = [
            sys.executable, 'scripts/character_detection/detect_all_characters.py',
            '--model_path', model_path,
            '--image_path', image_path,
            '--threshold', str(threshold)
        ]
        
        if multiple:
            cmd.extend(['--multiple', '--grid_size', str(grid_size)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("角色检测完成")
            logger.info(f"检测结果: {result.stdout}")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"检测脚本执行失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            return False
    
    def run_workflow(self, characters, test_image_path, batch_size=16, num_epochs=50, learning_rate=5e-5, num_workers=4, threshold=0.5, multiple=False, grid_size=3):
        """运行完整工作流
        
        Args:
            characters: 角色列表
            test_image_path: 测试图像路径
            batch_size: 批量大小
            num_epochs: 训练轮数
            learning_rate: 学习率
            num_workers: 数据加载线程数
            threshold: 检测阈值
            multiple: 是否检测多个角色
            grid_size: 网格大小
        """
        logger.info("开始运行角色检测工作流...")
        
        # 步骤1: 采集数据
        logger.info("步骤1: 采集角色数据")
        if not self.collect_data(characters):
            logger.error("数据采集失败，工作流终止")
            return False
        
        # 步骤2: 分割数据集
        logger.info("步骤2: 分割数据集")
        if not self.split_dataset():
            logger.error("数据集分割失败，工作流终止")
            return False
        
        # 步骤3: 训练模型
        logger.info("步骤3: 训练分类模型")
        if not self.train_model(batch_size, num_epochs, learning_rate, num_workers):
            logger.error("模型训练失败，工作流终止")
            return False
        
        # 步骤4: 检测角色
        logger.info("步骤4: 检测角色")
        if not self.detect_character(test_image_path, threshold=threshold, multiple=multiple, grid_size=grid_size):
            logger.error("角色检测失败，工作流终止")
            return False
        
        logger.info("角色检测工作流运行完成！")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='角色检测工作流脚本')
    
    # 工作流参数
    parser.add_argument('--characters', type=str, required=True, help='角色列表JSON字符串，格式: [{"name": "角色名", "series": "系列名"}]')
    parser.add_argument('--test_image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--output_dir', type=str, default='data/all_characters', help='数据输出目录')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--max_images', type=int, default=50, help='每个角色的最大图片数量')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 检测参数
    parser.add_argument('--threshold', type=float, default=0.5, help='检测阈值')
    parser.add_argument('--multiple', action='store_true', help='是否检测多个角色')
    parser.add_argument('--grid_size', type=int, default=3, help='网格大小')
    
    args = parser.parse_args()
    
    # 解析角色列表
    try:
        characters = json.loads(args.characters)
    except json.JSONDecodeError as e:
        logger.error(f"角色列表JSON格式错误: {e}")
        sys.exit(1)
    
    # 检查测试图像是否存在
    if not os.path.exists(args.test_image):
        logger.error(f"测试图像不存在: {args.test_image}")
        sys.exit(1)
    
    # 创建工作流实例
    workflow = CharacterDetectionWorkflow(
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        max_images=args.max_images
    )
    
    # 运行工作流
    success = workflow.run_workflow(
        characters=characters,
        test_image_path=args.test_image,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        threshold=args.threshold,
        multiple=args.multiple,
        grid_size=args.grid_size
    )
    
    if success:
        logger.info("工作流运行成功！")
        sys.exit(0)
    else:
        logger.error("工作流运行失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
