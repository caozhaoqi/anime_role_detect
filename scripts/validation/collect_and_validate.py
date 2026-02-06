#!/usr/bin/env python3
"""
采集不同角色数据并验证模型准确率
用于评估模型在不同角色上的表现
"""
import os
import sys
import json
import argparse
import logging
import time
import random
from datetime import datetime
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('collect_and_validate')

class DataCollector:
    """数据采集器"""
    
    def __init__(self, output_dir='data/validation_data', max_images_per_character=30):
        """初始化采集器
        
        Args:
            output_dir: 数据输出目录
            max_images_per_character: 每个角色的最大图片数量
        """
        self.output_dir = output_dir
        self.max_images_per_character = max_images_per_character
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'val'), exist_ok=True)
        
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        }
    
    def _init_session(self):
        """初始化会话"""
        try:
            import requests
            self.session = requests.Session()
            logger.info("HTTP会话初始化成功")
        except ImportError:
            logger.error("requests库未安装，请运行: pip install requests")
            raise
    
    def collect_character_images(self, character_name, series_name):
        """采集指定角色的图片
        
        Args:
            character_name: 角色名称
            series_name: 系列名称
            
        Returns:
            int: 采集的图片数量
        """
        if not self.session:
            self._init_session()
        
        # 创建角色目录
        character_dir = os.path.join(self.output_dir, series_name, character_name)
        os.makedirs(character_dir, exist_ok=True)
        
        logger.info(f"开始采集角色: {character_name} (系列: {series_name})")
        
        # 构建搜索查询
        search_query = f"{series_name} {character_name} 角色图片"
        
        image_count = 0
        page = 1
        
        while image_count < self.max_images_per_character and page <= 5:
            logger.info(f"搜索第 {page} 页")
            
            # 使用Bing图片搜索
            search_url = f"https://www.bing.com/images/search?q={search_query.replace(' ', '+')}&first={((page-1)*35)+1}"
            
            try:
                response = self.session.get(search_url, headers=self.headers, timeout=10)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"搜索失败: {e}")
                page += 1
                continue
            
            # 解析HTML
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                image_elements = soup.find_all('img', {'class': 'mimg'})
            except ImportError:
                logger.error("bs4库未安装，请运行: pip install beautifulsoup4")
                raise
            
            if not image_elements:
                logger.warning(f"第 {page} 页未找到图片")
                page += 1
                continue
            
            # 下载图片
            for img_element in image_elements:
                if image_count >= self.max_images_per_character:
                    break
                
                img_url = img_element.get('src') or img_element.get('data-src')
                if not img_url:
                    continue
                
                # 确保URL完整
                if not img_url.startswith('http'):
                    img_url = f"https://www.bing.com{img_url}"
                
                try:
                    # 下载图片
                    img_response = self.session.get(img_url, headers=self.headers, timeout=10)
                    img_response.raise_for_status()
                    
                    # 验证图片
                    from PIL import Image
                    from io import BytesIO
                    image = Image.open(BytesIO(img_response.content))
                    image.verify()
                    
                    # 保存图片
                    img_filename = f"{series_name}_{character_name}_{image_count:04d}.jpg"
                    img_path = os.path.join(character_dir, img_filename)
                    with open(img_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    image_count += 1
                    logger.info(f"已下载 {image_count:04d}/{self.max_images_per_character:04d} 张图片")
                    
                    # 随机延迟，避免被封禁
                    time.sleep(random.uniform(1.0, 3.0))
                    
                except Exception as e:
                    logger.error(f"下载图片失败: {e}")
                    continue
            
            page += 1
            time.sleep(random.uniform(2.0, 5.0))
        
        logger.info(f"角色 {character_name} 图片采集完成，共下载 {image_count} 张图片")
        return image_count

class ModelValidator:
    """模型验证器"""
    
    def __init__(self, model_path, data_dir, threshold=0.5):
        """初始化验证器
        
        Args:
            model_path: 模型路径
            data_dir: 数据目录
            threshold: 检测阈值
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.threshold = threshold
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        logger.info(f"模型路径: {model_path}")
        logger.info(f"数据目录: {data_dir}")
        logger.info(f"检测阈值: {threshold}")
    
    def validate_model(self, test_images):
        """验证模型准确率
        
        Args:
            test_images: 测试图像列表
            
        Returns:
            dict: 验证结果
        """
        logger.info("开始验证模型准确率...")
        
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            from PIL import Image
            import numpy as np
            from torchvision import transforms
        except ImportError as e:
            logger.error(f"缺少必要的库: {e}")
            raise
        
        # 加载模型
        logger.info("加载模型...")
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # 获取类别信息
        if 'class_to_idx' in checkpoint:
            class_to_idx = checkpoint['class_to_idx']
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            num_classes = len(class_to_idx)
        else:
            logger.error("模型中未找到class_to_idx信息")
            return {'error': '模型中未找到class_to_idx信息'}
        
        logger.info(f"模型包含 {num_classes} 个类别")
        
        # 创建模型
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理键名不匹配
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                name = k[9:]  # 移除 'backbone.'
            else:
                name = k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        logger.info("模型加载成功")
        
        # 定义图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 验证测试图像
        results = []
        correct_count = 0
        total_count = 0
        
        for test_image_path in test_images:
            if not os.path.exists(test_image_path):
                logger.warning(f"测试图像不存在: {test_image_path}")
                continue
            
            try:
                # 加载并预处理图像
                image = Image.open(test_image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                
                # 预测
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_idx].item()
                
                # 获取预测的角色名
                predicted_role = idx_to_class.get(predicted_idx, f"类别_{predicted_idx}")
                
                # 从文件名提取真实角色
                filename = os.path.basename(test_image_path)
                # 假设文件名格式为 series_character_count.jpg
                # 注意：series可能包含下划线（如honkai_star_rail）
                parts = filename.split('_')
                if len(parts) >= 4:
                    # 对于格式如 honkai_star_rail_火花_0004.jpg
                    # 角色名是倒数第二个部分
                    true_role = parts[-2]
                elif len(parts) >= 3:
                    # 对于格式如 series_character_0001.jpg
                    true_role = parts[1]
                else:
                    true_role = "unknown"
                
                # 判断是否正确
                is_correct = (predicted_role == true_role)
                if is_correct:
                    correct_count += 1
                
                total_count += 1
                
                result = {
                    'image_path': test_image_path,
                    'predicted_role': predicted_role,
                    'true_role': true_role,
                    'confidence': confidence,
                    'is_correct': is_correct
                }
                results.append(result)
                
                logger.info(f"图像: {filename} | 预测: {predicted_role} | 真实: {true_role} | 置信度: {confidence:.4f} | 正确: {is_correct}")
                
            except Exception as e:
                logger.error(f"处理图像 {test_image_path} 失败: {e}")
                continue
        
        # 计算准确率
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # 统计预测分布
        predictions = [r['predicted_role'] for r in results]
        prediction_counts = Counter(predictions)
        
        # 生成验证报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'total_images': total_count,
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'prediction_distribution': dict(prediction_counts),
            'results': results
        }
        
        logger.info(f"验证完成: 总计 {total_count} 张，正确 {correct_count} 张，准确率: {accuracy:.4f}")
        
        return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='采集不同角色数据并验证模型准确率')
    
    parser.add_argument('--characters', type=str, required=True,
                       help='角色列表JSON字符串，格式: [{"name": "角色名", "series": "系列名"}]')
    parser.add_argument('--output_dir', type=str, default='data/validation_data',
                       help='数据输出目录')
    parser.add_argument('--max_images', type=int, default=30,
                       help='每个角色的最大图片数量')
    parser.add_argument('--model_path', type=str, 
                       default='models/character_classifier_best_improved.pth',
                       help='模型文件路径')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='检测阈值')
    
    args = parser.parse_args()
    
    # 解析角色列表
    try:
        characters = json.loads(args.characters)
    except json.JSONDecodeError as e:
        logger.error(f"角色列表JSON格式错误: {e}")
        sys.exit(1)
    
    # 验证角色格式
    for i, char in enumerate(characters):
        if not isinstance(char, dict) or 'name' not in char or 'series' not in char:
            logger.error(f"角色 {i+1} 格式错误，需要包含 name 和 series 字段")
            sys.exit(1)
    
    logger.info(f"准备采集 {len(characters)} 个角色的数据")
    
    # 步骤1: 采集数据
    collector = DataCollector(
        output_dir=args.output_dir,
        max_images_per_character=args.max_images
    )
    
    total_images = 0
    for character in characters:
        image_count = collector.collect_character_images(
            character['name'],
            character['series']
        )
        total_images += image_count
        time.sleep(random.uniform(2.0, 5.0))
    
    logger.info(f"数据采集完成，共下载 {total_images} 张图片")
    
    # 步骤2: 准备测试图像
    logger.info("准备测试图像...")
    test_images = []
    
    for character in characters:
        character_dir = os.path.join(args.output_dir, character['series'], character['name'])
        if os.path.exists(character_dir):
            # 获取该角色目录下的所有图片
            image_files = [f for f in os.listdir(character_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # 随机选择一部分作为测试集
            test_count = max(1, len(image_files) // 5)  # 使用20%作为测试集
            test_files = random.sample(image_files, test_count)
            test_images.extend([os.path.join(character_dir, f) for f in test_files])
    
    logger.info(f"准备了 {len(test_images)} 张测试图像")
    
    # 步骤3: 验证模型
    validator = ModelValidator(
        model_path=args.model_path,
        data_dir=args.output_dir,
        threshold=args.threshold
    )
    
    validation_report = validator.validate_model(test_images)
    
    # 步骤4: 保存验证报告
    report_path = os.path.join(args.output_dir, f'validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"验证报告已保存: {report_path}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("数据采集和模型验证完成")
    print("="*60)
    print(f"采集角色数: {len(characters)}")
    print(f"总图片数: {total_images}")
    print(f"测试图像数: {len(test_images)}")
    print(f"正确预测数: {validation_report['correct_predictions']}")
    print(f"准确率: {validation_report['accuracy']:.4f}")
    print("="*60)
    print(f"\n详细报告已保存到: {report_path}")
    print(f"\n预测分布:")
    from collections import Counter
    prediction_counter = Counter(validation_report['prediction_distribution'])
    for role, count in prediction_counter.most_common(10):
        print(f"  {role}: {count} 次")

if __name__ == '__main__':
    main()
