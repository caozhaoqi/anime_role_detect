#!/usr/bin/env python3
"""
基于度量学习的角色识别推理
演示如何识别训练时没有的角色
"""
import os
import sys
import argparse
import logging
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('metric_learning_inference')


class MetricLearningModel(nn.Module):
    """度量学习模型"""
    
    def __init__(self, embedding_dim=512, num_classes=None):
        """初始化模型
        
        Args:
            embedding_dim: 特征嵌入维度
            num_classes: 类别数量
        """
        super(MetricLearningModel, self).__init__()
        
        # 使用EfficientNet作为骨干网络
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # 移除最后的分类层
        self.backbone.classifier = nn.Identity()
        
        # 添加投影层
        self.projection = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 分类层（可选）
        self.num_classes = num_classes
        if num_classes:
            self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像
            
        Returns:
            embedding: 特征嵌入
            logits: 分类logits（如果num_classes不为None）
        """
        features = self.backbone(x)
        embedding = self.projection(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        if self.num_classes:
            logits = self.fc(embedding)
            return embedding, logits
        
        return embedding
    
    def extract_feature(self, x):
        """提取特征
        
        Args:
            x: 输入图像
            
        Returns:
            embedding: 特征嵌入
        """
        self.eval()
        with torch.no_grad():
            embedding = self.forward(x)
        return embedding


class CharacterRecognizer:
    """角色识别器"""
    
    def __init__(self, model_path, threshold=0.65):
        """初始化识别器
        
        Args:
            model_path: 模型路径
            threshold: 相似度阈值
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.device = None
        self.transform = None
        self.character_database = {}
        
        self._load_model()
        self._init_transform()
    
    def _load_model(self):
        """加载模型"""
        # 检测设备
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取模型参数
        embedding_dim = checkpoint.get('embedding_dim', 512)
        class_to_idx = checkpoint.get('class_to_idx', {})
        
        # 初始化模型
        num_classes = len(class_to_idx) if class_to_idx else None
        self.model = MetricLearningModel(embedding_dim=embedding_dim, num_classes=num_classes)
        
        # 加载权重
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"模型加载成功: {self.model_path}")
        logger.info(f"特征嵌入维度: {embedding_dim}")
        if class_to_idx:
            logger.info(f"模型包含 {len(class_to_idx)} 个训练类别")
    
    def _init_transform(self):
        """初始化图像变换"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def add_character_to_database(self, character_name, image_path):
        """添加角色到特征库
        
        Args:
            character_name: 角色名称
            image_path: 角色参考图像路径
        """
        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 提取特征
            embedding = self.model.extract_feature(image_tensor).cpu().numpy()[0]
            
            # 添加到数据库
            self.character_database[character_name] = embedding
            logger.info(f"角色 '{character_name}' 已添加到特征库")
            
        except Exception as e:
            logger.error(f"添加角色失败: {e}")
    
    def add_multiple_characters(self, characters_dir):
        """批量添加角色到特征库
        
        Args:
            characters_dir: 角色图像目录，结构为 characters_dir/角色名/参考图像.jpg
        """
        for character_name in os.listdir(characters_dir):
            character_path = os.path.join(characters_dir, character_name)
            if os.path.isdir(character_path):
                # 选择第一张图像作为参考
                image_files = [f for f in os.listdir(character_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if image_files:
                    image_path = os.path.join(character_path, image_files[0])
                    self.add_character_to_database(character_name, image_path)
    
    def recognize(self, image_path):
        """识别角色
        
        Args:
            image_path: 待识别图像路径
            
        Returns:
            dict: 识别结果
        """
        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 提取特征
            embedding = self.model.extract_feature(image_tensor).cpu().numpy()[0]
            
            # 计算与特征库中所有角色的相似度
            if not self.character_database:
                return {
                    'character': 'Unknown Character (特征库为空)',
                    'similarity': 0.0,
                    'is_known': False
                }
            
            similarities = {}
            for character_name, char_embedding in self.character_database.items():
                # 计算余弦相似度
                similarity = np.dot(embedding, char_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(char_embedding)
                )
                similarities[character_name] = similarity
            
            # 找到最相似的角色
            most_similar_character = max(similarities, key=similarities.get)
            highest_similarity = similarities[most_similar_character]
            
            # 判断是否为已知角色
            if highest_similarity >= self.threshold:
                result = {
                    'character': most_similar_character,
                    'similarity': highest_similarity,
                    'is_known': True
                }
            else:
                result = {
                    'character': 'Unknown Character (新角色)',
                    'similarity': highest_similarity,
                    'is_known': False
                }
            
            # 记录详细信息
            logger.info(f"识别结果: {result['character']}")
            logger.info(f"相似度: {result['similarity']:.4f}")
            logger.info(f"是否已知: {result['is_known']}")
            
            # 记录前3个最相似的角色
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info("前3个最相似的角色:")
            for char, sim in sorted_similarities:
                logger.info(f"  {char}: {sim:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"识别失败: {e}")
            return {
                'character': 'Error',
                'similarity': 0.0,
                'is_known': False,
                'error': str(e)
            }
    
    def batch_recognize(self, images_dir):
        """批量识别图像
        
        Args:
            images_dir: 图像目录
            
        Returns:
            list: 识别结果列表
        """
        results = []
        
        for image_name in os.listdir(images_dir):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(images_dir, image_name)
                logger.info(f"识别图像: {image_name}")
                result = self.recognize(image_path)
                result['image_name'] = image_name
                results.append(result)
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于度量学习的角色识别')
    
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--image_path', type=str, help='待识别图像路径')
    parser.add_argument('--images_dir', type=str, help='待识别图像目录')
    parser.add_argument('--characters_dir', type=str, help='角色参考图像目录')
    parser.add_argument('--threshold', type=float, default=0.65, help='相似度阈值')
    
    args = parser.parse_args()
    
    # 初始化识别器
    recognizer = CharacterRecognizer(model_path=args.model_path, threshold=args.threshold)
    
    # 添加角色到特征库
    if args.characters_dir:
        logger.info(f"从目录加载角色: {args.characters_dir}")
        recognizer.add_multiple_characters(args.characters_dir)
    else:
        # 如果没有提供角色目录，使用默认角色
        logger.info("使用默认角色特征库")
        # 这里可以添加一些默认角色
        # 例如：recognizer.add_character_to_database('雷电将军', 'path/to/raiden.jpg')
    
    # 识别单个图像
    if args.image_path:
        logger.info(f"识别单个图像: {args.image_path}")
        result = recognizer.recognize(args.image_path)
        
        print("\n" + "="*60)
        print("识别结果")
        print("="*60)
        print(f"图像: {os.path.basename(args.image_path)}")
        print(f"角色: {result['character']}")
        print(f"相似度: {result['similarity']:.4f}")
        print(f"是否已知: {result['is_known']}")
        print("="*60)
    
    # 批量识别图像
    elif args.images_dir:
        logger.info(f"批量识别图像目录: {args.images_dir}")
        results = recognizer.batch_recognize(args.images_dir)
        
        print("\n" + "="*60)
        print("批量识别结果")
        print("="*60)
        for result in results:
            print(f"图像: {result['image_name']}")
            print(f"角色: {result['character']}")
            print(f"相似度: {result['similarity']:.4f}")
            print(f"是否已知: {result['is_known']}")
            print("-"*60)
    
    else:
        logger.error("请提供 --image_path 或 --images_dir 参数")
        sys.exit(1)


if __name__ == "__main__":
    # 导入必要的库
    import torch
    import torch.nn as nn
    from torchvision import models
    
    main()
