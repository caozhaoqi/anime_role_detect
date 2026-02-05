import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import logging

class EfficientNetInference:
    _instance = None
    
    def __new__(cls, model_path=None, data_dir=None):
        if cls._instance is None:
            cls._instance = super(EfficientNetInference, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, model_path=None, data_dir=None):
        if self.initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logging.info(f"EfficientNet推理使用设备: {self.device}")
        
        # 默认路径配置
        if model_path is None:
            # 尝试查找最新的模型
            base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
            candidates = [
                'character_classifier_best_improved.pth',
                'character_classifier_best_v2.pth',
                'character_classifier_best.pth'
            ]
            for cand in candidates:
                p = os.path.join(base_model_dir, cand)
                if os.path.exists(p):
                    model_path = p
                    break
            
            if model_path is None:
                raise FileNotFoundError("未找到预训练模型文件")

        if data_dir is None:
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'all_characters'))

        self.model_path = model_path
        self.data_dir = data_dir
        self.classes = self._load_classes()
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.initialized = True

    def _load_classes(self):
        """加载类别映射"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
            
        # 必须与训练时的排序逻辑一致
        classes = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        classes.sort()
        logging.info(f"加载了 {len(classes)} 个类别")
        return classes

    def _load_model(self):
        """加载模型结构和权重"""
        logging.info(f"加载模型: {self.model_path}")
        num_classes = len(self.classes)
        
        # 定义与训练时一致的模型结构
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # 加载权重
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # 处理可能的不同保存格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.info(f"模型加载失败: {e}")
            raise

        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        """预处理转换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, top_k=5):
        """预测图片角色"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
            # 获取Top-K结果
            top_prob, top_idx = torch.topk(probabilities, top_k)
            
            results = []
            for i in range(top_k):
                idx = top_idx[0][i].item()
                prob = top_prob[0][i].item()
                role_name = self.classes[idx]
                results.append({
                    "role": role_name,
                    "similarity": prob
                })
                
            # 返回最佳结果和完整列表
            best_role = results[0]["role"]
            best_score = results[0]["similarity"]
            
            return best_role, best_score, results
            
        except Exception as e:
            logging.info(f"推理失败: {e}")
            return None, 0.0, []

if __name__ == "__main__":
    # 测试代码
    try:
        infer = EfficientNetInference()
        logging.info("模型初始化成功")
        # 可以在这里添加简单的图片测试
    except Exception as e:
        logging.info(f"初始化失败: {e}")
