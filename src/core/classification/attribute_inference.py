import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
from collections import OrderedDict
import numpy as np
import time

class CharacterAttributeModel(nn.Module):
    """带有属性预测分支的角色分类模型"""
    def __init__(self, num_classes, num_attributes, base_model='mobilenet_v2'):
        super(CharacterAttributeModel, self).__init__()
        
        # 基础模型
        if base_model == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=False)
            self.base_model.classifier = nn.Identity()
            self.feature_dim = 1280
        elif base_model == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=False)
            self.base_model.classifier = nn.Identity()
            self.feature_dim = 1280
        elif base_model == 'resnet18':
            self.base_model = models.resnet18(pretrained=False)
            self.base_model.fc = nn.Identity()
            self.feature_dim = 512
        else:
            raise ValueError(f"不支持的基础模型: {base_model}")
        
        # 分类头部
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # 属性预测头部
        self.attribute_classifier = nn.Linear(self.feature_dim, num_attributes)
    
    def forward(self, x):
        features = self.base_model(x)
        class_output = self.classifier(features)
        attribute_output = self.attribute_classifier(features)
        return class_output, attribute_output

class AttributeInference:
    _instance = None
    
    def __new__(cls, model_path=None, enable_optimizations=True):
        if cls._instance is None:
            cls._instance = super(AttributeInference, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, model_path=None, enable_optimizations=True):
        if self.initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.enable_optimizations = enable_optimizations
        print(f"AttributeInference使用设备: {self.device}")
        print(f"启用优化: {self.enable_optimizations}")
        
        # 默认路径配置
        if model_path is None:
            # 尝试查找带有属性的模型
            base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
            
            # 优先查找带有属性的模型
            attribute_model_path = os.path.join(base_model_dir, 'character_classifier_with_attributes', 'model_best.pth')
            if os.path.exists(attribute_model_path):
                model_path = attribute_model_path
                print(f"使用带有属性预测的模型: {model_path}")
            else:
                raise FileNotFoundError("未找到带有属性预测的模型文件")

        self.model_path = model_path
        self.classes = self._load_classes()
        self.tags = self._load_tags()
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.initialized = True
        
        # 性能统计
        self.inference_times = []

    def _load_classes(self):
        """加载类别映射"""
        # 1. 首先尝试从模型文件中加载class_to_idx
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'class_to_idx' in checkpoint:
                class_to_idx = checkpoint['class_to_idx']
                # 转换为按索引排序的类别列表
                idx_to_class = {v: k for k, v in class_to_idx.items()}
                classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
                print(f"从模型文件加载了 {len(classes)} 个类别")
                return classes
        except Exception as e:
            print(f"从模型文件加载类别失败: {e}")
        
        # 2. 尝试从class_to_idx.json文件加载
        class_to_idx_path = os.path.join(os.path.dirname(self.model_path), "class_to_idx.json")
        if os.path.exists(class_to_idx_path):
            try:
                with open(class_to_idx_path, 'r', encoding='utf-8') as f:
                    class_to_idx = json.load(f)
                # 转换为按索引排序的类别列表
                idx_to_class = {int(v): k for k, v in class_to_idx.items()}
                classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
                print(f"从class_to_idx.json文件加载了 {len(classes)} 个类别")
                return classes
            except Exception as e:
                print(f"从class_to_idx.json文件加载类别失败: {e}")
        
        raise FileNotFoundError("无法加载类别映射")

    def _load_tags(self):
        """加载标签映射"""
        # 尝试从tag_to_idx.json文件加载
        tag_to_idx_path = os.path.join(os.path.dirname(self.model_path), "tag_to_idx.json")
        if os.path.exists(tag_to_idx_path):
            try:
                with open(tag_to_idx_path, 'r', encoding='utf-8') as f:
                    tag_to_idx = json.load(f)
                # 转换为按索引排序的标签列表
                idx_to_tag = {int(v): k for k, v in tag_to_idx.items()}
                tags = [idx_to_tag[i] for i in sorted(idx_to_tag.keys())]
                print(f"从tag_to_idx.json文件加载了 {len(tags)} 个标签")
                return tags
            except Exception as e:
                print(f"从tag_to_idx.json文件加载标签失败: {e}")
        
        raise FileNotFoundError("无法加载标签映射")

    def _load_model(self):
        """加载模型结构和权重"""
        print(f"加载模型: {self.model_path}")
        num_classes = len(self.classes)
        num_attributes = len(self.tags)
        
        # 加载模型结构
        model = CharacterAttributeModel(num_classes, num_attributes)
        
        # 加载权重
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 处理可能的不同保存格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 修复键名不匹配问题
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # 处理不同模型的键名前缀
                if k.startswith('backbone.'):
                    name = k[9:] # 移除 'backbone.'
                elif k.startswith('module.'):
                    name = k[7:] # 移除 'module.'
                else:
                    name = k
                new_state_dict[name] = v
                
            # 尝试加载权重，允许非严格匹配（如果还有其他微小差异）
            model.load_state_dict(new_state_dict, strict=False)
            print("模型权重加载成功 (已处理键名不匹配)")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

        model = model.to(self.device)
        model.eval()
        
        # 应用优化
        if self.enable_optimizations:
            model = self._optimize_model(model)
        
        return model

    def _optimize_model(self, model):
        """优化模型以提高推理速度"""
        print("应用模型优化...")
        
        # 1. 启用FP16精度（如果支持）
        if torch.cuda.is_available():
            model = model.half()
            print("启用FP16精度")
        
        return model

    def _get_transforms(self):
        """预处理转换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path, top_k=5, attribute_threshold=0.5):
        """预测图片角色和属性
        
        Args:
            image_path: 图片路径
            top_k: 返回前k个角色结果
            attribute_threshold: 属性预测阈值
            
        Returns:
            (best_role, best_score, results, attributes): 最佳角色、最佳相似度、所有角色结果、预测的属性
        """
        start_time = time.time()
        
        try:
            # 预测图片角色和属性
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 如果启用了FP16，转换输入
            if self.enable_optimizations and torch.cuda.is_available():
                image_tensor = image_tensor.half()
            
            with torch.no_grad():
                class_output, attribute_output = self.model(image_tensor)
                class_probabilities = torch.nn.functional.softmax(class_output, dim=1)
                attribute_probabilities = torch.sigmoid(attribute_output)
            
            # 获取Top-K角色结果
            k = min(top_k, len(self.classes))
            top_prob, top_idx = torch.topk(class_probabilities, k)
            
            results = []
            for i in range(k):
                idx = top_idx[0][i].item()
                prob = top_prob[0][i].item()
                if idx < len(self.classes):
                    role_name = self.classes[idx]
                    results.append({
                        "role": role_name,
                        "similarity": prob
                    })
            
            # 获取属性预测结果
            attributes = []
            attribute_probs = attribute_probabilities[0].cpu().numpy()
            for i, prob in enumerate(attribute_probs):
                if prob > attribute_threshold:
                    if i < len(self.tags):
                        attribute_name = self.tags[i]
                        attributes.append({
                            "tag": attribute_name,
                            "confidence": float(prob)
                        })
            
            # 按置信度排序属性
            attributes.sort(key=lambda x: x['confidence'], reverse=True)
            
            if not results:
                return "Unknown", 0.0, [], []
                
            # 返回最佳结果和完整列表
            best_role = results[0]["role"]
            best_score = results[0]["similarity"]
            
            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) % 100 == 0:
                avg_time = sum(self.inference_times) / len(self.inference_times)
                print(f"平均推理时间: {avg_time:.4f}秒")
            
            return best_role, best_score, results, attributes
            
        except Exception as e:
            print(f"推理失败: {e}")
            return None, 0.0, [], []

    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.inference_times:
            return {
                "total_inferences": 0,
                "average_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0
            }
        
        return {
            "total_inferences": len(self.inference_times),
            "average_time": sum(self.inference_times) / len(self.inference_times),
            "min_time": min(self.inference_times),
            "max_time": max(self.inference_times)
        }

    def clear_stats(self):
        """清除性能统计信息"""
        self.inference_times = []
        print("性能统计信息已清除")

if __name__ == "__main__":
    # 测试代码
    try:
        infer = AttributeInference()
        print("模型初始化成功")
    except Exception as e:
        print(f"初始化失败: {e}")
