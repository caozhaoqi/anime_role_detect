"""
模型服务 - 分类器模块
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

import numpy as np
from src.core.logging.global_logger import get_logger

logger = get_logger("model_service.classifiers")


class EfficientNetClassifier:
    """EfficientNet直接分类器，不依赖Faiss索引"""

    _instance = None
    _model = None
    _class_to_idx = None
    _idx_to_class = None
    _transform = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        import torch
        import torchvision
        from torchvision import transforms
        import json

        if EfficientNetClassifier._model is not None:
            self.model = EfficientNetClassifier._model
            self.idx_to_class = EfficientNetClassifier._idx_to_class
            self.transform = EfficientNetClassifier._transform
            return

        model_dir = os.path.join(
            project_root, "models", "efficientnet_b3_loli_optimized_v2_20260529_133654"
        )
        model_full_path = os.path.join(model_dir, "model_full.pth")

        if not os.path.exists(model_full_path):
            logger.warning(f"EfficientNet模型不存在: {model_full_path}")
            self.model = None
            return

        logger.info(f"加载EfficientNet分类模型: {model_full_path}")

        torch.serialization.add_safe_globals(
            [torchvision.models.efficientnet.EfficientNet]
        )
        model = torch.load(model_full_path, map_location="cpu", weights_only=False)
        model.eval()

        training_results_path = os.path.join(model_dir, "training_results.json")
        with open(training_results_path, "r", encoding="utf-8") as f:
            training_results = json.load(f)
        class_names = training_results.get("class_names", [])
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        EfficientNetClassifier._model = model
        EfficientNetClassifier._idx_to_class = idx_to_class
        EfficientNetClassifier._transform = transform

        self.model = model
        self.idx_to_class = idx_to_class
        self.transform = transform

        logger.info(f"EfficientNet分类器加载完成，类别数: {len(class_to_idx)}")

    def classify(self, image):
        """直接使用EfficientNet模型分类"""
        import torch
        if self.model is None:
            return "unknown", 0.0
        try:
            input_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, 1)
            role = self.idx_to_class.get(predicted_idx.item(), "unknown")
            confidence = max_prob.item()
            return role, confidence
        except Exception as e:
            logger.error(f"EfficientNet分类失败: {e}")
            return "unknown", 0.0

    def classify_with_features(self, image):
        """分类并提取512维特征向量"""
        import torch
        if self.model is None:
            return "unknown", 0.0, np.zeros(512, dtype=np.float32)
        try:
            input_tensor = self.transform(image).unsqueeze(0)
            features = []
            def hook_fn(module, input, output):
                features.append(output)
            handle = self.model.avgpool.register_forward_hook(hook_fn)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, 1)
            handle.remove()
            role = self.idx_to_class.get(predicted_idx.item(), "unknown")
            confidence = max_prob.item()
            if features:
                feat_1536 = features[0].squeeze()
                if not hasattr(self, '_projection'):
                    projection = torch.nn.Linear(1536, 512, bias=False)
                    torch.nn.init.xavier_normal_(projection.weight)
                    self._projection = projection
                with torch.no_grad():
                    feat_512 = self._projection(feat_1536)
                    norm = feat_512.norm()
                    if norm > 0:
                        feat_512 = feat_512 / norm
                feature = feat_512.numpy().astype(np.float32)
            else:
                feature = np.zeros(512, dtype=np.float32)
            return role, confidence, feature
        except Exception as e:
            logger.error(f"EfficientNet分类+特征提取失败: {e}")
            return "unknown", 0.0, np.zeros(512, dtype=np.float32)