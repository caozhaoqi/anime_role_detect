"""
模型服务 - 分类器模块
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
from src.core.logging.global_logger import get_logger
from src.core.config.device_manager import DeviceManager

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
            self._device = DeviceManager.get_device()
            return

        model_dir = os.path.join(
            project_root, "models", "efficientnet_b3_loli_optimized_v2_20260529_133654"
        )
        model_full_path = os.path.join(model_dir, "model_full.pth")

        if not os.path.exists(model_full_path):
            logger.warning(f"EfficientNet模型不存在: {model_full_path}")
            self.model = None
            self._device = DeviceManager.get_device()
            return

        logger.info(f"加载EfficientNet分类模型: {model_full_path}")

        # P0-1: 使用 DeviceManager 检测设备
        device = DeviceManager.get_device()
        self._device = device

        torch.serialization.add_safe_globals(
            [torchvision.models.efficientnet.EfficientNet]
        )
        model = torch.load(model_full_path, map_location=device, weights_only=False)
        model = model.to(device)
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
            input_tensor = self.transform(image).unsqueeze(0).to(self._device)
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
            input_tensor = self.transform(image).unsqueeze(0).to(self._device)
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
                    # P2-7: 尝试从训练好的权重文件加载投影层
                    projection_weights_path = os.path.join(
                        project_root, "models", "efficientnet_b3", "projection_weights.pth"
                    )
                    projection = torch.nn.Linear(1536, 512, bias=False)
                    if os.path.exists(projection_weights_path):
                        try:
                            state_dict = torch.load(projection_weights_path, map_location=self._device)
                            projection.load_state_dict(state_dict)
                            logger.info(f"投影层权重已加载: {projection_weights_path}")
                        except Exception as e:
                            logger.warning(f"加载投影层权重失败: {e}，使用 Xavier 初始化")
                            torch.nn.init.xavier_normal_(projection.weight)
                    else:
                        # P2-7: 权重文件不存在，使用 Xavier 初始化并记录 warning
                        torch.nn.init.xavier_normal_(projection.weight)
                        logger.warning(
                            "投影层权重文件 models/efficientnet_b3/projection_weights.pth 不存在，"
                            "使用随机 Xavier 初始化。建议训练投影层权重以提升 FAISS 检索质量。"
                        )
                    projection = projection.to(self._device)
                    self._projection = projection
                with torch.no_grad():
                    feat_512 = self._projection(feat_1536)
                    norm = feat_512.norm()
                    if norm > 0:
                        feat_512 = feat_512 / norm
                feature = feat_512.cpu().numpy().astype(np.float32)
            else:
                feature = np.zeros(512, dtype=np.float32)
            return role, confidence, feature
        except Exception as e:
            logger.error(f"EfficientNet分类+特征提取失败: {e}")
            return "unknown", 0.0, np.zeros(512, dtype=np.float32)

    def classify_batch(self, images: list, batch_size: int = 8) -> list:
        """批量推理分类（P1-1）

        将多张图片 transform 后 stack 成 (B, 3, 224, 224) batch tensor，
        单次 model.forward(batch_input) 推理，按 batch 维度拆分结果。

        Args:
            images: PIL.Image 列表
            batch_size: 每批大小，默认 8

        Returns:
            list[dict]: 每个元素为 {"role": str, "confidence": float}
        """
        import torch

        if self.model is None:
            return [{"role": "unknown", "confidence": 0.0} for _ in images]

        results: list = []
        total = len(images)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = images[start:end]

            try:
                # transform 所有图像并 stack 成 batch tensor
                tensors = [self.transform(img) for img in batch]
                batch_input = torch.stack(tensors).to(self._device)

                with torch.no_grad():
                    outputs = self.model(batch_input)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    max_probs, predicted_indices = torch.max(probabilities, dim=1)

                for j in range(len(batch)):
                    idx = predicted_indices[j].item()
                    conf = max_probs[j].item()
                    role = self.idx_to_class.get(idx, "unknown")
                    results.append({"role": role, "confidence": conf})
            except Exception as e:
                logger.error(f"批量推理失败 (batch {start}-{end}): {e}")
                for _ in range(len(batch)):
                    results.append({"role": "unknown", "confidence": 0.0})

        return results
