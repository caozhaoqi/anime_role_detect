#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grad-CAM 热力图生成器（手写实现，零外部依赖）。

EfficientNet-B3 的 FP32 独立模型副本，规避 FP16 梯度 NaN 问题。
目标层：model.features[8]（最后一个 Conv2dNormActivation，avgpool 前）。
"""

import base64
import io
import os

import numpy as np
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("detection.gradcam")

_MAX_LONG_EDGE = 1280


class GradCAMGenerator:
    """EfficientNet-B3 Grad-CAM 热力图生成器（FP32 独立副本）"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        import torch
        import torchvision
        from torchvision import transforms

        from src.core.config.device_manager import DeviceManager

        self._device = DeviceManager.get_device()

        model_dir = os.path.join(project_root, "models", "efficientnet_b3")
        model_best_path = os.path.join(model_dir, "model_best.pth")

        if not os.path.exists(model_best_path):
            logger.error(f"GradCAM: 模型文件不存在: {model_best_path}")
            self.model = None
            self.idx_to_class = {}
            return

        logger.info(f"GradCAM: 加载 FP32 模型副本: {model_best_path}")

        checkpoint = torch.load(
            model_best_path, map_location=self._device, weights_only=False
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
            class_to_idx = checkpoint["class_to_idx"]
        else:
            class_to_idx = {}

        num_classes = len(class_to_idx)

        # 构建与训练时一致的模型架构（自定义 classifier）
        base_model = torchvision.models.efficientnet_b3(weights=None)
        base_model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(base_model.classifier[1].in_features, 768),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(768),
            torch.nn.Dropout(p=0.15),
            torch.nn.Linear(768, num_classes),
        )
        model = base_model
        model.load_state_dict(state_dict)
        model = model.to(self._device)
        # 关键：不 half()，保留 FP32 以支持梯度反传（规避 FP16 梯度 NaN）
        model.eval()

        self.model = model
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        # 预处理委托给 src/common/preprocess 唯一真源（256），
        # 保证 Grad-CAM 热力图与线上推理看到的是同一张输入。
        from src.common.preprocess import build_eval_transform

        self.transform = build_eval_transform()
        # Grad-CAM 目标层：features[8]（最后一个 MBConv block 后的 Conv2dNormActivation）
        self.target_layer = model.features[8]

        logger.info(
            f"GradCAM FP32 模型副本加载完成 (device={self._device}, classes={num_classes})"
        )

    def generate(self, image: Image.Image, target_class: int = None) -> dict:
        """生成 Grad-CAM 热力图。

        Args:
            image: PIL.Image（RGB）
            target_class: 目标类别 idx，None 则用预测 top-1

        Returns:
            dict: target_class, target_label, confidence, cam_heatmap_base64, cam_raw
                  失败时返回 {"error": str, "cam_heatmap_base64": None}
        """
        import torch
        import torch.nn.functional as F

        if self.model is None:
            return {"error": "模型未加载", "cam_heatmap_base64": None}

        fwd_handle = None
        saved_activation = [None]
        saved_gradient = [None]

        try:
            # 统一到 RGB：本函数下方可视化分支已做 image.convert("RGB")，但张量分支
            # 此前直接送 transform，RGBA/CMYK 会产出 4 通道、L/P 产出 1 通道，
            # 与模型首层 3 通道不匹配而抛 RuntimeError。两条分支必须一致。
            if image.mode != "RGB":
                image = image.convert("RGB")

            orig_w, orig_h = image.size

            # 1. 预处理：FP32 tensor，开启梯度
            input_tensor = self.transform(image).unsqueeze(0).to(self._device)
            input_tensor.requires_grad_(True)

            # 2. 注册 forward hook 捕获激活，并在激活 tensor 上注册 hook 捕获梯度
            #    （tensor.register_hook 比 register_full_backward_hook 在 MPS 上更可靠）
            def forward_hook(module, inp, out):
                saved_activation[0] = out

                def save_grad(grad):
                    saved_gradient[0] = grad

                out.register_hook(save_grad)

            fwd_handle = self.target_layer.register_forward_hook(forward_hook)

            # 3. 前向传播（不使用 no_grad，保留计算图）
            with torch.enable_grad():
                output = self.model(input_tensor)

                # 4. 确定目标类别
                if target_class is None:
                    target_class = int(output.argmax(1).item())
                else:
                    target_class = int(target_class)
                    if target_class < 0 or target_class >= output.shape[1]:
                        return {
                            "error": f"target_class 越界: {target_class}",
                            "cam_heatmap_base64": None,
                        }

                # 5. 反向传播
                target_score = output[0, target_class]
                self.model.zero_grad()
                target_score.backward()

            # 置信度（softmax）
            probs = F.softmax(output, dim=1)
            confidence = float(probs[0, target_class].item())
            target_label = self.idx_to_class.get(target_class, "unknown")

            # 6. 提取激活和梯度
            activations = saved_activation[0]
            gradients = saved_gradient[0]

            if activations is None or gradients is None:
                return {
                    "error": "hook 未捕获到激活/梯度",
                    "cam_heatmap_base64": None,
                }

            if torch.isnan(gradients).any() or torch.isnan(activations).any():
                logger.warning("GradCAM: 检测到 NaN 梯度/激活，热力图可能无效")

            # 7. Grad-CAM: 全局平均池化得权重 → 加权求和 → ReLU
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (activations * weights).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().detach().numpy()

            # 8. 归一化 0-1
            cam_min, cam_max = float(cam.min()), float(cam.max())
            if cam_max - cam_min < 1e-8:
                cam = np.zeros_like(cam, dtype=np.float32)
            else:
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            # 9. resize 到原图尺寸 + 伪彩映射
            heatmap_colored = self._apply_colormap(cam, orig_w, orig_h)

            # 10. 叠加到原图（alpha=0.5）
            img_np = np.array(image.convert("RGB"))
            overlay = self._blend(img_np, heatmap_colored, alpha=0.5)
            overlay_pil = Image.fromarray(overlay)

            # 11. 下采样 + base64 JPEG data-URI 编码
            overlay_pil = self._downsample(overlay_pil)
            buf = io.BytesIO()
            overlay_pil.save(buf, format="JPEG", quality=85)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            data_uri = "data:image/jpeg;base64," + encoded

            return {
                "target_class": int(target_class),
                "target_label": target_label,
                "confidence": confidence,
                "cam_heatmap_base64": data_uri,
                "cam_raw": cam,
            }

        except Exception as e:
            logger.error(f"GradCAM 生成失败: {e}", exc_info=True)
            return {"error": str(e), "cam_heatmap_base64": None}
        finally:
            if fwd_handle is not None:
                fwd_handle.remove()
            try:
                self.model.zero_grad(set_to_none=True)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  内部工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_colormap(cam: np.ndarray, w: int, h: int) -> np.ndarray:
        """将 0-1 归一化热力图 resize 到 (w,h) 并应用 JET 伪彩，返回 RGB uint8。"""
        try:
            import cv2

            cam_resized = cv2.resize(cam, (w, h))
            cam_uint8 = np.uint8(255 * cam_resized)
            colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            return colored
        except ImportError:
            # fallback：纯 numpy JET 近似映射
            cam_resized = np.array(
                Image.fromarray(np.uint8(255 * cam)).resize((w, h), Image.BILINEAR),
                dtype=np.float32,
            )
            r = np.clip(cam_resized * 2.0 - 128.0, 0, 255)
            g = np.clip(255.0 - np.abs(cam_resized - 128.0) * 2.0, 0, 255)
            b = np.clip(255.0 - cam_resized * 2.0, 0, 255)
            return np.stack([r, g, b], axis=-1).astype(np.uint8)

    @staticmethod
    def _blend(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """alpha 混合原图与热力图，返回 uint8。"""
        try:
            import cv2

            return cv2.addWeighted(img_np, 1.0 - alpha, heatmap, alpha, 0)
        except ImportError:
            blended = img_np.astype(np.float32) * (1.0 - alpha) + heatmap.astype(
                np.float32
            ) * alpha
            return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _downsample(img: Image.Image) -> Image.Image:
        """最长边下采样到 <=1280px。"""
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > _MAX_LONG_EDGE:
            scale = _MAX_LONG_EDGE / float(long_edge)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.BILINEAR)
        return img
