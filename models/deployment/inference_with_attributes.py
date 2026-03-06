#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带属性预测的动漫角色检测器
"""

import os
import sys
import json
import numpy as np
import time
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
from torchvision import transforms, models
from src.core.classification.models import get_model_with_attributes


class AnimeRoleDetectorWithAttributes:
    """带属性预测的动漫角色检测器"""

    def __init__(self, model_path, class_to_idx_path=None, tag_to_idx_path=None):
        self.model_path = model_path
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.class_to_idx = {}
        self.idx_to_class = {}
        self.tag_to_idx = {}
        self.idx_to_tag = {}

        if class_to_idx_path and os.path.exists(class_to_idx_path):
            with open(class_to_idx_path, 'r', encoding='utf-8') as f:
                self.class_to_idx = json.load(f)
                self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if tag_to_idx_path and os.path.exists(tag_to_idx_path):
            with open(tag_to_idx_path, 'r', encoding='utf-8') as f:
                self.tag_to_idx = json.load(f)
                self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}

        self.model = self._load_model()
        print(f"初始化完成，支持 {len(self.class_to_idx)} 个角色")
        print(f"支持 {len(self.tag_to_idx)} 个属性标签")

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        model_type = checkpoint.get('model_type', 'mobilenet_v2')
        num_classes = len(self.class_to_idx)
        num_attributes = len(self.tag_to_idx)

        model = get_model_with_attributes(model_type, num_classes, num_attributes)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')

        preprocessed = self.preprocess(image)
        input_tensor = preprocessed.unsqueeze(0).to(self.device)

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(input_tensor)

        inference_time = (time.time() - start_time) * 1000

        if isinstance(outputs, (tuple, list)):
            class_outputs = outputs[0]
            attribute_outputs = outputs[1]
        else:
            class_outputs = outputs
            attribute_outputs = None

        class_probs = torch.softmax(class_outputs, dim=1)[0]
        predicted_class = torch.argmax(class_probs).item()
        confidence = class_probs[predicted_class].item()

        class_name = self.idx_to_class.get(predicted_class, '未知')

        result = {
            'class_name': class_name,
            'confidence': float(confidence),
            'inference_time_ms': inference_time,
            'fps': 1000 / inference_time if inference_time > 0 else 0,
            'all_classes': [
                {'name': self.idx_to_class.get(i, f'class_{i}'), 'probability': float(class_probs[i])}
                for i in range(len(class_probs))
            ]
        }

        # 处理属性预测结果
        if attribute_outputs is not None and len(self.idx_to_tag) > 0:
            attribute_probs = torch.sigmoid(attribute_outputs)[0]
            # 过滤置信度高于阈值的属性
            threshold = 0.5
            predicted_attributes = []
            for i, prob in enumerate(attribute_probs):
                if prob > threshold:
                    tag_name = self.idx_to_tag.get(i, f'tag_{i}')
                    predicted_attributes.append({
                        'tag': tag_name,
                        'confidence': float(prob)
                    })
            # 按置信度排序
            predicted_attributes.sort(key=lambda x: x['confidence'], reverse=True)
            result['attributes'] = predicted_attributes

        return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='带属性预测的动漫角色检测器')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--model', type=str, default='models/character_classifier_with_attributes/model_best.pth', help='模型路径')
    parser.add_argument('--class-to-idx', type=str, default=None, help='类别索引映射文件路径')
    parser.add_argument('--tag-to-idx', type=str, default=None, help='标签索引映射文件路径')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        return

    class_to_idx_path = args.class_to_idx
    if not class_to_idx_path:
        model_dir = os.path.dirname(args.model)
        for f in os.listdir(model_dir):
            if f.endswith('.json') and 'class' in f.lower():
                class_to_idx_path = os.path.join(model_dir, f)
                break

    tag_to_idx_path = args.tag_to_idx
    if not tag_to_idx_path:
        model_dir = os.path.dirname(args.model)
        for f in os.listdir(model_dir):
            if f.endswith('.json') and 'tag' in f.lower():
                tag_to_idx_path = os.path.join(model_dir, f)
                break

    if class_to_idx_path:
        print(f"使用类别映射文件: {class_to_idx_path}")
    if tag_to_idx_path:
        print(f"使用标签映射文件: {tag_to_idx_path}")

    detector = AnimeRoleDetectorWithAttributes(args.model, class_to_idx_path, tag_to_idx_path)

    result = detector.predict(args.image)

    print(f"\n预测角色: {result['class_name']}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    print(f"推理速度: {result['fps']:.2f} FPS")

    if 'all_classes' in result:
        print("\n所有类别预测结果:")
        for cls in sorted(result['all_classes'], key=lambda x: x['probability'], reverse=True):
            print(f"  {cls['name']}: {cls['probability']:.4f}")
    
    if 'attributes' in result and result['attributes']:
        print("\n预测的属性标签:")
        for attr in result['attributes'][:10]:  # 只显示前10个属性
            print(f"  {attr['tag']}: {attr['confidence']:.4f}")


if __name__ == '__main__':
    main()
