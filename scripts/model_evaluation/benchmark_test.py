#!/usr/bin/env python3
"""模型基准测试脚本"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        self.class_names = list(self.class_to_idx.keys())
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return None, label

class BenchmarkEvaluator:
    def __init__(self, model_path, num_classes=52):
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = None
        self.device = None
    
    def load_model(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"使用设备: {self.device}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
        else:
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            if "efficientnet" in self.model_path.lower():
                model = models.efficientnet_b3()
                if "classifier.1.weight" in state_dict:
                    num_classes = state_dict["classifier.1.weight"].shape[0]
                else:
                    num_classes = self.num_classes
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                self.model = model
            elif "resnet" in self.model_path.lower():
                model = models.resnet50()
                if "fc.weight" in state_dict:
                    num_classes = state_dict["fc.weight"].shape[0]
                else:
                    num_classes = self.num_classes
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                self.model = model
            else:
                model = models.mobilenet_v2()
                if "classifier.1.weight" in state_dict:
                    num_classes = state_dict["classifier.1.weight"].shape[0]
                else:
                    num_classes = self.num_classes
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                self.model = model
            
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"模型加载成功: {self.model_path}")
    
    def benchmark(self, test_loader):
        total_correct = 0
        total_samples = 0
        inference_times = []
        
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                if images is None:
                    continue
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                start_time = time.time()
                outputs = self.model(images)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                
                _, preds = torch.max(outputs, 1)
                total_correct += torch.sum(preds == labels).item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            "accuracy": accuracy,
            "avg_inference_time": avg_inference_time,
            "fps": fps,
            "total_samples": total_samples,
            "correct_predictions": total_correct,
        }

def main():
    model_paths = [
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/models/mobilenetv2_aug_best.pth",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/models/mobilenetv2_best.pth",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/models/resnet50_best.pth",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/models/efficientnet_b3_anime_20260616_132028/model_best.pth",
    ]
    
    test_dir = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"测试数据集: {len(test_dataset)} 样本, {len(test_dataset.class_names)} 类别")
    print(f"测试批次大小: 32\n")
    
    results = []
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"模型不存在: {model_path}")
            continue
        
        model_name = os.path.basename(model_path)
        print(f"=" * 50)
        print(f"基准测试: {model_name}")
        print(f"=" * 50)
        
        evaluator = BenchmarkEvaluator(model_path)
        evaluator.load_model()
        
        result = evaluator.benchmark(test_loader)
        
        print(f"准确率: {result['accuracy']:.4f}")
        print(f"平均推理时间: {result['avg_inference_time']:.4f} s")
        print(f"FPS: {result['fps']:.2f}")
        print(f"测试样本: {result['total_samples']}")
        print(f"正确预测: {result['correct_predictions']}\n")
        
        results.append({
            "model_name": model_name,
            "model_path": model_path,
            **result,
        })
    
    print("=" * 50)
    print("基准测试汇总")
    print("=" * 50)
    print(f"{'模型名称':<30} {'准确率':<10} {'推理时间(s)':<12} {'FPS':<8}")
    print("-" * 60)
    
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{r['model_name']:<30} {r['accuracy']:.4f}     {r['avg_inference_time']:.4f}         {r['fps']:.2f}")
    
    report_path = os.path.join(project_root, "models", "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_path}")

if __name__ == "__main__":
    main()