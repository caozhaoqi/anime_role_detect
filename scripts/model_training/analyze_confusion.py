#!/usr/bin/env python3
"""
生成混淆矩阵，分析 Furina/Paimon/Xiangling 等角色被误识别成谁
"""
import sys
import os

# 解码策略统一由 src/common/preprocess 提供，导入即继承，本脚本不再自行设置。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.common.preprocess  # noqa: E402

import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile
import warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings('ignore')

TRAIN_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
FINAL_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_and_preprocess_image(img_path):
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img)
    except:
        return None


def analyze_confusion_matrix(model, device, dataset_dir, class_to_idx, idx_to_class):
    """生成混淆矩阵并分析误分类"""
    print(f"\n📊 分析数据集: {dataset_dir.name}")
    
    test_images = []
    for char_dir in dataset_dir.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            char_lower = char_name.lower()
            
            matched_class = None
            for cls in class_to_idx.keys():
                if cls.lower() == char_lower:
                    matched_class = cls
                    break
            
            if matched_class:
                label_idx = class_to_idx[matched_class]
                for img_path in list(char_dir.glob('*.jpg')) + list(char_dir.glob('*.png')):
                    test_images.append((str(img_path), label_idx, matched_class))
    
    if not test_images:
        return None
    
    print(f"  测试图片数: {len(test_images)}")
    
    # 收集预测结果
    confusion = defaultdict(lambda: defaultdict(int))
    predictions = defaultdict(list)
    
    model.eval()
    batch_size = 32
    
    for i in tqdm(range(0, len(test_images), batch_size), desc="分析中"):
        batch = test_images[i:i+batch_size]
        batch_tensors = []
        batch_labels = []
        batch_classes = []
        
        for img_path, label_idx, class_name in batch:
            tensor = load_and_preprocess_image(img_path)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_labels.append(label_idx)
                batch_classes.append(class_name)
        
        if not batch_tensors:
            continue
        
        batch_tensor = torch.stack(batch_tensors).to(device)
        batch_labels = torch.tensor(batch_labels).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            for j, (pred, label, conf, cls) in enumerate(zip(preds, batch_labels, confs, batch_classes)):
                true_class = idx_to_class[label.item()]
                pred_class = idx_to_class[pred.item()]
                confusion[true_class][pred_class] += 1
                predictions[true_class].append({
                    'pred': pred_class,
                    'conf': conf.item(),
                    'correct': pred == label
                })
    
    return confusion, predictions


def main():
    print("=" * 70)
    print("📊 混淆矩阵分析")
    print("=" * 70)
    
    device = get_device()
    print(f"📱 使用设备: {device}")
    
    # 获取类别
    train_classes = sorted([d.name for d in TRAIN_DATA_DIR.iterdir() if d.is_dir()])
    class_to_idx = {cls: i for i, cls in enumerate(train_classes)}
    idx_to_class = {i: cls for i, cls in enumerate(train_classes)}
    num_classes = len(train_classes)
    
    # 加载模型
    print("\n📦 加载模型...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model_path = MODEL_DIR / "mobilenetv2_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"✅ 加载模型: {model_path}")
    
    # 分析 final_dataset（因为它更能反映真实场景）
    confusion, predictions = analyze_confusion_matrix(
        model, device, FINAL_DATA_DIR, class_to_idx, idx_to_class
    )
    
    if not confusion:
        print("❌ 无数据可分析")
        return
    
    # 重点分析几个表现差的角色
    focus_chars = ['Furina', 'Paimon', 'Xiangling', 'Ningguang', 'Klee', 'Rosaria', 'Sayu']
    
    print("\n" + "=" * 70)
    print("🔍 重点角色误分类分析")
    print("=" * 70)
    
    for char in focus_chars:
        if char not in predictions:
            continue
        
        preds = predictions[char]
        total = len(preds)
        correct = sum(1 for p in preds if p['correct'])
        accuracy = correct / total * 100 if total > 0 else 0
        
        # 统计被误分类成哪些角色
        misclass = defaultdict(int)
        for p in preds:
            if not p['correct']:
                misclass[p['pred']] += 1
        
        # 按误分类次数排序
        top_misclass = sorted(misclass.items(), key=lambda x: -x[1])[:5]
        
        print(f"\n【{char}】")
        print(f"  准确率: {accuracy:.1f}% ({correct}/{total})")
        print(f"  误分类去向 (TOP 5):")
        for pred, count in top_misclass:
            print(f"    → {pred}: {count}次 ({count/total*100:.1f}%)")
        
        # 找出最低置信度的预测
        low_conf = [p for p in preds if not p['correct']]
        low_conf.sort(key=lambda x: x['conf'])
        if low_conf:
            print(f"  最低置信度误分类: {low_conf[0]['pred']} (置信度: {low_conf[0]['conf']:.2f})")
    
    # 生成完整混淆矩阵摘要
    print("\n" + "=" * 70)
    print("📈 完整角色准确率排名")
    print("=" * 70)
    
    char_accuracies = []
    for char in confusion:
        total = sum(confusion[char].values())
        correct = confusion[char].get(char, 0)
        acc = correct / total * 100 if total > 0 else 0
        char_accuracies.append((char, acc, correct, total))
    
    char_accuracies.sort(key=lambda x: -x[1])
    
    print(f"\n{'角色':<15} {'准确率':>8} {'正确/总数':>12}")
    print("-" * 40)
    for char, acc, correct, total in char_accuracies:
        marker = "⚠️" if acc < 30 else ("🔴" if acc < 50 else "✅")
        print(f"{marker} {char:<13} {acc:>7.1f}% {correct:>6}/{total}")
    
    # 分析跨游戏误分类
    print("\n" + "=" * 70)
    print("🎮 跨游戏误分类分析")
    print("=" * 70)
    
    # 定义角色所属游戏（简化判断）
    games = {
        'Genshin': ['Furina', 'Paimon', 'Xiangling', 'Ningguang', 'Klee', 'Rosaria', 'Sayu', 
                    'Amber', 'Ayaka', 'Barbara', 'Beidou', 'Dehya', 'Eula', 'Ganyu', 'Hutao',
                    'Keqing', 'Kokomi', 'Lisa', 'Mona', 'Nahida', 'Nilou', 'Qiqi', 'Sara',
                    'Shenhe', 'Sucrose', 'Xinyan', 'Yae', 'Yoimiya', 'Yunjin', 'Yunli', 'Yelan',
                    'Clorinde', 'Lyney'],
        'HSR': ['Firefly', 'Herta', 'Kafka', 'SilverWolf', 'Arona', 'Plana', 'Hoshino', 'Shiroko',
                'Izuna', 'Hina', 'Ayane', 'Aris', 'Aru', 'Mutsuki', 'Hanako', 'Saya', 'Momoi', 'Mika'],
        'AK': ['Eyjafjalla', 'Rosmontis', 'Suzuran', 'Ceobe', 'Specter', 'Lappland', 'Exusiai',
               'SilverAsh', 'Chen', 'Saria', 'Hoshi', 'Kroos', 'Myrtle', 'Vigil'],
        'BlueArchive': ['Arona', 'Plana', 'Hoshino', 'Shiroko', 'Izuna', 'Hina', 'Ayane', 'Aris',
                       'Aru', 'Mutsuki', 'Neru', 'Hanako', 'Saya', 'Momoi', 'Mika', 'Asuna']
    }
    
    def get_game(char):
        for game, chars in games.items():
            if char in chars:
                return game
        return 'Other'
    
    cross_game_confusion = defaultdict(lambda: defaultdict(int))
    
    for true_char in confusion:
        true_game = get_game(true_char)
        for pred_char, count in confusion[true_char].items():
            pred_game = get_game(pred_char)
            if true_game != pred_game and true_char != pred_char:
                cross_game_confusion[f"{true_char}({true_game})"][f"{pred_char}({pred_game})"] += count
    
    print("\n跨游戏误分类示例:")
    for true_char, preds in sorted(cross_game_confusion.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
        total = sum(preds.values())
        top_pred = sorted(preds.items(), key=lambda x: -x[1])[0]
        print(f"  {true_char} → {top_pred[0]}: {total}次")


if __name__ == "__main__":
    main()