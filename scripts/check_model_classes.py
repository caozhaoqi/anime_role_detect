#!/usr/bin/env python3
"""
检查模型的训练类别信息
"""
import torch

# 加载模型
model_path = 'models/character_classifier_best_improved.pth'
checkpoint = torch.load(model_path, map_location='cpu')

# 检查是否有类别信息
if 'class_to_idx' in checkpoint:
    class_to_idx = checkpoint['class_to_idx']
    print('类别数量:', len(class_to_idx))
    print('\n前20个类别:')
    for i, cls in enumerate(list(class_to_idx.keys())[:20]):
        print(f'{i+1}. {cls}')
    
    # 检查是否包含测试角色
    test_characters = ['火花', '雷电将军', '琪亚娜']
    print('\n检查测试角色是否在训练类别中:')
    for char in test_characters:
        found = False
        for cls in class_to_idx.keys():
            if char in cls:
                print(f'✓ 找到包含 "{char}" 的类别: {cls}')
                found = True
                break
        if not found:
            print(f'✗ 未找到包含 "{char}" 的类别')
else:
    print('模型中没有class_to_idx信息')
    
print('\n模型检查完成!')
