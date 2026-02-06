import torch
from collections import OrderedDict

# 加载模型文件
checkpoint = torch.load('../models/character_classifier_best_improved.pth', map_location='cpu')

# 检查模型结构
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# 检查分类器权重
print('=== 分类器权重信息 ===')
for key in state_dict.keys():
    if 'classifier' in key:
        print(f"Key: {key}")
        print(f"Shape: {state_dict[key].shape}")
        print(f"Number of classes: {state_dict[key].shape[0] if len(state_dict[key].shape) > 0 else 'N/A'}")
        print()

# 检查class_to_idx
if 'class_to_idx' in checkpoint:
    print('=== class_to_idx 信息 ===')
    print(f"Number of classes: {len(checkpoint['class_to_idx'])}")
    print('First 10 classes:')
    for i, (class_name, idx) in enumerate(list(checkpoint['class_to_idx'].items())[:10]):
        print(f"  {i}: {class_name} -> {idx}")
