import torch

# 加载模型文件
checkpoint = torch.load('../models/character_classifier_best_improved.pth', map_location='cpu')
print('Keys in checkpoint:', list(checkpoint.keys()))

# 检查模型结构
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

print('First 10 keys in state_dict:', list(state_dict.keys())[:10])

# 检查分类器权重形状
for key in state_dict.keys():
    if 'classifier' in key:
        print('Found classifier key:', key)
        if 'weight' in key:
            print('Classifier weight shape:', state_dict[key].shape)
            print('Number of classes in model:', state_dict[key].shape[0])
        elif 'bias' in key:
            print('Classifier bias shape:', state_dict[key].shape)

# 检查class_to_idx
if 'class_to_idx' in checkpoint:
    print('Number of classes in class_to_idx:', len(checkpoint['class_to_idx']))
    print('First 10 classes:', list(checkpoint['class_to_idx'].keys())[:10])
