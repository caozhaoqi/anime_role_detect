import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

# 加载模型文件
checkpoint = torch.load('../models/character_classifier_best_improved.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']

print('=== 原始权重键 ===')
for key in state_dict.keys():
    if 'classifier' in key:
        print(f"Key: {key}")
        print(f"Shape: {state_dict[key].shape}")

# 修复键名
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('backbone.'):
        name = k[9:] # 移除 'backbone.'
    else:
        name = k
    new_state_dict[name] = v

print('\n=== 修复后权重键 ===')
for key in new_state_dict.keys():
    if 'classifier' in key:
        print(f"Key: {key}")
        print(f"Shape: {new_state_dict[key].shape}")

# 创建模型
num_classes = 131
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

print('\n=== 模型期望的键 ===')
for name, param in model.named_parameters():
    if 'classifier' in name:
        print(f"Name: {name}")
        print(f"Shape: {param.shape}")

# 尝试加载权重
print('\n=== 加载权重 ===')
try:
    # 先尝试严格匹配
    model.load_state_dict(new_state_dict, strict=True)
    print("✅ 权重加载成功（严格匹配）")
except Exception as e:
    print(f"❌ 严格匹配失败: {e}")
    try:
        # 再尝试非严格匹配
        model.load_state_dict(new_state_dict, strict=False)
        print("⚠️  权重加载成功（非严格匹配）")
    except Exception as e:
        print(f"❌ 非严格匹配也失败: {e}")

# 测试模型预测
print('\n=== 测试模型预测 ===')
try:
    # 创建一个随机输入
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # 设置模型为评估模式
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # 获取预测结果
    top_prob, top_idx = torch.topk(probabilities, 5)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Top 5 probabilities: {top_prob[0].tolist()}")
    print(f"Top 5 indices: {top_idx[0].tolist()}")
    
    # 检查是否所有预测都集中在类别0
    if top_idx[0][0].item() == 0 and top_prob[0][0].item() > 0.9:
        print("❌ 警告: 所有预测都集中在类别0，可能是权重加载问题")
    else:
        print("✅ 预测结果正常，没有集中在类别0")
        
except Exception as e:
    print(f"❌ 预测测试失败: {e}")
