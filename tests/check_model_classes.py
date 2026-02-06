#!/usr/bin/env python3
"""
检查EfficientNet模型的类别列表
"""
from src.core.classification.efficientnet_inference import EfficientNetInference

# 初始化EfficientNet模型
infer = EfficientNetInference()

# 打印模型类别信息
print('模型类别数量:', len(infer.classes))
print('模型类别示例:')
for i, cls in enumerate(infer.classes[:30]):
    print(f'{i}: {cls}')

# 检查测试角色是否在模型类别中
test_roles = [
    'one_piece_路飞',
    'demon_slayer_炭治郎', 
    'honkai_star_rail_三月七',
    'genshin_impact_胡桃',
    'genshin_impact_雷电将军',
    'genshin_impact_温迪',
    'demon_slayer_祢豆子',
    'dragon_ball_贝吉塔',
    'honkai_impact_3_琪亚娜·卡斯兰娜',
    'genshin_impact_神里绫华'
]

print('\n检查测试角色是否在模型类别中:')
for role in test_roles:
    print(f'{role}: {role in infer.classes}')

# 检查是否有相似的类别
print('\n检查是否有相似的类别:')
similar_classes = []
for cls in infer.classes:
    for role in test_roles:
        if role in cls or cls in role:
            similar_classes.append(cls)
            break

if similar_classes:
    print('找到相似的类别:')
    for cls in similar_classes:
        print(cls)
else:
    print('未找到相似的类别')
