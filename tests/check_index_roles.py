import os
import json

# 加载索引映射文件
with open('role_index_mapping.json', 'r', encoding='utf-8') as f:
    index_roles = json.load(f)

# 获取唯一角色
unique_index_roles = list(set(index_roles))
print(f"索引中包含 {len(unique_index_roles)} 个唯一角色:")
for role in sorted(unique_index_roles):
    print(f"- {role}")

# 获取训练数据目录中的角色目录
train_dir = 'data/train'
all_role_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
print(f"\n训练数据目录中包含 {len(all_role_dirs)} 个角色目录:")

# 找出没有被包含在索引中的角色目录
missing_roles = [role for role in all_role_dirs if role not in unique_index_roles]
print(f"\n没有被包含在索引中的角色目录 ({len(missing_roles)} 个):")
for role in sorted(missing_roles):
    print(f"- {role}")

# 检查这些目录是否包含图片
print("\n检查缺失角色目录的图片数量:")
for role in sorted(missing_roles):
    role_path = os.path.join(train_dir, role)
    image_files = [f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"{role}: {len(image_files)} 张图片")
