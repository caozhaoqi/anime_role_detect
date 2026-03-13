import os

data_dir = 'data/train'

# 检查数据目录是否存在
if os.path.exists(data_dir):
    print(f"数据目录存在: {data_dir}")
else:
    print(f"数据目录不存在: {data_dir}")
    exit(1)

# 获取角色目录列表
role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
print(f"发现 {len(role_dirs)} 个角色目录")

# 检查前5个角色目录
for role_name in role_dirs[:5]:
    role_dir = os.path.join(data_dir, role_name)
    image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"角色 '{role_name}' 有 {len(image_files)} 张图片")
    if len(image_files) > 0:
        print(f"  示例: {image_files[0]}")

# 检查日奈目录
hinata_dir = os.path.join(data_dir, '日奈')
if os.path.exists(hinata_dir):
    image_files = [f for f in os.listdir(hinata_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"\n日奈目录有 {len(image_files)} 张图片")
    if len(image_files) > 0:
        print(f"  示例: {image_files[:3]}")
else:
    print("\n日奈目录不存在")
