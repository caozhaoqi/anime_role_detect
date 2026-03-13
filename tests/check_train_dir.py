import os
import glob

data_dir = 'data/train'

# 获取角色目录列表
role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

print("data/train目录中包含图片的角色目录:")
for role_name in role_dirs:
    role_dir = os.path.join(data_dir, role_name)
    # 查找所有支持的图片格式
    image_files = glob.glob(os.path.join(role_dir, '*.jpg')) + \
                  glob.glob(os.path.join(role_dir, '*.jpeg')) + \
                  glob.glob(os.path.join(role_dir, '*.png')) + \
                  glob.glob(os.path.join(role_dir, '*.bmp')) + \
                  glob.glob(os.path.join(role_dir, '*.webp'))
    
    if len(image_files) > 0:
        print(f"{role_name}: {len(image_files)} 张图片")
