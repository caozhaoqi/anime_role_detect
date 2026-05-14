import os
import hashlib
from PIL import Image

def clean_dataset():
    dataset_dir = 'data/merged_english_dataset'
    
    print('=' * 80)
    print('          清理合并数据集')
    print('=' * 80)
    
    # 统计变量
    total_images = 0
    deleted_duplicates = 0
    deleted_low_quality = 0
    hash_dict = {}
    
    # 遍历所有角色目录
    for role_dir in os.listdir(dataset_dir):
        role_path = os.path.join(dataset_dir, role_dir)
        if not os.path.isdir(role_path):
            continue
        
        for filename in os.listdir(role_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            
            image_path = os.path.join(role_path, filename)
            total_images += 1
            
            # 检查是否重复
            try:
                with open(image_path, 'rb') as f:
                    md5_hash = hashlib.md5(f.read()).hexdigest()
            except:
                # 无法读取的文件直接删除
                os.remove(image_path)
                deleted_low_quality += 1
                continue
            
            if md5_hash in hash_dict:
                # 删除重复图片
                os.remove(image_path)
                deleted_duplicates += 1
                continue
            else:
                hash_dict[md5_hash] = image_path
            
            # 检查图片质量
            try:
                img = Image.open(image_path)
                width, height = img.size
                filesize = os.path.getsize(image_path)
                
                # 判断低质量标准
                is_low_quality = False
                
                # 文件大小小于10KB
                if filesize < 10 * 1024:
                    is_low_quality = True
                
                # 分辨率过低 (< 200x200)
                elif width < 200 or height < 200:
                    is_low_quality = True
                
                # 宽高比异常
                elif width / height < 0.2 or width / height > 5:
                    is_low_quality = True
                
                if is_low_quality:
                    os.remove(image_path)
                    deleted_low_quality += 1
                
                img.close()
            except Exception as e:
                # 无法打开的文件直接删除
                os.remove(image_path)
                deleted_low_quality += 1
    
    print(f"\n【清理结果】")
    print(f"  原始图片总数: {total_images} 张")
    print(f"  删除重复图片: {deleted_duplicates} 张")
    print(f"  删除低质量图片: {deleted_low_quality} 张")
    print(f"  剩余图片数: {len(hash_dict)} 张")
    
    print("\n" + "=" * 80)
    
    # 输出各角色剩余图片数量
    print("\n【各角色剩余图片数量】")
    print('-' * 40)
    role_stats = []
    for role_dir in os.listdir(dataset_dir):
        role_path = os.path.join(dataset_dir, role_dir)
        if not os.path.isdir(role_path):
            continue
        count = len([f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        role_stats.append((role_dir, count))
    
    # 按数量排序
    role_stats.sort(key=lambda x: x[1], reverse=True)
    
    for role, count in role_stats:
        print(f"{role:<15} {count:>3} 张")

if __name__ == '__main__':
    clean_dataset()
