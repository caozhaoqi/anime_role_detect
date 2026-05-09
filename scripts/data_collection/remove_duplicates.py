import os
import hashlib

ORGANIZED_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
REORGANIZED_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'

def get_all_images(base_dir):
    """获取目录下所有图片文件的路径"""
    images = {}
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                filepath = os.path.join(root, filename)
                images[filename] = filepath
    return images

def compute_md5(filepath):
    """计算文件MD5值"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def remove_duplicates():
    print("="*60)
    print("删除 organized_images 和 reorganized_dataset 中重复的图片")
    print("="*60)
    print(f"organized_images: {ORGANIZED_DIR}")
    print(f"reorganized_dataset: {REORGANIZED_DIR}")
    print()

    # 获取两个目录中的图片
    org_images = get_all_images(ORGANIZED_DIR)
    reorg_images = get_all_images(REORGANIZED_DIR)
    
    print(f"organized_images 中找到 {len(org_images)} 张图片")
    print(f"reorganized_dataset 中找到 {len(reorg_images)} 张图片")
    print()

    # 找出文件名相同的图片
    duplicate_filenames = set(org_images.keys()) & set(reorg_images.keys())
    print(f"找到 {len(duplicate_filenames)} 个文件名重复的图片")
    
    deleted_count = 0
    skipped_count = 0
    
    for filename in sorted(duplicate_filenames):
        org_path = org_images[filename]
        reorg_path = reorg_images[filename]
        
        # 验证文件是否都存在
        if not os.path.exists(org_path):
            skipped_count += 1
            continue
        if not os.path.exists(reorg_path):
            skipped_count += 1
            continue
        
        # 删除 organized_images 中的重复文件
        os.remove(org_path)
        deleted_count += 1
        
        if deleted_count % 50 == 0:
            print(f"  已删除 {deleted_count} 个重复文件...")
    
    print()
    print(f"删除完成！")
    print(f"  - 删除重复文件: {deleted_count} 个")
    print(f"  - 跳过（文件不存在）: {skipped_count} 个")
    print()

    # 统计清理后的文件数量
    org_images_after = get_all_images(ORGANIZED_DIR)
    reorg_images_after = get_all_images(REORGANIZED_DIR)
    
    print(f"清理后:")
    print(f"  organized_images: {len(org_images_after)} 张图片")
    print(f"  reorganized_dataset: {len(reorg_images_after)} 张图片")

if __name__ == '__main__':
    remove_duplicates()