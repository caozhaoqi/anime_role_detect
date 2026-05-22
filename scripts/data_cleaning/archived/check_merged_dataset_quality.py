import os
import hashlib
from PIL import Image

def check_image_quality():
    dataset_dir = 'data/merged_english_dataset'
    
    print('=' * 80)
    print('          检查合并数据集质量')
    print('=' * 80)
    
    # 统计变量
    total_images = 0
    duplicate_pairs = []
    low_quality_images = []
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
            
            # 计算MD5哈希值检测重复
            with open(image_path, 'rb') as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()
            
            if md5_hash in hash_dict:
                duplicate_pairs.append((hash_dict[md5_hash], image_path))
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
                    reason = f"文件过小 ({filesize/1024:.1f}KB)"
                
                # 分辨率过低 (< 200x200)
                elif width < 200 or height < 200:
                    is_low_quality = True
                    reason = f"分辨率过低 ({width}x{height})"
                
                # 宽高比异常
                elif width / height < 0.2 or width / height > 5:
                    is_low_quality = True
                    reason = f"宽高比异常 ({width}x{height})"
                
                if is_low_quality:
                    low_quality_images.append({
                        'path': image_path,
                        'role': role_dir,
                        'width': width,
                        'height': height,
                        'size_kb': filesize / 1024,
                        'reason': reason
                    })
                
                img.close()
            except Exception as e:
                low_quality_images.append({
                    'path': image_path,
                    'role': role_dir,
                    'width': 0,
                    'height': 0,
                    'size_kb': os.path.getsize(image_path) / 1024,
                    'reason': f"无法打开: {str(e)}"
                })
    
    # 输出结果
    print(f"\n【一、总体统计】")
    print(f"  图片总数: {total_images} 张")
    print(f"  重复图片对数: {len(duplicate_pairs)} 对")
    print(f"  低质量图片: {len(low_quality_images)} 张")
    print(f"  唯一图片数: {len(hash_dict)} 张")
    
    print(f"\n【二、重复图片详情】")
    if duplicate_pairs:
        print('-' * 70)
        for i, (original, duplicate) in enumerate(duplicate_pairs[:10], 1):
            role1 = original.split('/')[-2]
            role2 = duplicate.split('/')[-2]
            print(f"{i:2d}. {role1}/{os.path.basename(original)}")
            print(f"    ↔ {role2}/{os.path.basename(duplicate)}")
        if len(duplicate_pairs) > 10:
            print(f"    ... 还有 {len(duplicate_pairs) - 10} 对重复")
    else:
        print("  ✅ 未发现重复图片")
    
    print(f"\n【三、低质量图片详情】")
    if low_quality_images:
        print('-' * 70)
        print(f"{'序号':<4} {'角色':<12} {'文件名':<20} {'尺寸':<12} {'大小':<8} {'原因'}")
        print('-' * 70)
        for i, img in enumerate(low_quality_images[:15], 1):
            print(f"{i:<4} {img['role'][:11]:<12} {os.path.basename(img['path'])[:19]:<20} "
                  f"{img['width']}x{img['height']:<12} {img['size_kb']:.1f}KB     {img['reason']}")
        if len(low_quality_images) > 15:
            print(f"    ... 还有 {len(low_quality_images) - 15} 张低质量图片")
    else:
        print("  ✅ 未发现低质量图片")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    check_image_quality()
