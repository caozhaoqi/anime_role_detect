import os
import shutil

def merge_folders(source_dir, target_dir):
    """合并两个文件夹，处理重名文件"""
    if not os.path.exists(source_dir):
        print(f"源目录不存在: {source_dir}")
        return
    
    if not os.path.exists(target_dir):
        print(f"目标目录不存在: {target_dir}")
        return
    
    files_moved = 0
    files_renamed = 0
    
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        if os.path.isfile(source_path):
            if os.path.exists(target_path):
                # 处理重名文件
                name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(target_path):
                    new_name = f"{name}_{counter}{ext}"
                    target_path = os.path.join(target_dir, new_name)
                    counter += 1
                shutil.move(source_path, target_path)
                files_renamed += 1
            else:
                shutil.move(source_path, target_path)
                files_moved += 1
    
    # 删除空的源目录
    if len(os.listdir(source_dir)) == 0:
        os.rmdir(source_dir)
        print(f"已删除空目录: {source_dir}")
    
    print(f"\n合并完成:")
    print(f"  - 直接移动文件: {files_moved} 个")
    print(f"  - 重命名后移动: {files_renamed} 个")
    print(f"  - 总计移动: {files_moved + files_renamed} 个")

if __name__ == "__main__":
    base_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images"
    source_folder = os.path.join(base_path, "ni2dou4zi5")
    target_folder = os.path.join(base_path, "mi2dou4zi")
    
    print(f"合并 {source_folder} -> {target_folder}")
    print(f"源目录文件数: {len([f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))])}")
    print(f"目标目录文件数: {len([f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))])}")
    
    merge_folders(source_folder, target_folder)
    
    print(f"\n合并后目标目录文件数: {len([f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))])}")
