#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merged_english_dataset 目录整理脚本
将所有角色目录重命名为标准英文名，并合并重复数据
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")

# 目标目录
TARGET_DIR = PROJECT_ROOT / "data" / "merged_english_dataset"

# 角色名称映射表（各种名称 -> 标准英文名）
ROLE_MAPPING = {
    # 中文/拼音/日文 -> 英文
    'a1luo4na4': 'Arona',
    'a1ni4ya4': 'Anya',
    'ai4li4er3': 'Alice',
    'an1ka3xi1ya3': 'Aneka',
    'an1ke3': 'Anke',
    'bai2shang4chui1xue3': 'Shirogane',
    'bu4luo4ni2ya4': 'Bronya',
    'cong2yu3': 'Congyu',
    'de2li4sha1': 'Theresa',
    'di2ao4na4': 'Diona',
    'duo1li4': 'Dori',
    'fei1mi3li4si1': 'Fimilis',
    'fei1xie4er3': 'Feixieer',
    'fu2lan2': 'Fran',
    'fu2li4xi1ya4': 'Furixiya',
    'fu2xuan2': 'Fu Xuan',
    'gu3ming2di4lian4': 'Gumingdiliana',
    'hei1ta3': 'Herta',
    'hua1huo3': 'Sparkle',
    'ka3qi2na4': 'Kachina',
    'kai3lu4': 'Kai Lu',
    'ke3lin2': 'Kelin',
    'ke3li4': 'Klee',
    'ke4la1la1': 'Krala',
    'ke4luo2li4ke1': 'Krolike',
    'ke4luo2luo2': 'Krololo',
    'ke1xie4ni2ya4': 'Koshenia',
    'ke3lin2_wei1ke4si1': 'Kelin_Vikosi',
    'qi1qi1': 'Qiqi',
    'qing1que4': 'Qing Que',
    'sha1wu4': 'Shawu',
    'shen1yue4': 'Shen Yue',
    'si4mi4nai3': 'Siminai',
    'xia4ke4li3': 'Xia Keli',
    'xiao3mei3yan4': 'Xiao Mei Yan',
    'xue4xiao3ban3': 'Xue Xiaoban',
    'yaoyao': 'Yaoyao',
    'yaoyao yuan2shen2': 'Yaoyao',
    'yao2yao2': 'Yaoyao',
    'yao2yao2 yuan2shen2': 'Yaoyao',
    'yi1li4ya3': 'Illya',
    'yi1se4lin2': 'Iselin',
    'yin2lang2': 'Yin Lang',
    'zao3wu4': 'Zao Wu',
    'zao3you4': 'Zao You',
    'zhi4nai3': 'Zhinai',
    
    # 日文 -> 英文
    'カグラ': 'Kagura',
    'ユウホ': 'Youhu',
    'ありす': 'Alice',
    'アリエル': 'Ariel',
    'クロロ': 'Kloro',
    
    # 中文_游戏_英文_日文 -> 英文
    '可莉_原神_Klee_クレー': 'Klee',
    '迪奥娜_原神_Diona_ディオナ': 'Diona',
    '瑶瑶_原神_Yaoyao_瑶瑶': 'Yaoyao',
    '阿洛娜_蔚蓝档案_Arona_アロナ': 'Arona',
    '普拉娜_蔚蓝档案_Plana_プラナ': 'Plana',
    '缇宝_崩坏星穹铁道_Princess__Princess': 'Princess',
    '希格雯_原神_Sigewinne_シーewinne': 'Sigewinne',
    '蕾贝_幻塔_Rebe_レべ': 'Rebe',
    '纳西妲_原神_Nahida_ナヒダ': 'Nahida',
    
    # 其他变体
    'luo4qian4': 'Luo Qian',
    'luo4ke3ke3': 'Luo Keke',
    'luo4ke4ke4': 'Luo Keke',
    'luo4sha1li4ya3_a1lin2': 'Roshariya_Alina',
    'luo2sha1li4ya3_a1lin2': 'Roshariya_Alina',
    'mao1gong1you4nai4': 'Maogongyunai',
    'mei2bi3wu3si1': 'Meibiwusi',
    'mi3dou4zi5': 'Midouzi',
    'na4gan1': 'Nagan',
    'na4xi1da4': 'Nahida',
    'na4xi1da2': 'Nahida',
    'ni2dou4zi5': 'Nidouzi',
    'pei4li3ti2ya4': 'Peritia',
    'san1yue4qi1': 'San Yue Qi',
    'tian1tong2ai4li4si1': 'Tiantong Ailisi',
    'ti2bao3': 'Tibao',
    'wei2pu3lei3': 'Weipulei',
    'wei2li3nai4': 'Weilinai',
    'xing4': 'Xing',
    'you4hu2': 'Youhu',
    'zao3wu4': 'Zao Wu',
    'zao3you4': 'Zao You',
    'zhen1bu4': 'Zhenbu',
    'zhen1bu4_result': 'Zhenbu',
    
    # 带下划线的变体
    'li4li4ya3_a1lin2': 'Liliya_Alina',
    'li4li4ya3·a1lin2': 'Liliya_Alina',
    'li4li4ya3a1lin2': 'Liliya_Alina',
    'ren3ye3ren3': 'Ren Ye Ren',
    'shen2le4 yin1yang2shi1': 'Shen Le Yin Yang Shi',
    'shen2le4_yin1yang2shi1': 'Shen Le Yin Yang Shi',
    'xiao3mai2': 'Xiao Mai',
    'xiao3shan3': 'Xiao Shan',
    'yue4qian1ye4': 'Yue Qian Ye',
    
    # 带后缀的变体
    'Aris wei4lan2dang4an4': 'Aris',
    'Yaoyao yuan2shen2': 'Yaoyao',
    'yao2yao2 yuan2shen2': 'Yaoyao',
    'ke3lin2_wei1ke4si1': 'Kelin_Vikosi',
}


def get_file_md5(file_path):
    """计算文件MD5值用于去重"""
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except:
        return None


def get_standard_english_name(dir_name):
    """获取标准英文角色名"""
    # 先尝试精确匹配
    if dir_name in ROLE_MAPPING:
        return ROLE_MAPPING[dir_name]
    
    # 尝试清理后匹配
    clean_name = dir_name.strip()
    if clean_name in ROLE_MAPPING:
        return ROLE_MAPPING[clean_name]
    
    # 如果已经是纯英文格式（只包含字母、空格、下划线）
    if all(c.isalnum() or c in [' ', '_'] for c in clean_name):
        # 标准化：空格转下划线，去除多余下划线
        normalized = clean_name.replace(' ', '_').replace('__', '_').strip('_')
        return normalized
    
    # 对于其他情况，返回清理后的名称
    cleaned = ''.join(c for c in dir_name if c.isalnum() or c in ['_', ' ']).strip()
    cleaned = cleaned.replace(' ', '_').replace('__', '_').strip('_')
    return cleaned if cleaned else dir_name


def organize_merged_dataset():
    """整理 merged_english_dataset 目录"""
    if not TARGET_DIR.exists():
        print(f"目标目录不存在: {TARGET_DIR}")
        return
    
    # 统计信息
    stats = {
        'total_dirs': 0,
        'renamed_dirs': 0,
        'merged_dirs': 0,
        'total_files_scanned': 0,
        'total_files_copied': 0,
        'duplicates_found': 0,
        'empty_dirs_removed': 0,
        'errors': 0,
    }
    
    # 用于去重的MD5缓存
    seen_md5 = set()
    
    # 图片扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
    
    # 收集所有角色目录
    role_dirs = []
    for item in TARGET_DIR.iterdir():
        if item.is_dir():
            role_dirs.append(item)
    
    stats['total_dirs'] = len(role_dirs)
    
    # 按标准英文名分组
    role_groups = defaultdict(list)
    for role_dir in role_dirs:
        standard_name = get_standard_english_name(role_dir.name)
        role_groups[standard_name].append(role_dir)
    
    print(f"发现 {len(role_dirs)} 个角色目录")
    print(f"分组后为 {len(role_groups)} 个标准角色")
    
    # 处理每个角色组
    for standard_name, dirs in role_groups.items():
        if len(dirs) == 1:
            # 只有一个目录，只需重命名
            role_dir = dirs[0]
            if role_dir.name != standard_name:
                new_path = TARGET_DIR / standard_name
                try:
                    role_dir.rename(new_path)
                    stats['renamed_dirs'] += 1
                    print(f"重命名: {role_dir.name} -> {standard_name}")
                except Exception as e:
                    stats['errors'] += 1
                    print(f"重命名失败: {role_dir.name} -> {standard_name}, 错误: {e}")
        else:
            # 多个目录，需要合并
            print(f"\n合并角色: {standard_name}")
            print(f"  源目录: {[d.name for d in dirs]}")
            
            # 创建目标目录
            target_role_dir = TARGET_DIR / standard_name
            target_role_dir.mkdir(exist_ok=True)
            
            # 收集目标目录中已有的文件MD5
            for file_path in target_role_dir.iterdir():
                if file_path.is_file():
                    md5 = get_file_md5(file_path)
                    if md5:
                        seen_md5.add(md5)
            
            # 合并所有源目录的文件
            for source_dir in dirs:
                if source_dir == target_role_dir:
                    continue
                
                file_count = 0
                for file_path in source_dir.iterdir():
                    if not file_path.is_file():
                        continue
                    
                    ext = file_path.suffix.lower()
                    if ext not in valid_extensions:
                        continue
                    
                    stats['total_files_scanned'] += 1
                    
                    # 计算MD5去重
                    md5 = get_file_md5(file_path)
                    if md5 and md5 in seen_md5:
                        stats['duplicates_found'] += 1
                        continue
                    
                    if md5:
                        seen_md5.add(md5)
                    
                    # 复制文件到目标目录
                    try:
                        target_file = target_role_dir / file_path.name
                        # 如果目标文件已存在但MD5不同，添加序号
                        if target_file.exists():
                            base_name = file_path.stem
                            counter = 1
                            while target_file.exists():
                                target_file = target_role_dir / f"{base_name}_{counter}{ext}"
                                counter += 1
                        
                        shutil.copy2(file_path, target_file)
                        stats['total_files_copied'] += 1
                        file_count += 1
                    except Exception as e:
                        stats['errors'] += 1
                        print(f"  复制失败: {file_path.name}, 错误: {e}")
                
                print(f"  {source_dir.name}: {file_count} 个文件")
                
                # 删除已合并的源目录
                try:
                    shutil.rmtree(source_dir)
                    stats['merged_dirs'] += 1
                except Exception as e:
                    stats['errors'] += 1
                    print(f"  删除目录失败: {source_dir.name}, 错误: {e}")
    
    # 清理空目录
    print("\n清理空目录...")
    for role_dir in TARGET_DIR.iterdir():
        if role_dir.is_dir():
            try:
                files = list(role_dir.iterdir())
                if not files:
                    role_dir.rmdir()
                    stats['empty_dirs_removed'] += 1
                    print(f"删除空目录: {role_dir.name}")
            except Exception as e:
                pass
    
    # 输出统计报告
    print("\n" + "="*60)
    print("目录整理完成!")
    print("="*60)
    print(f"总角色目录数: {stats['total_dirs']}")
    print(f"重命名目录数: {stats['renamed_dirs']}")
    print(f"合并目录数: {stats['merged_dirs']}")
    print(f"扫描文件总数: {stats['total_files_scanned']}")
    print(f"成功复制文件: {stats['total_files_copied']}")
    print(f"发现重复文件: {stats['duplicates_found']}")
    print(f"删除空目录数: {stats['empty_dirs_removed']}")
    print(f"错误数量: {stats['errors']}")
    print("="*60)
    
    # 统计最终结果
    final_dirs = [d for d in TARGET_DIR.iterdir() if d.is_dir()]
    print(f"\n最终角色目录数: {len(final_dirs)}")
    
    # 保存统计报告
    report_file = PROJECT_ROOT / "docs" / "merged_english_dataset_整理报告.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# merged_english_dataset 目录整理报告\n\n")
        f.write(f"**整理时间**: {os.popen('date').read().strip()}\n\n")
        f.write("## 统计概览\n\n")
        f.write(f"- 总角色目录数: {stats['total_dirs']}\n")
        f.write(f"- 重命名目录数: {stats['renamed_dirs']}\n")
        f.write(f"- 合并目录数: {stats['merged_dirs']}\n")
        f.write(f"- 扫描文件总数: {stats['total_files_scanned']}\n")
        f.write(f"- 成功复制文件: {stats['total_files_copied']}\n")
        f.write(f"- 发现重复文件: {stats['duplicates_found']}\n")
        f.write(f"- 删除空目录数: {stats['empty_dirs_removed']}\n")
        f.write(f"- 错误数量: {stats['errors']}\n")
        f.write(f"- 最终角色目录数: {len(final_dirs)}\n\n")
        f.write("## 角色目录列表\n\n")
        f.write("| 序号 | 角色名称 |\n")
        f.write("|------|----------|\n")
        for idx, role_dir in enumerate(sorted(final_dirs, key=lambda x: x.name), 1):
            file_count = len([f for f in role_dir.iterdir() if f.is_file()])
            f.write(f"| {idx} | {role_dir.name} ({file_count} 张) |\n")
    
    print(f"\n报告已保存到: {report_file}")


if __name__ == "__main__":
    organize_merged_dataset()
