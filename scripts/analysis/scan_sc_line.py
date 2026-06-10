import os
import sys
import datetime

# 扫描目录下所有 .py 文件的代码行数

def count_py_lines(target_dir, output_filename="code_lines_report.txt"):
    # 默认排除的目录，避免统计虚拟环境、缓存或版本控制文件
    exclude_dirs = {
        '.git', '.venv', 'venv', 'env', '.env', 
        '__pycache__', '.idea', '.vscode', 'build', 'dist'
    }
    
    total_files = 0
    total_lines = 0
    details = []

    # 递归遍历目录
    for root, dirs, files in os.walk(target_dir):
        # 过滤排除目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    # 使用 utf-8 编码读取，忽略解码错误以防非文本文件干扰
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        
                        total_lines += line_count
                        total_files += 1
                        
                        # 记录相对路径和行数
                        rel_path = os.path.relpath(file_path, target_dir)
                        details.append((rel_path, line_count))
                except Exception as e:
                    print(f"[警告] 无法读取文件 {file_path}: {e}")

    # 按代码行数降序排序
    details.sort(key=lambda x: x[1], reverse=True)

    # 构造输出报告的内容
    report = []
    report.append("="*70)
    report.append(f"Python 代码行数统计报告")
    report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"扫描目录: {os.path.abspath(target_dir)}")
    report.append(f"有效 .py 文件数: {total_files} 个")
    report.append(f"总代码行数: {total_lines} 行")
    report.append("="*70 + "\n")
    
    if details:
        report.append("详细清单 (按行数降序排列):")
        for path, count in details:
            report.append(f" - {count:<6} 行 | {path}")
    else:
        report.append("未在指定目录下检测到任何 .py 文件。")

    report_content = "\n".join(report)

    # 将内容写入到 TXT 文件中
    try:
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            out_f.write(report_content)
        print(f"\n[成功] 统计完成！报告已成功写入到本地文件:\n -> {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"\n[错误] 写入 TXT 文件失败: {e}")

if __name__ == "__main__":
    # 若不带参数，默认统计当前目录 "."
    path_to_scan = sys.argv[1] if len(sys.argv) > 1 else '.'
    count_py_lines(path_to_scan)