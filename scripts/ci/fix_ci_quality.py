#!/usr/bin/env python3
"""
CI 代码质量检测脚本
检测并修复 flake8 E501 (行太长) 和 black 格式化问题
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, cwd=None):
    """执行命令并返回结果"""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def check_flake8_e501(path=".", exclude_dirs=None):
    """检测 flake8 E501 错误"""
    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__", "env", "venv", "node_modules", "archived"]

    exclude_args = ""
    for ex_dir in exclude_dirs:
        exclude_args += f' --exclude="*/{ex_dir}/*"'

    cmd = f"flake8 {path} --select=E501 {exclude_args}"
    returncode, stdout, stderr = run_command(cmd)

    errors = []
    for line in stdout.splitlines():
        if line.strip():
            errors.append(line)

    return errors


def check_black_format(path=".", exclude_dirs=None):
    """检测 black 格式化问题"""
    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__", "env", "venv", "node_modules", "archived"]

    exclude_args = ""
    for ex_dir in exclude_dirs:
        exclude_args += f' --exclude=".*/{ex_dir}/.*"'

    cmd = f"black --check {path} {exclude_args}"
    returncode, stdout, stderr = run_command(cmd)

    files_to_format = []
    for line in stdout.splitlines():
        if "would be reformatted" in line:
            files_to_format.append(line)

    return returncode != 0, files_to_format


def fix_black_format(path=".", exclude_dirs=None):
    """修复 black 格式化问题"""
    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__", "env", "venv", "node_modules", "archived"]

    exclude_args = ""
    for ex_dir in exclude_dirs:
        exclude_args += f' --exclude=".*/{ex_dir}/.*"'

    cmd = f"black {path} {exclude_args}"
    return run_command(cmd)


def fix_flake8_e501_manual(files):
    """手动修复 E501 错误（通过换行）"""
    fixed_files = []

    for file_path in files:
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        changed = False

        for i, line in enumerate(lines):
            if len(line.rstrip()) > 79 and not line.strip().startswith("#"):
                # 尝试自动修复：找到合适的位置换行
                stripped = line.rstrip()

                # 如果包含赋值操作符，在操作符处换行
                if " = " in stripped and len(stripped) <= 100:
                    parts = stripped.split(" = ")
                    if len(parts) == 2 and len(parts[0]) < 40:
                        new_line = f"{parts[0]} = \\\n    {parts[1].strip()}\n"
                        new_lines.append(new_line)
                        changed = True
                        continue

                # 如果包含逗号分割的参数，在逗号处换行
                if ", " in stripped and len(stripped) <= 120:
                    new_lines.append(line)
                    continue

                # 如果包含函数调用，在左括号后换行
                if "(" in stripped and ")" in stripped:
                    new_lines.append(line)
                    continue

                new_lines.append(line)
            else:
                new_lines.append(line)

        if changed:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            fixed_files.append(file_path)

    return fixed_files


def main():
    project_root = Path(__file__).parent.parent

    print("=" * 60)
    print("🔍 CI 代码质量检测")
    print("=" * 60)
    print()

    # 检测 flake8 E501 错误
    print("📋 检测 flake8 E501 错误...")
    e501_errors = check_flake8_e501(project_root)

    if e501_errors:
        print(f"❌ 发现 {len(e501_errors)} 处 E501 错误!")
        print()

        # 按文件分组
        error_by_file = {}
        for error in e501_errors:
            parts = error.split(":")
            if len(parts) >= 3:
                file_path = ":".join(parts[:-2])
                line_num = parts[-2]
                message = parts[-1]
                if file_path not in error_by_file:
                    error_by_file[file_path] = []
                error_by_file[file_path].append((line_num, message.strip()))

        print("错误分布:")
        for file_path, errors in sorted(error_by_file.items()):
            print(f"  📄 {file_path}: {len(errors)} 处错误")

        print()
        print("详细错误:")
        for error in e501_errors[:20]:  # 只显示前20条
            print(f"  {error}")

        if len(e501_errors) > 20:
            print(f"  ... 还有 {len(e501_errors) - 20} 处错误")

    else:
        print("✅ 没有发现 E501 错误")

    print()

    # 检测 black 格式化问题
    print("📋 检测 black 格式化问题...")
    has_issues, files_to_format = check_black_format(project_root)

    if has_issues:
        print(f"❌ 发现 {len(files_to_format)} 个文件需要格式化!")
        for f in files_to_format[:10]:
            print(f"  {f}")
        if len(files_to_format) > 10:
            print(f"  ... 还有 {len(files_to_format) - 10} 个文件")

        print()
        response = input("是否自动修复? (y/n): ")
        if response.lower() == "y":
            print("🔧 正在修复 black 格式化问题...")
            returncode, stdout, stderr = fix_black_format(project_root)
            if returncode == 0:
                print("✅ black 格式化问题已修复!")
            else:
                print(f"❌ 修复失败: {stderr}")
    else:
        print("✅ 所有文件已正确格式化")

    print()
    print("=" * 60)
    print("💡 提示:")
    print("  - 使用 flake8 --select=E501 查看所有 E501 错误")
    print("  - 使用 black . 自动修复格式化问题")
    print("  - 手动修复复杂的长行问题时，考虑:")
    print("    1. 将长表达式拆分到多行")
    print("    2. 使用括号包裹长表达式")
    print("    3. 提取为变量或常量")
    print("=" * 60)


if __name__ == "__main__":
    main()
