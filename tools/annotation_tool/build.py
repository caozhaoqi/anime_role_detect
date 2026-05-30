#!/usr/bin/env python3
"""
动漫角色标注工具 - 跨平台打包脚本
支持 Windows、Mac、Linux 平台

使用方法:
    # 安装依赖
    pip install pyinstaller

    # 打包当前平台
    python build.py

    # 指定平台打包
    python build.py --platform windows
    python build.py --platform mac
    python build.py --platform linux
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def build_app(platform=None):
    """打包应用程序"""
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / "annotation_tool_desktop.py"
    app_name = "anime_role_annotator"

    if platform is None:
        # 自动检测平台
        if sys.platform.startswith("win"):
            platform = "windows"
        elif sys.platform.startswith("darwin"):
            platform = "mac"
        elif sys.platform.startswith("linux"):
            platform = "linux"
        else:
            print(f"不支持的平台: {sys.platform}")
            return False

    print(f"正在为 {platform} 平台打包...")

    # PyInstaller 基本参数
    cmd = [
        "pyinstaller",
        "--name",
        app_name,
        "--windowed",
        "--onefile",
        "--icon",
        str(current_dir / "icon.ico"),
        "--add-data",
        f"{current_dir}/roles.json:.",
        "--add-data",
        f"{current_dir}/annotations/:annotations/",
    ]

    # 平台特定参数
    if platform == "mac":
        cmd.extend(
            [
                "--target-arch",
                "universal2",
                "--osx-bundle-identifier",
                "com.example.animeroleannotator",
            ]
        )
    elif platform == "windows":
        cmd.extend(
            [
                "--version-file",
                "version_info.txt",
            ]
        )

    cmd.append(str(script_path))

    try:
        subprocess.run(cmd, check=True, cwd=current_dir)
        print(f"{platform} 平台打包完成！")
        print(f"输出目录: {current_dir / 'dist'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 打包失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="打包动漫角色标注工具")
    parser.add_argument(
        "--platform", choices=["windows", "mac", "linux"], help="目标平台 (默认自动检测)"
    )
    args = parser.parse_args()

    # 检查 pyinstaller 是否安装
    try:
        subprocess.run(["pyinstaller", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("❌ 请先安装 pyinstaller: pip install pyinstaller")
        sys.exit(1)

    success = build_app(args.platform)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
