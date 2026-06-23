#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖自动安装模块

功能：
- 自动检测并安装缺失的依赖包
- 支持国内镜像源（清华、阿里云、豆瓣）
- 区分基础依赖和扩展依赖
- 提供友好的安装进度反馈
"""

import os
import sys
import subprocess
import site
from pathlib import Path
from typing import List, Optional, Tuple

# 国内镜像源配置
MIRRORS = {
    "tsinghua": {
        "name": "清华镜像",
        "url": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "priority": 1,
    },
    "aliyun": {
        "name": "阿里云镜像",
        "url": "https://mirrors.aliyun.com/pypi/simple",
        "priority": 2,
    },
    "douban": {
        "name": "豆瓣镜像",
        "url": "https://pypi.doubanio.com/simple",
        "priority": 3,
    },
    "official": {
        "name": "官方源",
        "url": "https://pypi.python.org/simple",
        "priority": 99,
    },
}

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class DependencyManager:
    """依赖管理器"""

    def __init__(self, mirror_priority: List[str] = None):
        """
        初始化依赖管理器

        Args:
            mirror_priority: 镜像源优先级列表，默认 ["tsinghua", "aliyun", "douban"]
        """
        self.mirror_priority = mirror_priority or ["tsinghua", "aliyun", "douban"]
        self.installed_packages = set()
        self.failed_packages = {}
        self._load_installed_packages()

    def _load_installed_packages(self):
        """加载已安装的包列表"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "==" in line:
                        pkg_name = line.split("==")[0].lower().replace("-", "_")
                        self.installed_packages.add(pkg_name)
        except Exception:
            pass

    def _get_pip_command(self) -> List[str]:
        """获取 pip 命令（兼容虚拟环境）"""
        # 优先使用当前 Python 的 pip
        python_path = sys.executable
        pip_path = str(Path(python_path).parent / "pip")

        # 检查虚拟环境中的 pip
        venv_pip = Path(python_path).parent / "Scripts" / "pip.exe"  # Windows
        if venv_pip.exists():
            return [str(venv_pip)]

        venv_pip = Path(python_path).parent / "bin" / "pip"  # Linux/macOS
        if venv_pip.exists():
            return [str(venv_pip)]

        # 回退到 python -m pip
        return [python_path, "-m", "pip"]

    def _build_pip_index_url(self) -> str:
        """构建 pip 镜像 URL"""
        for mirror_name in self.mirror_priority:
            if mirror_name in MIRRORS:
                return f"-i {MIRRORS[mirror_name]['url']}"
        return ""

    def is_installed(self, package_name: str) -> bool:
        """检查包是否已安装"""
        normalized = package_name.lower().replace("-", "_").split("==")[0]

        # 特殊映射
        alias_map = {
            "pil": "pillow",
            "pillow": "pillow",
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "yaml": "pyyaml",
        }
        check_names = [normalized]
        if normalized in alias_map:
            check_names.append(alias_map[normalized].replace("-", "_"))

        for name in check_names:
            if name in self.installed_packages:
                return True
        return False

    def install_package(
        self, package: str, verbose: bool = True, upgrade: bool = False
    ) -> Tuple[bool, str]:
        """
        安装单个包

        Args:
            package: 包名（可带版本号）
            verbose: 是否显示详细输出
            upgrade: 是否强制升级

        Returns:
            (成功标志, 消息)
        """
        package_name = package.split("==")[0]

        if self.is_installed(package_name) and not upgrade:
            return True, f"{package_name} 已安装"

        pip_cmd = self._get_pip_command()
        index_url = self._build_pip_index_url()

        cmd = pip_cmd + ["install"]
        if upgrade:
            cmd.append("--upgrade")
        if index_url:
            cmd.extend(index_url.split())
        cmd.append(package)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                normalized = package_name.lower().replace("-", "_")
                self.installed_packages.add(normalized)
                msg = f"✅ {package_name} 安装成功"
                if verbose:
                    print(msg)
                return True, msg
            else:
                error_msg = result.stderr.strip().split("\n")[-1]
                msg = f"❌ {package_name} 安装失败: {error_msg}"
                if verbose:
                    print(msg)
                self.failed_packages[package_name] = error_msg
                return False, msg

        except subprocess.TimeoutExpired:
            msg = f"❌ {package_name} 安装超时"
            if verbose:
                print(msg)
            self.failed_packages[package_name] = "安装超时"
            return False, msg
        except Exception as e:
            msg = f"❌ {package_name} 安装异常: {str(e)}"
            if verbose:
                print(msg)
            self.failed_packages[package_name] = str(e)
            return False, msg

    def install_requirements(
        self, requirements_file: str, verbose: bool = True
    ) -> Tuple[int, int]:
        """
        从 requirements 文件安装依赖

        Args:
            requirements_file: requirements 文件路径
            verbose: 是否显示详细输出

        Returns:
            (成功数量, 失败数量)
        """
        req_path = Path(requirements_file)
        if not req_path.exists():
            return 0, 0

        success_count = 0
        fail_count = 0

        with open(req_path, "r", encoding="utf-8") as f:
            packages = []
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if not line or line.startswith("#"):
                    continue
                # 处理 -r 递归引用
                if line.startswith("-r "):
                    sub_file = line[3:].strip()
                    sub_path = req_path.parent / sub_file
                    if sub_path.exists():
                        packages.extend(self._parse_requirements_file(sub_path))
                    continue
                # 提取包名（去掉版本号的版本部分用于检查）
                pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                packages.append(line)

        if verbose:
            print(f"\n📦 从 {req_path.name} 安装 {len(packages)} 个包...")

        for package in packages:
            pkg_name = package.split("==")[0]
            if self.is_installed(pkg_name):
                if verbose:
                    print(f"  ⏭️  {pkg_name} 已安装，跳过")
                success_count += 1
            else:
                success, _ = self.install_package(package, verbose=verbose)
                if success:
                    success_count += 1
                else:
                    fail_count += 1

        return success_count, fail_count

    def _parse_requirements_file(self, path: Path) -> List[str]:
        """解析 requirements 文件"""
        packages = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-r "):
                    continue
                packages.append(line)
        return packages

    def install_core(self, verbose: bool = True) -> Tuple[int, int]:
        """
        安装核心依赖（ML 优先，再基础依赖）

        安装顺序：
        1. requirements-ml.txt (深度学习、numpy 等，避免版本冲突)
        2. requirements-base.txt (Web 框架等基础服务)

        Returns:
            (成功数量, 失败数量)
        """
        print("\n" + "=" * 60)
        print("📦 安装核心依赖")
        print("=" * 60)

        ml_file = PROJECT_ROOT / "requirements-ml.txt"
        base_file = PROJECT_ROOT / "requirements-base.txt"

        total_success = 0
        total_fail = 0

        # 1. 先安装 ML 依赖（包含 torch, numpy 等核心包）
        if ml_file.exists():
            print("\n📦 [1/2] 安装机器学习依赖 (torch, numpy 等)...")
            s, f = self.install_requirements(str(ml_file), verbose)
            total_success += s
            total_fail += f

        # 2. 再安装基础依赖（Web 服务等）
        if base_file.exists():
            print("\n📦 [2/2] 安装基础依赖 (fastapi, uvicorn 等)...")
            s, f = self.install_requirements(str(base_file), verbose)
            total_success += s
            total_fail += f

        return total_success, total_fail

    def install_all(self, verbose: bool = True) -> Tuple[int, int]:
        """
        安装所有依赖

        Returns:
            (成功数量, 失败数量)
        """
        print("\n" + "=" * 60)
        print("📦 安装所有依赖")
        print("=" * 60)

        requirements_file = PROJECT_ROOT / "requirements.txt"

        if requirements_file.exists():
            return self.install_requirements(str(requirements_file), verbose)

        # 如果没有总 requirements，回退到核心依赖
        return self.install_core(verbose)

    def ensure_essential(self) -> bool:
        """
        确保核心包已安装（用于启动前检查）

        Returns:
            是否所有核心包都可用
        """
        essential_packages = [
            "numpy",
            "torch",
            "fastapi",
            "uvicorn",
            "PIL",
        ]

        missing = []
        for pkg in essential_packages:
            if not self.is_installed(pkg):
                missing.append(pkg)

        if missing:
            print(f"\n⚠️  缺少核心依赖: {', '.join(missing)}")
            print("请运行以下命令安装:")
            print(f"  python -m src.core.dependency_manager install-core")
            return False

        return True

    def get_status(self) -> dict:
        """获取依赖状态"""
        return {
            "installed_count": len(self.installed_packages),
            "failed_count": len(self.failed_packages),
            "failed_packages": self.failed_packages,
        }


def show_mirror_info():
    """显示镜像源信息"""
    print("\n可用镜像源:")
    print("-" * 40)
    for name, info in sorted(MIRRORS.items(), key=lambda x: x[1]["priority"]):
        print(f"  {info['name']}: {info['url']}")
    print("-" * 40)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="依赖自动安装工具")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["install", "install-core", "check", "mirrors"],
        default="check",
        help="操作类型",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--mirror", "-m", help="指定镜像源")
    parser.add_argument("--package", "-p", help="安装指定包")

    args = parser.parse_args()

    # 指定镜像源
    mirror_priority = ["tsinghua", "aliyun", "douban"]
    if args.mirror:
        mirror_priority = [args.mirror]
        print(f"使用镜像源: {MIRRORS.get(args.mirror, {}).get('name', args.mirror)}")

    manager = DependencyManager(mirror_priority=mirror_priority)

    if args.action == "mirrors":
        show_mirror_info()
        return

    if args.action == "check":
        print("\n" + "=" * 60)
        print("🔍 检查依赖状态")
        print("=" * 60)
        status = manager.get_status()
        print(f"已安装包数量: {status['installed_count']}")

        if status["failed_count"] > 0:
            print(f"\n安装失败的包:")
            for pkg, err in status["failed_packages"].items():
                print(f"  - {pkg}: {err}")

        if manager.ensure_essential():
            print("\n✅ 核心依赖检查通过")
        else:
            print("\n❌ 核心依赖检查失败，请运行安装")
        return

    if args.action == "install-core":
        success, fail = manager.install_core(verbose=args.verbose)
        print(f"\n安装完成: ✅ {success} 个, ❌ {fail} 个")
        return

    if args.action == "install":
        # 安装指定包
        if args.package:
            success, msg = manager.install_package(args.package, verbose=True)
            print(msg)
        else:
            # 安装所有
            success, fail = manager.install_all(verbose=args.verbose)
            print(f"\n安装完成: ✅ {success} 个, ❌ {fail} 个")
        return


if __name__ == "__main__":
    main()
