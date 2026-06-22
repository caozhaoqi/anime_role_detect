#!/usr/bin/env python3
"""配置验证脚本 - 验证项目配置的完整性"""
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigValidator:
    """配置验证器"""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """验证所有配置"""
        logger.info("开始验证项目配置...")

        self._validate_directory_structure()
        self._validate_required_files()
        self._validate_data_directories()
        self._validate_model_directories()
        self._validate_python_environment()

        self._report_results()
        return len(self.errors) == 0

    def _validate_directory_structure(self) -> None:
        """验证目录结构"""
        required_dirs = [
            "src",
            "data",
            "models",
            "logs",
            "scripts",
        ]

        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self.errors.append(f"缺少必需目录: {dir_name}")
            elif not dir_path.is_dir():
                self.errors.append(f"路径不是目录: {dir_name}")

    def _validate_required_files(self) -> None:
        """验证必需文件"""
        required_files = [
            "README.md",
            ".gitignore",
            "requirements.txt",
            ".env.example",
        ]

        for file_name in required_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                self.warnings.append(f"缺少推荐文件: {file_name}")

    def _validate_data_directories(self) -> None:
        """验证数据目录"""
        data_dir = self.project_root / "data"

        if not data_dir.exists():
            self.errors.append("数据目录不存在")
            return

        required_subdirs = [
            "training_dataset",
            "final_dataset",
        ]

        for subdir in required_subdirs:
            subdir_path = data_dir / subdir
            if not subdir_path.exists():
                self.warnings.append(f"数据子目录不存在: {subdir}")
            elif not any(subdir_path.iterdir()):
                self.warnings.append(f"数据子目录为空: {subdir}")

    def _validate_model_directories(self) -> None:
        """验证模型目录"""
        models_dir = self.project_root / "models"

        if not models_dir.exists():
            self.warnings.append("模型目录不存在")
            return

        model_files = list(models_dir.glob("*.pth"))
        if not model_files:
            self.warnings.append("模型目录中没有找到模型文件")

    def _validate_python_environment(self) -> None:
        """验证Python环境"""
        venv_dir = self.project_root / ".venv"

        if not venv_dir.exists():
            self.warnings.append("虚拟环境不存在")
            return

        python_exe = venv_dir / "bin" / "python" if os.name != "nt" else venv_dir / "Scripts" / "python.exe"

        if not python_exe.exists():
            self.warnings.append("虚拟环境Python可执行文件不存在")

        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                import pkg_resources
                with open(requirements_file) as f:
                    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

                missing_packages = []
                for req in requirements:
                    try:
                        pkg_resources.require(req)
                    except pkg_resources.DistributionNotFound:
                        missing_packages.append(req)

                if missing_packages:
                    self.warnings.append(f"缺少依赖包: {', '.join(missing_packages)}")
            except Exception as e:
                self.warnings.append(f"无法验证依赖包: {e}")

    def _report_results(self) -> None:
        """报告验证结果"""
        logger.info("\n" + "=" * 60)
        logger.info("配置验证结果")
        logger.info("=" * 60)

        if not self.errors and not self.warnings:
            logger.info("✅ 所有配置验证通过")
            return

        if self.errors:
            logger.error(f"\n❌ 发现 {len(self.errors)} 个错误:")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        if self.errors:
            logger.error("\n请修复上述错误后再继续")
        elif self.warnings:
            logger.info("\n警告不影响基本功能，但建议修复")


def main():
    project_root = Path(__file__).parent.parent.parent

    validator = ConfigValidator(project_root)
    is_valid = validator.validate()

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()