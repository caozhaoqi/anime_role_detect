#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能安装功能测试脚本
"""

import sys
import os

# 设置环境变量
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ardc.store.registry import SkillRegistry
from ardc.store.metadata import SkillMetadata
from datetime import datetime, timezone


def test_skill_installation():
    """测试技能安装功能"""

    print("=" * 60)
    print("技能安装功能测试")
    print("=" * 60)

    # 1. 创建技能注册表
    print("\n[步骤 1] 初始化技能注册表...")
    registry = SkillRegistry()
    print("✅ 技能注册表初始化成功")

    # 2. 创建测试技能元数据
    print("\n[步骤 2] 创建测试技能...")
    test_skill = SkillMetadata(
        id="test-skill-demo",
        name="测试演示技能",
        version="1.0.0",
        description="这是一个用于测试安装功能的技能",
        author="test-user",
        category="utility",
        entry_point="main.py",
        tags=["test", "demo"],
    )
    print(f"✅ 创建测试技能: {test_skill.id} v{test_skill.version}")

    # 3. 注册技能
    print("\n[步骤 3] 注册技能...")
    success = registry.register_skill(test_skill, "初始测试版本")
    if success:
        print("✅ 技能注册成功")
    else:
        print("⚠️  技能已存在，继续测试")

    # 4. 安装技能
    print("\n[步骤 4] 安装技能...")

    # 先检查技能是否在注册表中
    print(f"注册表中的技能: {list(registry._registry.keys())}")
    if "test-skill-demo" in registry._registry:
        print(f"技能版本: {list(registry._registry['test-skill-demo'].keys())}")

    # 检查技能是否已安装
    print(f"已安装技能: {list(registry._installed_skills.keys())}")

    # 如果已安装，先卸载
    if "test-skill-demo" in registry._installed_skills:
        print("技能已安装，先卸载...")
        registry.uninstall_skill("test-skill-demo")
        print("✅ 卸载成功")

    try:
        install_success = registry.install_skill("test-skill-demo", "1.0.0")
        if install_success:
            print("✅ 技能安装成功！")
        else:
            print("❌ 技能安装失败（返回 False）")
            return False
    except Exception as e:
        print(f"❌ 安装过程出错: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 5. 验证安装结果
    print("\n[步骤 5] 验证安装结果...")
    installed_skills = registry.list_installed_skills()
    print(f"已安装技能数量: {len(installed_skills)}")

    for skill_info in installed_skills:
        print(f"\n技能 ID: {skill_info.metadata.id}")
        print(f"  名称: {skill_info.metadata.name}")
        print(f"  版本: {skill_info.metadata.version}")
        print(f"  安装路径: {skill_info.install_path}")
        print(f"  安装时间: {skill_info.installed_at}")
        print(f"  启用状态: {skill_info.enabled}")

    # 6. 检查安装目录
    print("\n[步骤 6] 检查安装目录...")
    skill_dir = os.path.expanduser("~/.ardc/skills/test-skill-demo")
    if os.path.exists(skill_dir):
        print(f"✅ 技能目录存在: {skill_dir}")
        print("目录内容:")
        for item in os.listdir(skill_dir):
            item_path = os.path.join(skill_dir, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                for sub_item in os.listdir(item_path):
                    print(f"      - {sub_item}")
            else:
                print(f"  📄 {item}")
    else:
        print(f"❌ 技能目录不存在: {skill_dir}")
        return False

    # 7. 测试技能执行
    print("\n[步骤 7] 测试技能执行...")
    script_path = os.path.join(skill_dir, "scripts", "main.py")
    if os.path.exists(script_path):
        print(f"✅ 技能脚本存在: {script_path}")

        # 读取脚本内容
        with open(script_path, "r", encoding="utf-8") as f:
            script_content = f.read()
        print("\n脚本内容预览:")
        print("-" * 40)
        print(script_content[:200] + "...")
        print("-" * 40)
    else:
        print(f"❌ 技能脚本不存在: {script_path}")
        return False

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_skill_installation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
