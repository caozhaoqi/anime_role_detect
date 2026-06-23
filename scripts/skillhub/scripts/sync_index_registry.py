#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同步 index 和 registry 数据
将 skill_index.json 中的技能同步到 registry.json
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ardc.store.index import SkillIndex
from ardc.store.registry import SkillRegistry
from ardc.store.metadata import SkillMetadata
from datetime import datetime


def sync_index_to_registry():
    print("正在同步 index 和 registry 数据...")

    index = SkillIndex()
    registry = SkillRegistry()

    index_skills = index.get_all_skills()
    print(f"index 中共有 {len(index_skills)} 个技能")

    registry_skill_ids = set(registry._registry.keys())
    print(f"registry 中共有 {len(registry_skill_ids)} 个技能")

    synced_count = 0
    for skill in index_skills:
        if skill.id not in registry_skill_ids:
            print(f"  同步技能: {skill.id} ({skill.name})")
            registry.register_skill(skill, release_notes="从 index 同步")
            synced_count += 1

    if synced_count > 0:
        print(f"\n✅ 成功同步 {synced_count} 个技能")
    else:
        print("\n✅ index 和 registry 数据已一致")

    # 验证同步结果
    new_registry_skill_ids = set(registry._registry.keys())
    print(f"\n同步后 registry 中共有 {len(new_registry_skill_ids)} 个技能")


if __name__ == "__main__":
    sync_index_to_registry()
