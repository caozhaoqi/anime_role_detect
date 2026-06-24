#!/usr/bin/env python3
"""
一键注册所有技能到技能中心
扫描 skills/ 目录下的所有 skill.json，通过 API 注册到注册中心
"""

import json
import sys
import requests
from pathlib import Path

API_BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/api"
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

def load_skills():
    skills = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_json = skill_dir / "skill.json"
        if not skill_json.exists():
            print(f"  ⚠️ 跳过 {skill_dir.name}: 无 skill.json")
            continue
        with open(skill_json) as f:
            data = json.load(f)
        skills.append(data)
    return skills

def register_skill(skill, token):
    url = f"{API_BASE}/skills"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "id": skill["id"],
        "name": skill["name"],
        "version": skill["version"],
        "description": skill.get("description", ""),
        "author": skill.get("author", "ARD Team"),
        "category": skill.get("category", "utility"),
        "entry_point": skill.get("entry_point", ""),
        "tags": skill.get("tags", []),
        "release_notes": f"初始注册 v{skill['version']}",
    }
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code == 200:
        return True, resp.json().get("message", "")
    elif resp.status_code == 500 and "已存在" in resp.text:
        return True, "已存在（跳过）"
    else:
        return False, f"{resp.status_code}: {resp.text[:100]}"

def main():
    token = sys.argv[2] if len(sys.argv) > 2 else input("请输入 API Token: ")

    print(f"📂 扫描技能目录: {SKILLS_DIR}")
    skills = load_skills()
    print(f"  找到 {len(skills)} 个技能\n")

    success = 0
    failed = 0
    for skill in skills:
        print(f"📦 [{skill['id']}] {skill['name']} v{skill['version']} ... ", end="", flush=True)
        ok, msg = register_skill(skill, token)
        if ok:
            print(f"✅ {msg}")
            success += 1
        else:
            print(f"❌ {msg}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"完成: {success} 成功, {failed} 失败, {len(skills)} 总计")

if __name__ == "__main__":
    main()