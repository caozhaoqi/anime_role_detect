#!/usr/bin/env python3
"""清理旧的初始化技能数据"""
import sys
sys.path.insert(0, '/Users/caozhaoqi/PycharmProjects/anime_role_detect')
from scripts.database import SessionLocal, Skill

# 旧技能列表（需要删除的）
OLD_SKILL_IDS = [
    "ardc-cleaner",     # 旧版本
    "ardc-collector",   # 旧版本  
    "ardc-classifier",  # 旧版本
    "ardc-trainer",     # 旧版本
    "ardc-search",      # 旧版本
    "ardc-analyzer",    # 旧版本
    "ardc-utility"      # 旧版本
]

def main():
    db = SessionLocal()
    try:
        print("清理旧技能数据...")
        
        # 删除所有英文名的旧技能（以"ARD "开头的）
        old_skills = db.query(Skill).filter(Skill.name.like("ARD %")).all()
        for skill in old_skills:
            print(f"✓ 删除旧版本: {skill.name} ({skill.skill_id}) v{skill.version}")
            db.delete(skill)
        
        db.commit()
        print(f"\n清理完成！共删除 {len(old_skills)} 个旧技能")
        
        # 显示当前技能列表
        print("\n当前技能列表:")
        skills = db.query(Skill).all()
        for skill in skills:
            print(f"  - {skill.name} ({skill.skill_id}) v{skill.version}")
        
        print(f"\n总技能数: {len(skills)}")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
