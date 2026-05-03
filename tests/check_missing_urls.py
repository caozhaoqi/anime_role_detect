#!/usr/bin/env python3
"""检查缺少URL的角色"""
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "data" / "role_images.db"

def main():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # 统计各角色URL数量
    cursor.execute('SELECT role_name, COUNT(1) FROM raw_urls GROUP BY role_name HAVING COUNT(1) < 100 ORDER BY COUNT(1)')
    low_url_roles = cursor.fetchall()

    # 统计完全没有URL的角色
    cursor.execute('SELECT r.name FROM roles r LEFT JOIN raw_urls u ON r.name = u.role_name WHERE u.id IS NULL')
    no_url_roles = cursor.fetchall()

    # 统计已下载的情况
    cursor.execute('SELECT COUNT(DISTINCT role_name) FROM raw_urls')
    total_roles_with_url = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(DISTINCT role_name) FROM downloaded_images WHERE status = "success"')
    total_downloaded = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM roles')
    total_roles = cursor.fetchone()[0]

    conn.close()

    print("=" * 60)
    print(" 📊 角色URL采集状态统计")
    print("=" * 60)
    print(f"\n总角色数: {total_roles}")
    print(f"有URL的角色: {total_roles_with_url}")
    print(f"已下载的角色: {total_downloaded}")
    print(f"URL少于100的角色: {len(low_url_roles)}")
    print(f"完全没有URL的角色: {len(no_url_roles)}")

    if no_url_roles:
        print(f"\n❌ 完全没有URL的角色:")
        for (role,) in no_url_roles[:10]:
            print(f"  • {role}")
        if len(no_url_roles) > 10:
            print(f"  ... 还有 {len(no_url_roles) - 10} 个角色")

    if low_url_roles:
        print(f"\n⚠️ URL不足100的角色:")
        for role, count in low_url_roles[:15]:
            print(f"  • {role}: {count} 个URL")
        if len(low_url_roles) > 15:
            print(f"  ... 还有 {len(low_url_roles) - 15} 个角色")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
