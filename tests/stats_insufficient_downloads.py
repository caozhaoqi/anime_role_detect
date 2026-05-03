#!/usr/bin/env python3
"""统计下载不足的角色"""
import os
import sys
from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "data" / "role_images.db"

def main():
    print("=" * 80)
    print(" 📊 下载不足角色统计")
    print("=" * 80)

    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # 获取每个角色的待下载数量
        cursor.execute('''
            SELECT role_name, COUNT(*) as pending_count
            FROM raw_urls 
            WHERE status = "pending" 
            GROUP BY role_name 
            ORDER BY pending_count DESC
        ''')
        pending_by_role = {row[0]: row[1] for row in cursor.fetchall()}

        # 获取每个角色的已下载数量
        cursor.execute('''
            SELECT role_name, COUNT(*) as downloaded_count
            FROM downloaded_images 
            WHERE status = "success" 
            GROUP BY role_name
        ''')
        downloaded_by_role = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        # 合并统计
        stats = []
        all_roles = set(pending_by_role.keys()) | set(downloaded_by_role.keys())

        for role in all_roles:
            pending = pending_by_role.get(role, 0)
            downloaded = downloaded_by_role.get(role, 0)
            total = pending + downloaded
            stats.append({
                'role': role,
                'pending': pending,
                'downloaded': downloaded,
                'total': total,
                'pct': (downloaded / total * 100) if total > 0 else 0
            })

        # 按总数量排序
        stats.sort(key=lambda x: x['total'], reverse=True)

        print(f"\n📋 角色总数: {len(stats)}")
        print(f"📊 待下载URL总数: {sum(s['pending'] for s in stats):,}")
        print(f"✅ 已下载URL总数: {sum(s['downloaded'] for s in stats):,}")

        # 定义阈值
        thresholds = [
            {'name': '严重不足', 'max_total': 100, 'color': '🔴'},
            {'name': '数量较少', 'max_total': 200, 'color': '🟡'},
            {'name': '基本充足', 'max_total': 500, 'color': '🟢'},
        ]

        for threshold in thresholds:
            filtered = [s for s in stats if s['total'] <= threshold['max_total']]
            if filtered:
                print(f"\n{threshold['color']} {threshold['name']} (<= {threshold['max_total']}):")
                print(f"  {'角色':<20} {'已下载':>8} {'待下载':>8} {'总计':>8} {'进度':>6}")
                print("  " + "-" * 60)
                for s in filtered:
                    print(f"  {s['role']:<20} {s['downloaded']:>8} {s['pending']:>8} {s['total']:>8} {s['pct']:>5.1f}%")

        # 下载进度滞后的角色（已下载<50%）
        lagging = [s for s in stats if s['total'] > 0 and s['pct'] < 50 and s['pending'] > 0]
        if lagging:
            print(f"\n⚠️ 下载进度滞后（已下载<50%）:")
            print(f"  {'角色':<20} {'已下载':>8} {'待下载':>8} {'总计':>8} {'进度':>6}")
            print("  " + "-" * 60)
            for s in sorted(lagging, key=lambda x: x['pct']):
                print(f"  {s['role']:<20} {s['downloaded']:>8} {s['pending']:>8} {s['total']:>8} {s['pct']:>5.1f}%")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"❌ 统计失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
