#!/usr/bin/env python3
"""
RDS MySQL 数据库工具模块
替代本地 SQLite (image_hashes.db) + JSON (.pack_state.json) 存储

用法:
    from db_utils import DB
    DB.load_all_hashes()
    DB.append_hashes({"hash1", "hash2"}, "arlecchino")
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import pymysql

# ── 默认连接配置（优先从环境变量读取） ──
DB_CONFIG = {
    "host": os.environ.get("RDS_HOST", "czq.rwlb.rds.aliyuncs.com"),
    "port": int(os.environ.get("RDS_PORT", "3306")),
    "user": os.environ.get("RDS_USER", "czq"),
    "password": os.environ.get("RDS_PASSWORD", "Caozhaoqi@828079"),
    "database": os.environ.get("RDS_DATABASE", "anime_role_detect"),
    "charset": "utf8mb4",
}


class DB:
    """RDS MySQL 数据库操作类"""

    _conn = None

    # ── 连接管理 ──

    @classmethod
    def _get_conn(cls):
        if cls._conn is None or not cls._is_connected():
            cls._conn = pymysql.connect(**DB_CONFIG)
            cls._conn.autocommit(True)
        return cls._conn

    @classmethod
    def _is_connected(cls):
        try:
            if cls._conn:
                cls._conn.ping(reconnect=True)
                return True
        except Exception:
            pass
        return False

    @classmethod
    def _execute(cls, sql: str, params: tuple = None) -> int:
        """执行 SQL，返回影响行数"""
        conn = cls._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.rowcount

    @classmethod
    def _fetchall(cls, sql: str, params: tuple = None) -> list:
        conn = cls._get_conn()
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()

    @classmethod
    def _fetchone(cls, sql: str, params: tuple = None) -> Optional[dict]:
        rows = cls._fetchall(sql, params)
        return rows[0] if rows else None

    @classmethod
    def close(cls):
        if cls._conn:
            cls._conn.close()
            cls._conn = None

    # ── image_hashes 操作 ──

    @classmethod
    def load_all_hashes(cls) -> Tuple[Set[str], int]:
        """
        加载所有哈希到内存集合
        返回: (hashes_set, total_count)
        """
        conn = cls._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT hash FROM image_hashes")
            rows = cur.fetchall()
            hashes = {r[0] for r in rows}
            total = len(hashes)
        print(f"   已加载全局哈希库: {total} 条 (RDS: {DB_CONFIG['host']})")
        return hashes, total

    @classmethod
    def append_hashes(cls, new_hashes: Set[str], role_name: str) -> int:
        """
        批量写入新哈希到 RDS，重复哈希自动忽略 (INSERT IGNORE)
        返回: 实际插入数
        """
        if not new_hashes:
            return 0
        conn = cls._get_conn()
        inserted = 0
        with conn.cursor() as cur:
            for h in new_hashes:
                # INSERT IGNORE 避免重复
                cur.execute(
                    "INSERT IGNORE INTO image_hashes (hash, roles) VALUES (%s, %s)",
                    (h, role_name),
                )
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    # 已存在的哈希，追加角色名
                    cur.execute(
                        "UPDATE image_hashes SET file_count = file_count + 1, "
                        "roles = CASE WHEN INSTR(roles, %s) = 0 "
                        "             THEN CONCAT(roles, ',', %s) "
                        "             ELSE roles END "
                        "WHERE hash = %s",
                        (role_name, role_name, h),
                    )
        conn.commit()
        print(f"    💾 已持久化 {len(new_hashes)} 个哈希到 RDS (新增 {inserted})")
        return inserted

    # ── pack_state 操作 ──

    @classmethod
    def load_pack_state(cls) -> dict:
        """从 RDS 加载打包状态，格式兼容原有 .pack_state.json"""
        rows = cls._fetchall(
            "SELECT * FROM pack_state ORDER BY pack_number ASC"
        )
        state = {
            "pack_number": 0,
            "packed_files": [],
        }
        for row in rows:
            no = row["pack_number"]
            state["pack_number"] = no
            key = f"pack_{no}"
            pack_files = json.loads(row["pack_files"]) if row["pack_files"] else []
            state[key] = {
                "zip": row["zip_name"],
                "size_mb": float(row["size_mb"]) if row["size_mb"] else 0,
                "new_image_count": row["new_image_count"],
                "total_image_count": row["total_image_count"],
                "pack_files": pack_files,
                "cleaned_count": row["cleaned_count"],
                "cleaned_mb": float(row["cleaned_mb"]) if row["cleaned_mb"] else 0,
                "timestamp": row["created_at"].isoformat() if row["created_at"] else "",
            }
            state["packed_files"].extend(pack_files)
        return state

    @classmethod
    def save_pack_state(cls, state: dict) -> None:
        """将打包状态保存到 RDS（增量更新）"""
        pack_no = state.get("pack_number", 0)
        if pack_no == 0:
            return
        key = f"pack_{pack_no}"
        pack_info = state.get(key, {})

        # 检查包是否已存在
        existing = cls._fetchone(
            "SELECT id FROM pack_state WHERE pack_number = %s", (pack_no,)
        )

        pack_files_json = json.dumps(pack_info.get("pack_files", []), ensure_ascii=False)
        if existing:
            cls._execute(
                "UPDATE pack_state SET zip_name=%s, size_mb=%s, new_image_count=%s, "
                "total_image_count=%s, pack_files=%s, cleaned_count=%s, cleaned_mb=%s "
                "WHERE pack_number=%s",
                (
                    pack_info.get("zip", ""),
                    pack_info.get("size_mb", 0),
                    pack_info.get("new_image_count", 0),
                    pack_info.get("total_image_count", 0),
                    pack_files_json,
                    pack_info.get("cleaned_count", 0),
                    pack_info.get("cleaned_mb", 0),
                    pack_no,
                ),
            )
        else:
            cls._execute(
                "INSERT INTO pack_state (pack_number, zip_name, size_mb, new_image_count, "
                "total_image_count, pack_files, cleaned_count, cleaned_mb) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    pack_no,
                    pack_info.get("zip", ""),
                    pack_info.get("size_mb", 0),
                    pack_info.get("new_image_count", 0),
                    pack_info.get("total_image_count", 0),
                    pack_files_json,
                    pack_info.get("cleaned_count", 0),
                    pack_info.get("cleaned_mb", 0),
                ),
            )

    # ── collection_records 操作 ──

    @classmethod
    def add_collection_record(cls, role_name: str, role_tag: str = "",
                               site: str = "", success_count: int = 0,
                               fail_count: int = 0, total_needed: int = 0,
                               existing_before: int = 0,
                               new_hashes_added: int = 0) -> None:
        """记录一次角色采集结果"""
        cls._execute(
            "INSERT INTO collection_records "
            "(role_name, role_tag, site, success_count, fail_count, "
            " total_needed, existing_before, new_hashes_added) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (role_name, role_tag, site, success_count, fail_count,
             total_needed, existing_before, new_hashes_added),
        )

    # ── system_logs 操作 ──

    @classmethod
    def log(cls, level: str, source: str, message: str) -> None:
        """写入系统日志"""
        cls._execute(
            "INSERT INTO system_logs (level, source, message) VALUES (%s, %s, %s)",
            (level.upper(), source[:100], message),
        )

    @classmethod
    def get_recent_logs(cls, limit: int = 50, level: str = None) -> list:
        """获取最近日志"""
        if level:
            rows = cls._fetchall(
                "SELECT * FROM system_logs WHERE level=%s ORDER BY id DESC LIMIT %s",
                (level.upper(), limit),
            )
        else:
            rows = cls._fetchall(
                "SELECT * FROM system_logs ORDER BY id DESC LIMIT %s", (limit,)
            )
        return rows

    # ── resource_metrics 操作 ──

    @classmethod
    def add_metric(cls, cpu: float, memory: float, disk: float,
                   disk_free_gb: float, total_images: int = 0) -> None:
        """记录资源监控数据"""
        cls._execute(
            "INSERT INTO resource_metrics (cpu_percent, memory_percent, disk_percent, "
            "disk_free_gb, total_images) VALUES (%s, %s, %s, %s, %s)",
            (cpu, memory, disk, disk_free_gb, total_images),
        )

    @classmethod
    def get_latest_metric(cls) -> Optional[dict]:
        """获取最近一条监控数据"""
        return cls._fetchone(
            "SELECT * FROM resource_metrics ORDER BY id DESC LIMIT 1"
        )


# ── 快速测试 ──
if __name__ == "__main__":
    # 测试连接
    try:
        hashes, count = DB.load_all_hashes()
        print(f"✅ 连接 RDS 成功! image_hashes: {count} 条")

        state = DB.load_pack_state()
        print(f"✅ pack_state: {state['pack_number']} 个包, {len(state['packed_files'])} 个文件")

        DB.log("INFO", "db_utils", "数据库连接测试成功")
        print("✅ system_logs 写入成功")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        DB.close()