#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式采集系统 - 服务端
负责执行采集任务、打包数据、提供API
"""

import os
import sys
import json
import time
import shutil
import sqlite3
import zipfile
import tempfile
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from flask import Flask, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = Flask(__name__)

# 配置
DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
OUTPUT_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/packages")
DB_PATH = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/packages.db")
CONFIG_PATH = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/notification_config.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据库连接
db_conn = None


def init_database():
    """初始化数据库"""
    global db_conn
    db_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cursor = db_conn.cursor()

    # 数据包记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS packages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            package_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            file_size INTEGER,
            total_chars INTEGER,
            total_images INTEGER,
            md5_hash TEXT,
            download_count INTEGER DEFAULT 0,
            last_download_time TEXT,
            created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 客户端下载记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS client_downloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            package_name TEXT NOT NULL,
            download_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN DEFAULT 1,
            UNIQUE(client_id, package_name)
        )
    ''')

    db_conn.commit()


def get_db():
    """获取数据库连接"""
    global db_conn
    if db_conn is None:
        init_database()
    return db_conn


def record_package(package_name, created_at, file_size, stats):
    """记录数据包"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO packages
        (package_name, created_at, file_size, total_chars, total_images)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        package_name,
        created_at,
        file_size,
        stats.get('total_chars', 0),
        stats.get('total_images', 0)
    ))

    conn.commit()


def get_package_record(package_name):
    """获取数据包记录"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM packages WHERE package_name = ?
    ''', (package_name,))

    row = cursor.fetchone()
    if row:
        return {
            'id': row[0],
            'package_name': row[1],
            'created_at': row[2],
            'file_size': row[3],
            'total_chars': row[4],
            'total_images': row[5],
            'md5_hash': row[6],
            'download_count': row[7],
            'last_download_time': row[8]
        }
    return None


def record_client_download(client_id, package_name, success=True):
    """记录客户端下载"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO client_downloads
        (client_id, package_name, success)
        VALUES (?, ?, ?)
    ''', (client_id, package_name, success))

    # 更新包的下载计数
    cursor.execute('''
        UPDATE packages
        SET download_count = download_count + 1,
            last_download_time = ?
        WHERE package_name = ?
    ''', (datetime.now().isoformat(), package_name))

    conn.commit()


def has_client_downloaded(client_id, package_name):
    """检查客户端是否已下载"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT COUNT(*) FROM client_downloads
        WHERE client_id = ? AND package_name = ? AND success = 1
    ''', (client_id, package_name))

    count = cursor.fetchone()[0]
    return count > 0


def get_client_downloaded_packages(client_id):
    """获取客户端已下载的数据包列表"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT package_name, download_time
        FROM client_downloads
        WHERE client_id = ? AND success = 1
        ORDER BY download_time DESC
    ''', (client_id,))

    return [{'package_name': row[0], 'download_time': row[1]} for row in cursor.fetchall()]

# 任务状态
task_status = {
    "status": "idle",  # idle, running, completed, error
    "start_time": None,
    "end_time": None,
    "characters_processed": 0,
    "total_characters": 0,
    "images_collected": 0,
    "current_character": None,
    "error": None,
    "pid": None
}

# 进程引用
collector_process = None


def get_dataset_stats():
    """获取数据集统计信息"""
    character_stats = defaultdict(lambda: {"jpg": 0, "png": 0})

    if not DATA_DIR.exists():
        return None

    for char_dir in DATA_DIR.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            for img_file in char_dir.iterdir():
                if img_file.is_file():
                    ext = img_file.suffix.lower()
                    if ext in [".jpg", ".jpeg"]:
                        character_stats[char_name]["jpg"] += 1
                    elif ext == ".png":
                        character_stats[char_name]["png"] += 1

    total_chars = len(character_stats)
    total_images = sum(s["jpg"] + s["png"] for s in character_stats.values())
    total_jpg = sum(s["jpg"] for s in character_stats.values())
    total_png = sum(s["png"] for s in character_stats.values())

    # 图片数分布
    distribution = defaultdict(int)
    for char_name, stats in character_stats.items():
        count = stats["jpg"] + stats["png"]
        if count >= 100:
            distribution["100+"] += 1
        elif count >= 50:
            distribution["50-99"] += 1
        elif count >= 30:
            distribution["30-49"] += 1
        elif count >= 10:
            distribution["10-29"] += 1
        else:
            distribution["0-9"] += 1

    # TOP15
    sorted_chars = sorted(character_stats.items(),
                          key=lambda x: x[1]["jpg"] + x[1]["png"],
                          reverse=True)[:15]

    return {
        "total_chars": total_chars,
        "total_images": total_images,
        "total_jpg": total_jpg,
        "total_png": total_png,
        "distribution": dict(distribution),
        "top_chars": [(name, stats) for name, stats in sorted_chars],
        "timestamp": datetime.now().isoformat()
    }


def create_package():
    """创建数据包"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"dataset_{timestamp}.zip"
    package_path = OUTPUT_DIR / package_name

    # 获取最新统计数据
    stats = get_dataset_stats()
    if not stats or stats["total_images"] == 0:
        return None, None

    # 创建zip包
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for char_dir in DATA_DIR.iterdir():
            if char_dir.is_dir():
                char_name = char_dir.name
                for img_file in char_dir.iterdir():
                    if img_file.is_file():
                        arcname = f"{char_name}/{img_file.name}"
                        zf.write(img_file, arcname)

    # 获取文件大小
    file_size = package_path.stat().st_size

    # 创建元数据文件
    meta = {
        "package_name": package_name,
        "created_at": timestamp,
        "stats": stats,
        "file_size": file_size
    }
    meta_path = OUTPUT_DIR / f"{package_name}.meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 记录到数据库
    record_package(package_name, timestamp, file_size, stats)

    # 更新包列表
    update_package_list()

    return package_path, meta


def update_package_list():
    """更新包列表"""
    packages = []
    for pkg_file in sorted(OUTPUT_DIR.glob("dataset_*.zip"), reverse=True):
        meta_file = OUTPUT_DIR / f"{pkg_file.name}.meta.json"
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            packages.append({
                "name": meta["package_name"],
                "created_at": meta["created_at"],
                "size": pkg_file.stat().st_size,
                "stats": meta["stats"]
            })
    # 只保留最新10个包
    packages = packages[:10]

    list_file = OUTPUT_DIR / "package_list.json"
    with open(list_file, 'w', encoding='utf-8') as f:
        json.dump(packages, f, ensure_ascii=False, indent=2)

    return packages


def start_collector():
    """启动采集脚本"""
    global collector_process, task_status

    if task_status["status"] == "running":
        return False, "采集任务已在运行中"

    COLLECTOR_SCRIPT = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/spider_image_system/src/danbooru/multi_site_enhanced_collector.py"
    KEYWORDS_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/auto_spider_img/keywords"

    cmd = [
        "python3", COLLECTOR_SCRIPT,
        "--input-dir", KEYWORDS_DIR,
        "--output-dir", str(DATA_DIR),
        "--target-count", "100"
    ]

    try:
        collector_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )

        task_status["status"] = "running"
        task_status["start_time"] = datetime.now().isoformat()
        task_status["pid"] = collector_process.pid
        task_status["error"] = None

        # 启动状态监控线程
        threading.Thread(target=monitor_collector, daemon=True).start()

        return True, f"采集任务已启动 (PID: {collector_process.pid})"
    except Exception as e:
        task_status["status"] = "error"
        task_status["error"] = str(e)
        return False, str(e)


def stop_collector():
    """停止采集脚本"""
    global collector_process, task_status

    if collector_process:
        collector_process.terminate()
        collector_process = None

    task_status["status"] = "idle"
    task_status["end_time"] = datetime.now().isoformat()

    return True, "采集任务已停止"


def monitor_collector():
    """监控采集进程"""
    global collector_process, task_status

    while collector_process and collector_process.poll() is None:
        time.sleep(30)

        # 更新统计
        stats = get_dataset_stats()
        if stats:
            task_status["total_characters"] = stats["total_chars"]
            task_status["images_collected"] = stats["total_images"]

    # 进程结束
    if collector_process:
        retcode = collector_process.returncode
        if retcode == 0:
            task_status["status"] = "completed"
        else:
            task_status["status"] = "error"
            task_status["error"] = f"进程异常退出 (返回码: {retcode})"

        task_status["end_time"] = datetime.now().isoformat()

        # 自动创建新包
        if task_status["status"] == "completed":
            create_package()


# ==================== API 路由 ====================

@app.route("/api/status", methods=["GET"])
def get_status():
    """获取采集状态"""
    stats = get_dataset_stats()
    return jsonify({
        "task": task_status,
        "dataset": stats
    })


@app.route("/api/start", methods=["POST"])
def start_task():
    """启动采集任务"""
    success, message = start_collector()
    return jsonify({"success": success, "message": message})


@app.route("/api/stop", methods=["POST"])
def stop_task():
    """停止采集任务"""
    success, message = stop_collector()
    return jsonify({"success": success, "message": message})


@app.route("/api/package/list", methods=["GET"])
def list_packages():
    """获取数据包列表"""
    list_file = OUTPUT_DIR / "package_list.json"
    if list_file.exists():
        with open(list_file, 'r', encoding='utf-8') as f:
            packages = json.load(f)
    else:
        packages = []
    return jsonify({"packages": packages})


@app.route("/api/package/latest", methods=["GET"])
def get_latest_package():
    """获取最新数据包"""
    packages = update_package_list()
    if packages:
        return jsonify({"package": packages[0]})
    return jsonify({"package": None})


@app.route("/api/package/<filename>", methods=["GET"])
def download_package(filename):
    """下载数据包"""
    # 获取客户端ID（从请求参数或header）
    client_id = request.args.get('client_id') or request.headers.get('X-Client-ID')

    package_path = OUTPUT_DIR / filename
    if not package_path.exists():
        return jsonify({"error": "数据包不存在"}), 404

    # 检查客户端是否已下载过
    if client_id and has_client_downloaded(client_id, filename):
        return jsonify({
            "error": "已下载过此数据包",
            "package": filename,
            "client_id": client_id
        }), 409

    # 记录下载
    if client_id:
        record_client_download(client_id, filename)

    return send_file(
        package_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name=filename
    )


@app.route("/api/client/packages", methods=["GET"])
def get_client_packages():
    """获取客户端已下载的数据包列表"""
    client_id = request.args.get('client_id') or request.headers.get('X-Client-ID')
    if not client_id:
        return jsonify({"error": "缺少client_id"}), 400

    downloaded = get_client_downloaded_packages(client_id)
    return jsonify({"client_id": client_id, "downloaded": downloaded})


@app.route("/api/package/create", methods=["POST"])
def manual_create_package():
    """手动创建数据包"""
    package_path, meta = create_package()
    if package_path:
        return jsonify({
            "success": True,
            "package": meta
        })
    return jsonify({"success": False, "error": "无数据可打包"})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """获取详细统计"""
    stats = get_dataset_stats()
    return jsonify(stats)


@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='分布式采集服务端')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=5000, help='监听端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    # 初始化数据库
    init_database()

    print("=" * 60)
    print("分布式采集系统 - 服务端")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"数据库: {DB_PATH}")
    print(f"API地址: http://{args.host}:{args.port}")
    print("=" * 60)
    print("\n可用API:")
    print("  GET  /api/status       - 获取采集状态")
    print("  POST /api/start        - 启动采集")
    print("  POST /api/stop         - 停止采集")
    print("  GET  /api/package/list - 获取数据包列表")
    print("  GET  /api/package/latest - 获取最新数据包")
    print("  GET  /api/package/<filename>?client_id=<id> - 下载数据包")
    print("  POST /api/package/create - 手动创建数据包")
    print("  GET  /api/client/packages?client_id=<id> - 获取客户端已下载列表")
    print("  GET  /api/stats        - 获取统计信息")
    print("  GET  /api/health       - 健康检查")
    print("=" * 60)
    print("\n注意:")
    print("  - 下载接口需要传递 client_id 参数避免重复下载")
    print("  - client_id 可通过 URL参数或 Header(X-Client-ID)传递")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
