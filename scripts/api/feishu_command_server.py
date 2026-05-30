#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞书指令接收服务
通过 ngrok 暴露本地服务，接收飞书机器人消息并执行相应操作
"""

import os
import sys
import json
import time
import hmac
import hashlib
import base64
import threading
from urllib.parse import unquote
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify
import requests

# 添加项目根目录
PROJECT_ROOT = "/Users/caozhaoqi/PycharmProjects/anime_role_detect"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.services.notification_service import get_notification_manager

app = Flask(__name__)

# 配置
CONFIG_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/notification_config.json"
API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305"

# 加载配置
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    notification_config = json.load(f)

# 飞书配置
FEISHU_APP_ID = notification_config["feishu"]["app_id"]
FEISHU_APP_SECRET = notification_config["feishu"]["app_secret"]
FEISHU_VERIFICATION_TOKEN = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")

# 通知管理器
notification_manager = None

# 命令处理函数映射
COMMAND_HANDLERS = {}


def init_notification():
    """初始化通知服务"""
    global notification_manager
    os.environ["NOTIFICATION_ENABLED"] = "true"
    os.environ["NOTIFICATION_PLATFORM"] = notification_config["platform"]
    os.environ["FEISHU_APP_ID"] = FEISHU_APP_ID
    os.environ["FEISHU_APP_SECRET"] = FEISHU_APP_SECRET
    os.environ["FEISHU_RECEIVE_ID"] = notification_config["feishu"]["receive_id"]
    os.environ["FEISHU_RECEIVE_ID_TYPE"] = notification_config["feishu"]["receive_id_type"]
    notification_manager = get_notification_manager()


def verify_feishu_sign(
    encrypt_key: str, msg_signature: str, timestamp: str, nonce: str, content: str
) -> bool:
    """验证飞书签名"""
    sorted_list = sorted([encrypt_key, timestamp, nonce, content])
    sign_str = "".join(sorted_list)
    hashed = hmac.new(sign_str.encode(), digestmod=hashlib.sha256).digest()
    return base64.b64encode(hashed).decode() == msg_signature


def get_feishu_access_token() -> Optional[str]:
    """获取飞书 Access Token"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json"}
    data = {"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        result = response.json()
        if result.get("code") == 0:
            return result.get("tenant_access_token")
    except Exception as e:
        print(f"获取飞书 Access Token 失败: {e}")
    return None


def send_feishu_reply(receive_id: str, msg_type: str, content: str):
    """回复飞书消息"""
    access_token = get_feishu_access_token()
    if not access_token:
        return False

    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
    params = {"receive_id_type": "chat_id"}
    data = {
        "receive_id": receive_id,
        "msg_type": msg_type,
        "content": json.dumps({"text": content}),
    }
    try:
        response = requests.post(url, headers=headers, json=data, params=params, timeout=10)
        return response.json().get("code") == 0
    except Exception as e:
        print(f"回复飞书消息失败: {e}")
        return False


def register_command(cmd: str, handler):
    """注册命令处理器"""
    COMMAND_HANDLERS[cmd] = handler


def parse_command(text: str) -> tuple:
    """解析命令文本，返回 (命令, 参数列表)"""
    parts = text.strip().split()
    if not parts:
        return "", []
    return parts[0].lower(), parts[1:]


def cmd_help(args: list, receive_id: str) -> str:
    """帮助命令"""
    help_text = """🤖 **可用命令列表**

**采集控制:**
`start <角色名>` - 开始采集指定角色
`stop` - 停止当前采集
`status` - 查看当前采集状态
`progress` - 查看采集进度
`list` - 查看所有角色列表

**下载控制:**
`download start` - 开始下载所有URL
`download status` - 查看下载状态
`download stop` - 停止下载
`download pause` - 暂停下载
`download resume` - 恢复下载

**统计查看:**
`stats` - 查看采集统计
`log [行数]` - 查看最近日志 (默认30行)

**示例:**
- `start 纳西妲`
- `download start`
- `stats`
- `log 50`"""
    return help_text


def cmd_start(args: list, receive_id: str) -> str:
    """启动爬虫命令"""
    if not args:
        return "❌ 请指定角色名，例如: `start 纳西妲`"

    role_name = args[0]
    url = f"{API_BASE_URL}/sis/spider_start/single"
    params = {"key_word": role_name}

    try:
        response = requests.post(url, params=params, timeout=30)
        result = response.json()

        if result.get("code") == 0:
            reply_msg = f"✅ 已开始采集角色: **{role_name}**\n请等待采集完成..."
            send_feishu_reply(receive_id, "text", reply_msg)
            return f"已启动采集: {role_name}"
        else:
            return f"❌ 启动失败: {result.get('msg', '未知错误')}"
    except Exception as e:
        return f"❌ 请求失败: {str(e)}"


def cmd_status(args: list, receive_id: str) -> str:
    """查看状态命令"""
    img_url_dir = (
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url"
    )

    spider_info = {"status": "idle", "keyword": "无", "count": 0}
    try:
        response = requests.get(f"{API_BASE_URL}/sis/spider/status", timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                data = result.get("data", {})
                is_running = data.get("is_running", False)
                spider_info = {
                    "status": "running" if is_running else "idle",
                    "keyword": unquote(data.get("current_keyword", "无")),
                    "count": data.get("current_count", 0),
                }
    except:
        pass

    status_text = {
        "running": "🔄 采集中",
        "completed": "✅ 完成",
        "idle": "⏸️ 空闲",
        "error": "❌ 错误",
        "unknown": "❓ 未知",
    }
    display_status = status_text.get(spider_info["status"], f"❓ {spider_info['status']}")

    collected_roles = 0
    total_urls = 0
    try:
        files = [f for f in os.listdir(img_url_dir) if f.endswith("_img.txt")]
        collected_roles = len(files)
        for f in files:
            with open(os.path.join(img_url_dir, f), "r") as fp:
                total_urls += len([l for l in fp if l.strip()])
    except:
        pass

    return f"""📊 **采集状态**

🕷️ **爬虫状态**: {display_status}
👤 **当前角色**: {spider_info['keyword']}
📸 **当前进度**: {spider_info['count']} URLs

📈 **总进度统计**:
- 已采集角色: {collected_roles} 个
- 总 URL 数: {total_urls} 个"""


def cmd_list(args: list, receive_id: str) -> str:
    """查看角色列表"""
    role_list_file = (
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt"
    )
    try:
        with open(role_list_file, "r", encoding="utf-8") as f:
            roles = [line.strip().split()[0] for line in f if line.strip()]
        return (
            f"📋 **角色列表** (共 {len(roles)} 个)\n\n"
            + ", ".join(roles[:20])
            + ("..." if len(roles) > 20 else "")
        )
    except Exception as e:
        return f"❌ 无法读取角色列表: {str(e)}"


def cmd_progress(args: list, receive_id: str) -> str:
    """查看进度命令"""
    img_url_dir = (
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url"
    )
    try:
        files = [f for f in os.listdir(img_url_dir) if f.endswith("_img.txt")]
        total_urls = 0
        for f in files:
            with open(os.path.join(img_url_dir, f), "r") as fp:
                total_urls += len([l for l in fp if l.strip()])
        return f"📈 **采集进度**\n\n- 已采集角色: {len(files)} 个\n- 总 URL 数: {total_urls} 个"
    except Exception as e:
        return f"❌ 无法获取进度: {str(e)}"


def cmd_stop(args: list, receive_id: str) -> str:
    """停止采集命令"""
    try:
        response = requests.post(f"{API_BASE_URL}/sis/spider_stop/stop_all", timeout=10)
        if response.status_code == 200:
            return "🛑 已发送停止命令"
    except:
        pass
    return "❌ 无法停止，请确认服务是否运行"


def cmd_log(args: list, receive_id: str) -> str:
    """查看日志命令"""
    log_dir = (
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src/run/log_dir"
    )
    img_url_dir = (
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url"
    )

    lines = int(args[0]) if args and args[0].isdigit() else 30
    lines = min(lines, 50)

    result_parts = []

    result_parts.append("📋 **系统日志**")
    try:
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        if log_files:
            latest_log = max(log_files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
            log_path = os.path.join(log_dir, latest_log)
            with open(log_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]
            log_content = "".join(recent_lines).replace("|", "│").replace("\n", "\n")
            result_parts.append(f"```\n{log_content}```")
        else:
            result_parts.append("暂无系统日志")
    except Exception as e:
        result_parts.append(f"读取系统日志失败: {str(e)}")

    result_parts.append("\n📸 **采集动态** (最近修改的角色)")
    try:
        files_with_time = []
        for f in os.listdir(img_url_dir):
            if f.endswith("_img.txt"):
                fpath = os.path.join(img_url_dir, f)
                mtime = os.path.getmtime(fpath)
                url_count = len(open(fpath, "r").readlines())
                role_name = f.replace("_img.txt", "")
                files_with_time.append((mtime, role_name, url_count))

        files_with_time.sort(reverse=True)
        for mtime, role_name, url_count in files_with_time[:10]:
            time_str = time.strftime("%m-%d %H:%M", time.localtime(mtime))
            result_parts.append(f"  {time_str} │ {role_name}: {url_count} URLs")
    except Exception as e:
        result_parts.append(f"读取采集动态失败: {str(e)}")

    result_parts.append("\n� **爬虫状态**")
    try:
        response = requests.get(f"{API_BASE_URL}/sis/spider/status", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                data = result.get("data", {})
                is_running = data.get("is_running", False)
                status = "running" if is_running else "idle"
                keyword = unquote(data.get("current_keyword", "无"))
                count = data.get("current_count", 0)

                status_emoji = {"running": "🔄", "completed": "✅", "idle": "⏸️", "error": "❌"}
                emoji = status_emoji.get(status, "❓")
                result_parts.append(f"  {emoji} 状态: {status}")
                result_parts.append(f"  👤 当前: {keyword}")
                result_parts.append(f"  📊 进度: {count} URLs")
        else:
            result_parts.append(f"  ❌ API响应错误: {response.status_code}")
    except Exception as e:
        result_parts.append(f"  ❌ 无法获取状态: {str(e)}")

    return "\n".join(result_parts)


# 下载器实例（全局）
downloader_instance = None
downloader_thread = None


def cmd_download(args: list, receive_id: str) -> str:
    """下载命令"""
    global downloader_instance, downloader_thread

    if not args:
        return """📥 **下载命令帮助**

`download start` - 开始下载所有URL
`download status` - 查看下载状态
`download stop` - 停止下载
`download pause` - 暂停下载
`download resume` - 恢复下载"""

    sub_cmd = args[0].lower()

    if sub_cmd == "start":
        if downloader_instance and downloader_thread and downloader_thread.is_alive():
            return "⚠️ 下载任务已在运行中"

        import threading
        from scripts.data_collection.downloaders.download_all_with_notify import SmartDownloader

        downloader_instance = SmartDownloader(notify=True)
        downloader_thread = threading.Thread(target=downloader_instance.download_all, daemon=True)
        downloader_thread.start()

        return "✅ 下载任务已启动，将通过飞书通知进度"

    elif sub_cmd == "status":
        if not downloader_instance:
            return "❌ 没有正在运行的下载任务"

        stats = downloader_instance.stats
        status = (
            "暂停中"
            if downloader_instance.pause_flag
            else ("运行中" if downloader_thread and downloader_thread.is_alive() else "已停止")
        )

        return f"""📥 **下载状态**

状态: {status}
总URL数: {stats['total']:,}
成功: {stats['success']:,}
失败: {stats['failed']:,}
跳过: {stats['skipped']:,}"""

    elif sub_cmd == "stop":
        if downloader_instance:
            downloader_instance.stop()
            return "🛑 已发送停止信号"
        return "❌ 没有正在运行的下载任务"

    elif sub_cmd == "pause":
        if downloader_instance:
            downloader_instance.pause()
            return "⏸️ 已暂停下载"
        return "❌ 没有正在运行的下载任务"

    elif sub_cmd == "resume":
        if downloader_instance:
            downloader_instance.resume()
            return "▶️ 已恢复下载"
        return "❌ 没有正在运行的下载任务"

    else:
        return f"❌ 未知子命令: {sub_cmd}"


def cmd_stats(args: list, receive_id: str) -> str:
    """统计命令"""
    img_url_dir = (
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url"
    )

    try:
        files = [f for f in os.listdir(img_url_dir) if f.endswith("_img.txt")]

        total_urls = 0
        role_stats = []

        for f in files:
            with open(os.path.join(img_url_dir, f), "r") as fp:
                count = len([l for l in fp if l.strip()])
            role_name = f.replace("_img.txt", "")
            role_stats.append((role_name, count))
            total_urls += count

        role_stats.sort(key=lambda x: x[1], reverse=True)

        sufficient = len([r for r, c in role_stats if c >= 200])
        insufficient = len([r for r, c in role_stats if 100 <= c < 200])
        low = len([r for r, c in role_stats if c < 100])

        msg = f"""📊 **采集统计**

总角色数: {len(files)}
总URL数: {total_urls:,}

✅ 充足 (≥200): {sufficient}
⚠️ 不足 (100-199): {insufficient}
❌ 较少 (<100): {low}

**Top 10 角色:**"""

        for i, (role, count) in enumerate(role_stats[:10], 1):
            msg += f"\n{i}. {role}: {count}"

        return msg

    except Exception as e:
        return f"❌ 统计失败: {str(e)}"


# 注册命令处理器
register_command("help", cmd_help)
register_command("start", cmd_start)
register_command("status", cmd_status)
register_command("list", cmd_list)
register_command("progress", cmd_progress)
register_command("stop", cmd_stop)
register_command("log", cmd_log)
register_command("download", cmd_download)
register_command("stats", cmd_stats)


@app.route("/feishu/webhook", methods=["GET", "POST"])
def feishu_webhook():
    """飞书事件回调接口"""
    print(f"[Webhook] Method: {request.method}, Args: {request.args}")

    # 处理 URL 验证请求 (GET)
    if request.method == "GET":
        challenge = request.args.get("challenge", "")
        if challenge:
            print(f"[验证成功] challenge: {challenge}")
            return jsonify({"challenge": challenge})
        return jsonify({"code": 0, "msg": "ok"})

    # 处理事件回调 (POST)
    data = request.json
    print(f"[POST Data] {data}")

    # 验证请求
    if not data:
        return jsonify({"code": 0})

    # 检查是否是 URL 验证请求 (飞书可能用 POST 验证)
    challenge = data.get("challenge", "")
    if challenge:
        print(f"[POST 验证成功] challenge: {challenge}")
        return jsonify({"challenge": challenge})

    # 处理事件类型
    header = data.get("header", {})
    event_type = header.get("event_type", "")
    event = data.get("event", {})

    print(f"[事件类型] type={event_type}")

    if event_type == "im.message.receive_v1":
        # 接收消息事件
        message = event.get("message", {})
        content = message.get("content", "{}")

        try:
            content_obj = json.loads(content)
            text = content_obj.get("text", "").strip()
        except:
            text = ""

        chat_id = message.get("chat_id", "")
        sender_id = event.get("sender", {}).get("sender_id", {}).get("open_id", "")

        print(f"[收到消息] sender={sender_id}, chat={chat_id}, text={text}")

        if text:
            cmd, args = parse_command(text)
            print(f"[命令解析] cmd={cmd}, args={args}")
            handler = COMMAND_HANDLERS.get(cmd)
            print(f"[处理器] handler={handler}")

            if handler:
                reply_text = handler(args, chat_id)
                print(f"[回复内容] {reply_text[:100] if reply_text else 'None'}...")
                if reply_text:
                    success = send_feishu_reply(chat_id, "text", reply_text)
                    print(f"[回复结果] {'成功' if success else '失败'}")
            else:
                print("[未知命令]")
                send_feishu_reply(
                    chat_id, "text", f"❓ 未知命令: `{cmd}`\n发送 `help` 查看可用命令"
                )

    return jsonify({"code": 0})


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({"status": "ok", "timestamp": time.time()})


def start_server(port: int = 5000):
    """启动 Flask 服务器"""
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="飞书指令接收服务")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    args = parser.parse_args()

    print("=" * 50)
    print("飞书指令接收服务")
    print("=" * 50)
    print("1. 确保 ngrok 正在运行并暴露此服务")
    print("2. 将 ngrok 提供的 URL 配置到飞书应用的事件订阅地址")
    print("3. 配置 Verification Token")
    print(f"4. 服务启动在 http://0.0.0.0:{args.port}")
    print("=" * 50)

    init_notification()
    start_server(args.port)
