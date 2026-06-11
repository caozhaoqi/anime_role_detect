# -*- coding: utf-8 -*-
"""
日志快捷搜索工具入口
"""

import threading
import time

from .app import app, HOST, PORT


def _open_browser():
    time.sleep(1.5)
    try:
        import webbrowser
        webbrowser.open(f"http://{HOST}:{PORT}")
    except Exception:
        pass


def main():
    print(f"""
╔══════════════════════════════════════╗
║     📋 Log Viewer - 日志快捷搜索工具    ║
║──────────────────────────────────────║
║  日志目录: /Users/caozhaoqi/PycharmProjects/anime_role_detect/logs
║  访问地址: http://{HOST}:{PORT}
║  自动刷新: 每10秒                    ║
║  Ctrl+C 退出                         ║
╚══════════════════════════════════════╝
    """)
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()