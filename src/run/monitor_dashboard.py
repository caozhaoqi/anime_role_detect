#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务监控仪表板 - 统一查看所有服务状态
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services_config import SERVICES

# 监控配置
MONITOR_PORT = 9000
REFRESH_INTERVAL = 5  # 刷新间隔（秒）


def check_service_health(service_config: dict) -> dict:
    """检查单个服务的健康状态"""
    import requests

    result = {
        "name": service_config["name"],
        "port": service_config["port"],
        "status": "unknown",
        "response_time": 0,
        "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "enabled": service_config.get("enabled", True),
        "is_core": service_config.get("is_core", False),
    }

    if not result["enabled"]:
        result["status"] = "disabled"
        return result

    try:
        health_path = service_config.get("health_path", "/health")
        url = f"http://localhost:{service_config['port']}{health_path}"

        start_time = time.time()
        response = requests.get(url, timeout=2)
        end_time = time.time()

        result["response_time"] = round((end_time - start_time) * 1000, 2)

        if response.status_code == 200:
            result["status"] = "healthy"
        else:
            result["status"] = f"error_{response.status_code}"
    except requests.exceptions.Timeout:
        result["status"] = "timeout"
    except requests.exceptions.ConnectionError:
        result["status"] = "unreachable"
    except Exception as e:
        result["status"] = f"error: {str(e)[:20]}"

    return result


def get_all_services_status() -> List[dict]:
    """获取所有服务的状态"""
    services_status = []

    for key, config in SERVICES.items():
        status = check_service_health(config)
        status["key"] = key
        services_status.append(status)

    return services_status


def generate_html_dashboard(services_status: List[dict]) -> str:
    """生成HTML仪表板"""

    # 统计信息
    total = len(services_status)
    healthy = sum(1 for s in services_status if s["status"] == "healthy")
    enabled = sum(1 for s in services_status if s["enabled"])
    core_services = [s for s in services_status if s["is_core"]]
    core_healthy = sum(1 for s in core_services if s["status"] == "healthy")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>动漫角色识别系统 - 服务监控仪表板</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .stat-card h3 {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .services-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .service-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .service-card:hover {{
            transform: translateY(-5px);
        }}
        
        .service-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .service-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        
        .service-badge {{
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        
        .badge-core {{
            background: #667eea;
            color: white;
        }}
        
        .badge-aux {{
            background: #e0e0e0;
            color: #666;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-healthy {{
            background: #4caf50;
            box-shadow: 0 0 10px #4caf50;
        }}
        
        .status-unreachable {{
            background: #f44336;
            box-shadow: 0 0 10px #f44336;
        }}
        
        .status-timeout {{
            background: #ff9800;
            box-shadow: 0 0 10px #ff9800;
        }}
        
        .status-disabled {{
            background: #9e9e9e;
        }}
        
        .status-unknown {{
            background: #9e9e9e;
        }}
        
        .service-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }}
        
        .info-item {{
            background: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
        }}
        
        .info-label {{
            font-size: 0.8em;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }}
        
        .api-links {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
        }}
        
        .api-link {{
            display: inline-block;
            padding: 8px 15px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-right: 10px;
            font-size: 0.9em;
            transition: background 0.2s;
        }}
        
        .api-link:hover {{
            background: #5568d3;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }}
        
        .refresh-info {{
            text-align: center;
            color: white;
            margin-bottom: 20px;
            opacity: 0.9;
        }}
    </style>
    <script>
        // 自动刷新
        setTimeout(function() {{
            location.reload();
        }}, {REFRESH_INTERVAL * 1000});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 动漫角色识别系统</h1>
            <p>服务监控仪表板</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>核心服务健康</h3>
                <div class="value">{core_healthy}/{len(core_services)}</div>
            </div>
            <div class="stat-card">
                <h3>已启用服务</h3>
                <div class="value">{enabled}/{total}</div>
            </div>
            <div class="stat-card">
                <h3>健康服务</h3>
                <div class="value">{healthy}/{enabled}</div>
            </div>
            <div class="stat-card">
                <h3>总服务数</h3>
                <div class="value">{total}</div>
            </div>
        </div>
        
        <div class="refresh-info">
            <p>⏱️ 自动刷新间隔: {REFRESH_INTERVAL} 秒 | 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="services-grid">
"""

    # 添加服务卡片
    for service in services_status:
        status_class = f"status-{service['status']}"
        badge_class = "badge-core" if service["is_core"] else "badge-aux"
        badge_text = "核心服务" if service["is_core"] else "辅助服务"

        status_text = {
            "healthy": "✅ 运行正常",
            "unreachable": "❌ 无法连接",
            "timeout": "⏰ 超时",
            "disabled": "⏸️ 已禁用",
            "unknown": "❓ 未知",
        }.get(service["status"], f"⚠️ {service['status']}")

        api_url = f"http://localhost:{service['port']}"
        docs_url = f"{api_url}/docs"

        # 只在服务健康运行时显示链接
        if service["status"] == "healthy":
            links_html = f"""
                <div class="api-links">
                    <a href="{api_url}" class="api-link" target="_blank">🔗 访问API</a>
                    <a href="{docs_url}" class="api-link" target="_blank">📚 Swagger文档</a>
                </div>
            """
        else:
            links_html = ""

        html += f"""
            <div class="service-card">
                <div class="service-header">
                    <div>
                        <span class="status-indicator {status_class}"></span>
                        <span class="service-name">{service['name']}</span>
                    </div>
                    <span class="service-badge {badge_class}">{badge_text}</span>
                </div>
                
                <div class="service-info">
                    <div class="info-item">
                        <div class="info-label">状态</div>
                        <div class="info-value">{status_text}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">端口</div>
                        <div class="info-value">{service['port']}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">响应时间</div>
                        <div class="info-value">{service['response_time']} ms</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">最后检查</div>
                        <div class="info-value" style="font-size: 0.9em;">{service['last_check']}</div>
                    </div>
                </div>
                
                {links_html}
            </div>
"""

    html += f"""
        </div>
        
        <div class="footer">
            <p>🚀 动漫角色识别系统 v1.0.0 | 监控仪表板</p>
        </div>
    </div>
</body>
</html>
"""

    return html


def start_monitor_server():
    """启动监控服务器"""
    from flask import Flask, jsonify, render_template_string

    app = Flask(__name__)

    @app.route("/")
    def dashboard():
        """仪表板主页"""
        services_status = get_all_services_status()
        html = generate_html_dashboard(services_status)
        return html

    @app.route("/api/status")
    def api_status():
        """API状态接口"""
        services_status = get_all_services_status()
        return jsonify(
            {"success": True, "timestamp": datetime.now().isoformat(), "services": services_status}
        )

    @app.route("/api/health")
    def health():
        """健康检查"""
        services_status = get_all_services_status()
        all_healthy = all(s["status"] == "healthy" for s in services_status if s["enabled"])
        return jsonify(
            {
                "status": "healthy" if all_healthy else "degraded",
                "timestamp": datetime.now().isoformat(),
            }
        )

    print(f"🚀 启动监控仪表板...")
    print(f"📊 仪表板地址: http://localhost:{MONITOR_PORT}")
    print(f"📡 API状态: http://localhost:{MONITOR_PORT}/api/status")
    print(f"💚 健康检查: http://localhost:{MONITOR_PORT}/api/health")
    print(f"\n按 Ctrl+C 停止监控服务")

    app.run(host="0.0.0.0", port=MONITOR_PORT, debug=False)


if __name__ == "__main__":
    start_monitor_server()
