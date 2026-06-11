# -*- coding: utf-8 -*-
"""
Flask Web服务 - API路由
"""

from pathlib import Path

import flask

from .engine import get_cached_logs, filter_logs, get_stats, tail_logs

# ============ 配置 ============
HOST = "127.0.0.1"
PORT = 58888
# ==============================

HERE = Path(__file__).parent
TEMPLATE_PATH = HERE / "templates" / "index.html"

app = flask.Flask(__name__)


def _load_template():
    """加载HTML模板"""
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text("utf-8")
    return "<html><body><h1>Template not found</h1></body></html>"


@app.route("/")
def index():
    return flask.render_template_string(_load_template())


@app.route("/api/stats")
def api_stats():
    force = flask.request.args.get("force", "").lower() == "true"
    logs = get_cached_logs(force_refresh=force)
    return flask.jsonify(get_stats(logs))


@app.route("/api/services")
def api_services():
    """获取所有服务名列表"""
    logs = get_cached_logs()
    services = set()
    for entry in logs:
        svc = entry.get("_service", "")
        if svc:
            services.add(svc)
    return flask.jsonify(sorted(services))


@app.route("/api/search")
def api_search():
    keyword = flask.request.args.get("q", "")
    level = flask.request.args.get("level", "")
    service = flask.request.args.get("service", "")
    date_from = flask.request.args.get("from", "")
    date_to = flask.request.args.get("to", "")
    sort_order = flask.request.args.get("sort", "desc")
    offset = int(flask.request.args.get("offset", 0))
    limit = int(flask.request.args.get("limit", 100))

    logs = get_cached_logs()
    results, total = filter_logs(
        logs, keyword=keyword, level=level, service=service,
        date_from=date_from, date_to=date_to, sort_order=sort_order,
        offset=offset, limit=limit,
    )
    return flask.jsonify({"total": total, "offset": offset, "limit": limit, "results": results})


@app.route("/api/tail")
def api_tail():
    lines = int(flask.request.args.get("lines", 50))
    results = tail_logs(lines)
    return flask.jsonify({"total": len(results), "results": results})