#!/bin/sh
set -e

case "$1" in
  health-check)
    exec python scripts/monitoring/health_check.py --daemon --interval 60
    ;;
  log-monitor)
    exec python scripts/monitoring/log_monitor.py --daemon --interval 10
    ;;
  log-viewer)
    exec python -m scripts.tools.log_viewer
    ;;
  resource-monitor)
    exec python scripts/monitoring/resource_monitor.py --daemon --interval 30
    ;;
  monitor-dashboard)
    exec python src/run/monitor/monitor_dashboard.py
    ;;
  *)
    echo "Usage: $0 {health-check|log-monitor|log-viewer|resource-monitor|monitor-dashboard}"
    exit 1
    ;;
esac
