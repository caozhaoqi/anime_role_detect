#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
exec sudo python3 "$PROJECT_DIR/spider_image_system/src/run/sis_main_process.py"