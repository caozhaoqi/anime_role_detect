@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion


set -e

SCRIPT_DIR="$(cd \d "$(dirname "%0%")" && pwd)"
PROJECT_ROOT="$(cd \d "%SCRIPT_DIR%\..\.." && pwd)"
cd \d "%PROJECT_ROOT%"

echo
echo
echo

PYTHON=""
if IF IF EXIST "".venv\Scripts\python.exe"";
PYTHON=".venv\Scripts\python.exe"
echo : .venv
elif command -v python &>\dev\null;
PYTHON="python"
echo Python: $(python --version 2>&1)
else
echo python
exit 1

echo ...
DEP_MISSING=0
FOR %%1 IN (requests oss2) DO
if ! "%PYTHON%" -c "import %mod%" 2>\dev\null;
echo
echo %mod%...
"%PYTHON%" -m pip install -q "%mod%" || {
echo %mod%
exit 1
}
echo ok

if IF ! IF EXIST ""data\image_hashes.db"";
echo ...
"%PYTHON%" scripts\data_collection\build_hash_db.py

FEISHU_ARGS=""
if IF IF EXIST ""scripts\notification_config.json"";
echo : scripts\notification_config.json
else
echo
FEISHU_ARGS="--no-feishu"

echo
echo : data\final_dataset
echo : data\image_hashes.db ($(IF IF EXIST "data\image_hashes.db"&& du -h data\image_hashes.db | cut -f1 || echo N\A))
echo : scripts\data_collection\collect_from_keywords.py
echo

echo ...
echo
exec "%PYTHON%" scripts\data_collection\collector_runner.py "$@" %FEISHU_ARGS%

pause