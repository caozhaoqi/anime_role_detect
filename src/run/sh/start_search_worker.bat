@echo off
chcp 65001 >nul

set PROJECT_DIR=/Users/caozhaoqi/PycharmProjects/anime_role_detect
set PYTHONPATH=%PROJECT_DIR%;%PROJECT_DIR%\.venv\Lib\site-packages

if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    "%PROJECT_DIR%\.venv\Scripts\python.exe" src\services\search_service\search_worker.py
) else (
    python src\services\search_service\search_worker.py
)