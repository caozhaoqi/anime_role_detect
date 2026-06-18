@echo off
:: 设置编码为 UTF-8，以支持控制台正常显示 Emoji
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ── 项目根目录 ──
:: %~dp0 是当前脚本所在目录的绝对路径（含尾部斜杠）
:: 假设脚本处于 scripts\data_collection\，向上两级切换至根目录
cd /d "%~dp0..\.."
set "PROJECT_ROOT=%cd%"

echo ╔══════════════════════════════════════════╗
echo ║      📸 一键启动采集任务                  ║
echo ╚══════════════════════════════════════════╝

:: ── 1. 检查 Python ──
set "PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    echo   ✅ 虚拟环境: .venv (Scripts\python.exe)
) else (
    where python3 >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON=python3"
        for /f "tokens=*" %%i in ('python3 --version 2^>^&1') do set "PY_VER=%%i"
        echo   ✅ 系统 Python: !PY_VER!
    ) else (
        where python >nul 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON=python"
            for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
            echo   ✅ 系统 Python: !PY_VER!
        ) else (
            echo   ❌ 未找到 python3 或 python，请先安装 Python 并将其添加到系统 PATH 环境变量。
            exit /b 1
        )
    )
)

:: ── 2. 检查必要依赖 ──
<nul set /p ="  🔍 检查依赖... "
for %%m in (requests oss2) do (
    "!PYTHON!" -c "import %%m" >nul 2>nul
    if !errorlevel! neq 0 (
        echo.
        echo   ⚠️ 缺少 %%m，正在安装...
        "!PYTHON!" -m pip install -q %%m
        if !errorlevel! neq 0 (
            echo   ❌ 安装 %%m 失败
            exit /b 1
        )
    )
)
echo ok

:: ── 3. 检查哈希数据库 ──
if not exist "data\image_hashes.db" (
    echo   ⚠️ 哈希数据库不存在，正在构建...
    "!PYTHON!" scripts\data_collection\build_hash_db.py
)

:: ── 4. 检查飞书配置 ──
set "FEISHU_ARGS="
if exist "scripts\notification_config.json" (
    echo   ✅ 飞书配置: scripts\notification_config.json
) else (
    echo   ⚠️ 飞书配置不存在，将禁用消息推送
    set "FEISHU_ARGS=--no-feishu"
)

:: ── 5. 显示环境信息 ──
set "DB_SIZE=N/A"
if exist "data\image_hashes.db" (
    for %%A in ("data\image_hashes.db") do (
        set /a "size_kb=%%~zA / 1024"
        if !size_kb! gtr 1024 (
            set /a "size_mb=!size_kb! / 1024"
            set "DB_SIZE=!size_mb! MB"
        ) else (
            set "DB_SIZE=!size_kb! KB"
        )
    )
)

echo.
echo   📂 数据集目录: data\final_dataset
echo   📄 哈希数据库: data\image_hashes.db (!DB_SIZE!)
echo   📝 采集脚本: scripts\data_collection\collect_from_keywords.py
echo.

:: ── 6. 启动采集 ──
echo   🚀 启动采集...
echo ─────────────────────────────────────────

:: 使用 %* 将所有输入给 bat 的参数透传给底层 Python 运行脚本
"!PYTHON!" scripts\data_collection\collector_runner.py %* !FEISHU_ARGS!

endlocal