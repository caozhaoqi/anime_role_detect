@echo off
setlocal enabledelayedexpansion

set "ARD_INSTALL_DIR=%USERPROFILE%\.ardc"
set "ARD_BIN_DIR=%ARD_INSTALL_DIR%\bin"
set "ARD_CLI_URL=http://47.79.91.89:8888/api/install/cli.py"

echo ==============================================
echo        ARD Skill Hub CLI 安装程序
echo ==============================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到 Python
    echo 请先安装 Python 3.8 或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查 Python 版本
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%v"
echo ✅ 检测到 Python 版本: %PYTHON_VERSION%

REM 创建安装目录
echo.
echo 📁 创建安装目录...
if not exist "%ARD_INSTALL_DIR%" mkdir "%ARD_INSTALL_DIR%"
if not exist "%ARD_BIN_DIR%" mkdir "%ARD_BIN_DIR%"

REM 下载 CLI 工具
echo.
echo 📥 下载 CLI 工具...
powershell -Command "Invoke-WebRequest -Uri '%ARD_CLI_URL%' -OutFile '%ARD_BIN_DIR%\ardc.py'"
if %errorlevel% neq 0 (
    echo ❌ 下载失败，请检查网络连接
    pause
    exit /b 1
)

REM 创建批处理包装器
echo @echo off > "%ARD_BIN_DIR%\ardc.bat"
echo python "%ARD_BIN_DIR%\ardc.py" %%* >> "%ARD_BIN_DIR%\ardc.bat"

echo ✅ CLI 工具下载完成

REM 添加到 PATH
echo.
echo 🔗 配置系统环境变量...
set "CURRENT_PATH=%PATH%"
echo %CURRENT_PATH% | find /i "%ARD_BIN_DIR%" >nul
if %errorlevel% neq 0 (
    setx PATH "%PATH%;%ARD_BIN_DIR%"
    echo ✅ 已将 %ARD_BIN_DIR% 添加到 PATH
) else (
    echo ✅ PATH 已配置
)

REM 安装 Python 依赖
echo.
echo 📦 安装 Python 依赖...
pip install requests pydantic fastapi uvicorn >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 依赖安装成功
) else (
    echo ⚠️  警告: 部分依赖安装失败
    echo 请手动运行: pip install requests pydantic fastapi uvicorn
)

REM 创建配置文件
echo.
echo 📄 创建配置文件...
echo { > "%ARD_INSTALL_DIR%\config.json"
echo   "api_url": "http://47.79.91.89:8888/api", >> "%ARD_INSTALL_DIR%\config.json"
echo   "install_dir": "%ARD_INSTALL_DIR%", >> "%ARD_INSTALL_DIR%\config.json"
echo   "version": "1.0.0", >> "%ARD_INSTALL_DIR%\config.json"
echo   "installed_at": "%date:~0,4%-%date:~5,2%-%date:~8,2%T%time:~0,2%:%time:~3,2%:%time:~6,2%Z" >> "%ARD_INSTALL_DIR%\config.json"
echo } >> "%ARD_INSTALL_DIR%\config.json"
echo ✅ 配置文件已创建

REM 显示安装信息
echo.
echo ==============================================
echo          ✅ ARD Skill Hub CLI 安装完成！
echo ==============================================
echo.
echo 📍 安装位置: %ARD_BIN_DIR%
echo 📄 配置文件: %ARD_INSTALL_DIR%\config.json
echo.
echo 📝 使用方法:
echo   1. 重启命令行终端
echo.
echo   2. 验证安装:
echo      ardc --version
echo.
echo   3. 查看帮助:
echo      ardc --help
echo.
echo   4. 列出所有技能:
echo      ardc skill list
echo.
echo   5. 搜索技能:
echo      ardc skill search 关键词
echo.
echo   6. 安装技能:
echo      ardc skill install 技能ID
echo.
echo 🔗 更多信息请访问: http://47.79.91.89:8888
echo.
pause