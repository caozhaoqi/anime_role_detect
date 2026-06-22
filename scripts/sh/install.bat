@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set INSTALL_DIR=%USERPROFILE%\.ardc
set SKILL_DIR=%INSTALL_DIR%\skills
set BIN_DIR=%USERPROFILE%\.local\bin

echo ARDC SkillHub 
echo ==============================

echo.
echo  Python ...
where python >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR Python 
    echo  Python 3.8+ 
    pause
    exit /b 1
)

echo OK Python 

echo.
echo ...
mkdir "%SKILL_DIR%" 2>nul
mkdir "%BIN_DIR%" 2>nul

echo OK 

echo.
echo ...
curl -fsSL -o "%BIN_DIR%\ardc-skill-sync.py" "https://47.79.91.89:8888/api/install/ardc-skill-sync.py"
echo @echo off > "%BIN_DIR%\ardc-skill-sync.bat"
echo python "%BIN_DIR%\ardc-skill-sync.py" %%* >> "%BIN_DIR%\ardc-skill-sync.bat"

echo OK 

echo.
echo ...
if not exist "%INSTALL_DIR%\config.json" (
    echo { > "%INSTALL_DIR%\config.json"
    echo   "skill_hub_url": "http://47.79.91.89:8888", >> "%INSTALL_DIR%\config.json"
    echo   "timeout": 30, >> "%INSTALL_DIR%\config.json"
    echo   "log_level": "INFO", >> "%INSTALL_DIR%\config.json"
    echo   "auto_update": true >> "%INSTALL_DIR%\config.json"
    echo } >> "%INSTALL_DIR%\config.json"
)
echo OK 

echo.
echo ...
pip install requests --quiet
echo OK 

echo.
echo ...
set PATH=%BIN_DIR%;%PATH%
setx PATH "%PATH%"
echo OK 

echo.
echo ==============================
echo 
echo ==============================
echo.
echo :
echo   - : %SKILL_DIR%
echo   - : %BIN_DIR%\ardc-skill-sync.bat
echo   - : %INSTALL_DIR%\config.json
echo.
echo :
echo   :
echo   ardc-skill-sync.bat --help

pause