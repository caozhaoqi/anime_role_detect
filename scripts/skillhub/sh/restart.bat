@echo off
chcp 65001 >nul

echo ARD Skill Hub 
echo ==========================

echo.
echo ...
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq *uvicorn*" >nul 2>&1
tasklist /FI "IMAGENAME eq python.exe" | findstr /i "uvicorn ardc.api.main" >nul
if !errorlevel! equ 0 (
    for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr "ardc.api.main"') do (
        set pid=%%a
        set pid=!pid:"=!
        taskkill /F /PID !pid! >nul 2>&1
    )
)
timeout /t 2 /nobreak >nul

echo.
echo ...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do (
    echo WARNING  8000 ...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

cd /d "~/czq/anime_role_detect/skillhub"

echo.
echo ...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo WARNING  Python
)

echo.
echo ...
start "ARD Skill Hub" uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4
timeout /t 3 /nobreak >nul

echo.
echo !
echo.

echo ...
curl -s http://localhost:8000/api/health | findstr "healthy" >nul
if !errorlevel! equ 0 (
    echo OK !
    echo.
    echo :
    echo    : 8000
    echo    : 
) else (
    echo ERROR !
)

echo.
echo ==========================
echo 

pause