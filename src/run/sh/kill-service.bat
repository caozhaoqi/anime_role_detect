@echo off
chcp 65001 >nul

echo ===  ===

set ports=9001 8000 3000

echo [INFO] ...
for %%p in (%ports%) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p.*LISTENING"') do (
        echo [WARN]  %%p  PID:%%a ...
        taskkill /F /PID %%a >nul 2>&1
    )
)

echo [INFO] ...
tasklist /FI "IMAGENAME eq python.exe" | findstr /i "supervisord" >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr "supervisord"') do (
        set pid=%%a
        set pid=!pid:"=!
        echo [WARN]  supervisord  PID:!pid!...
        taskkill /F /PID !pid! >nul 2>&1
    )
)

echo ===  ===

pause