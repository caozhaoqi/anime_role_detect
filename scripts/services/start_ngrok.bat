@echo off
chcp 65001 >nul

set NGROK_PORT=5000

echo ==========================================
echo   ngrok 
echo ==========================================
echo : %NGROK_PORT%
echo.

where ngrok >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR ngrok 
    echo  ngrok: https://ngrok.com/download
    pause
    exit /b 1
)

echo  ngrok...
ngrok http %NGROK_PORT% --log=stdout

pause