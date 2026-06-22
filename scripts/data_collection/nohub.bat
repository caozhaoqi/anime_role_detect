@echo off
chcp 65001 >nul

echo ...
echo.
echo ...
pause >nul

start "" cmd /k "%~dp0start_collector.bat %*"

echo 
echo 