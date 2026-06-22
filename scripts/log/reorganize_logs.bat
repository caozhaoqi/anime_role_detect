@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..
set LOG_DIR=%PROJECT_ROOT%\logs

echo ==========================================
echo 
echo ==========================================
echo.

if not exist "%LOG_DIR%" (
    echo ERROR: logs
    pause
    exit /b 1
)

echo : %LOG_DIR%
echo.

set /p "confirm=? (y/n): "
if /i not "%confirm%"=="y" (
    echo 
    pause
    exit /b 0
)

for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set today=%%a%%b%%c
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set now=%%a%%b
set BACKUP_DIR=%LOG_DIR%\backup_%today%_%now%

echo : %BACKUP_DIR%
mkdir "%BACKUP_DIR%"
xcopy "%LOG_DIR%\*" "%BACKUP_DIR%\" /E /H /Y >nul 2>&1
echo OK 
echo.

echo ...

mkdir "%LOG_DIR%\services\api-service" 2>nul
mkdir "%LOG_DIR%\services\model-service" 2>nul
mkdir "%LOG_DIR%\services\api-gateway" 2>nul
mkdir "%LOG_DIR%\services\multimedia-service" 2>nul
mkdir "%LOG_DIR%\services\search-service" 2>nul
mkdir "%LOG_DIR%\services\inference-worker" 2>nul
mkdir "%LOG_DIR%\services\frontend" 2>nul
mkdir "%LOG_DIR%\services\monitoring" 2>nul
mkdir "%LOG_DIR%\functional\health_check" 2>nul
mkdir "%LOG_DIR%\functional\inference" 2>nul
mkdir "%LOG_DIR%\functional\training" 2>nul
mkdir "%LOG_DIR%\functional\system" 2>nul
mkdir "%LOG_DIR%\functional\download" 2>nul
mkdir "%LOG_DIR%\functional\error" 2>nul
mkdir "%LOG_DIR%\archive\compressed" 2>nul

echo OK 
echo.

echo ...

set move_list=^
api-service.log services\api-service^
api-service.err.log services\api-service^
model-service.log services\model-service^
model-service.err.log services\model-service^
api-gateway.log services\api-gateway^
api-gateway.err.log services\api-gateway^
multimedia-service.log services\multimedia-service^
multimedia-service.err.log services\multimedia-service^
search-service.log services\search-service^
search-service.err.log services\search-service^
search-worker.log services\search-service^
search-worker.err.log services\search-service^
inference-worker.log services\inference-worker^
inference-worker.err.log services\inference-worker^
frontend.log services\frontend^
frontend.err.log services\frontend^
monitor-dashboard.log services\monitoring^
monitor-dashboard.err.log services\monitoring^
health-check.log services\monitoring^
health-check.err.log services\monitoring^
log-monitor.log services\monitoring^
log-monitor.err.log services\monitoring^
resource-monitor.log services\monitoring^
resource-monitor.err.log services\monitoring

setlocal enabledelayedexpansion
set "list=%move_list%"
:move_loop
for /f "tokens=1-2" %%a in ("%list%") do (
    if exist "%LOG_DIR%\%%a" (
        move "%LOG_DIR%\%%a" "%LOG_DIR%\%%b\" >nul
        echo   OK %%a -^> %%b\
    )
    set "list=!list:*%%a %%b=!"
    if not "!list!"=="" goto move_loop
)
endlocal

echo.
echo ...

if exist "%LOG_DIR%\health_check" (
    xcopy "%LOG_DIR%\health_check\*" "%LOG_DIR%\functional\health_check\" /E /H /Y >nul
    rmdir "%LOG_DIR%\health_check"
    echo   OK health_check/ -^> functional/health_check/
)

if exist "%LOG_DIR%\inference" (
    xcopy "%LOG_DIR%\inference\*" "%LOG_DIR%\functional\inference\" /E /H /Y >nul
    rmdir "%LOG_DIR%\inference"
    echo   OK inference/ -^> functional/inference/
)

if exist "%LOG_DIR%\training" (
    xcopy "%LOG_DIR%\training\*" "%LOG_DIR%\functional\training\" /E /H /Y >nul
    rmdir "%LOG_DIR%\training"
    echo   OK training/ -^> functional/training/
)

if exist "%LOG_DIR%\system" (
    xcopy "%LOG_DIR%\system\*" "%LOG_DIR%\functional\system\" /E /H /Y >nul
    rmdir "%LOG_DIR%\system"
    echo   OK system/ -^> functional/system/
)

if exist "%LOG_DIR%\download" (
    xcopy "%LOG_DIR%\download\*" "%LOG_DIR%\functional\download\" /E /H /Y >nul
    rmdir "%LOG_DIR%\download"
    echo   OK download/ -^> functional/download/
)

if exist "%LOG_DIR%\error" (
    xcopy "%LOG_DIR%\error\*" "%LOG_DIR%\functional\error\" /E /H /Y >nul
    rmdir "%LOG_DIR%\error"
    echo   OK error/ -^> functional/error/
)

echo.
echo :
for %%f in (supervisord.log unified.log redis.log github_action.log) do (
    if exist "%LOG_DIR%\%%f" echo   OK %%f
)

echo.
echo ==========================================
echo !
echo ==========================================
echo.
echo :
echo logs/
echo ^|-- services/              # 
echo ^|   ^|-- api-service/
echo ^|   ^|-- model-service/
echo ^|   ^|-- api-gateway/
echo ^|   ^|-- multimedia-service/
echo ^|   ^|-- search-service/
echo ^|   ^|-- inference-worker/
echo ^|   ^|-- frontend/
echo ^|   ^|-- monitoring/
echo ^|-- functional/            # 
echo ^|   ^|-- health_check/
echo ^|   ^|-- inference/
echo ^|   ^|-- training/
echo ^|   ^|-- system/
echo ^|   ^|-- download/
echo ^|   ^|-- error/
echo ^|-- archive/               # 
echo ^|   ^|-- compressed/
echo ^|-- backup_*/              # ()
echo ^|-- supervisord.log
echo ^|-- unified.log
echo ^|-- redis.log
echo.

pause