@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..\..\..
set SUPERVISOR_CONF=%PROJECT_DIR%\supervisord.conf
set LOG_DIR=%PROJECT_DIR%\logs
set RUN_DIR=%PROJECT_DIR%\run
set PID_FILE=%RUN_DIR%\supervisord.pid

set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set NC=[0m

:info
    echo %GREEN%[INFO]%NC% %1
    goto :eof

:warn
    echo %YELLOW%[WARN]%NC% %1
    goto :eof

:error
    echo %RED%[ERROR]%NC% %1
    goto :eof

:create_log_dir
    call :info "..."
    set services=model-service api-service api-gateway multimedia-service search-service inference-worker frontend log-viewer
    for %%s in (%services%) do (
        if not exist "%LOG_DIR%\services\%%s" mkdir "%LOG_DIR%\services\%%s"
    )
    goto :eof

:create_run_dir
    call :info "..."
    if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
    goto :eof

:release_ports
    call :info "..."
    set ports=9001 8000 3000
    for %%p in (%ports%) do (
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p.*LISTENING"') do (
            call :warn " %%p  PID:%%a ..."
            taskkill /F /PID %%a >nul 2>&1
        )
    )
    timeout /t 1 /nobreak >nul
    goto :eof

:start_services
    call :info "..."
    call :create_log_dir
    call :create_run_dir

    where supervisord >nul 2>&1
    if !errorlevel! neq 0 (
        call :error "supervisord : pip install supervisor"
        pause
        exit /b 1
    )

    if not exist "%SUPERVISOR_CONF%" (
        call :error ": %SUPERVISOR_CONF%"
        pause
        exit /b 1
    )

    if exist "%PID_FILE%" (
        set /p pid=<"%PID_FILE%"
        tasklist /FI "PID eq !pid!" | findstr /i supervisord >nul 2>&1
        if !errorlevel! equ 0 (
            call :warn " supervisord  (PID: !pid!)"
            call :info "..."
            supervisorctl -c "%SUPERVISOR_CONF%" restart all
            goto :eof
        )
    )

    call :release_ports

    call :info " supervisord..."
    cd /d "%PROJECT_DIR%"
    supervisord -c "%SUPERVISOR_CONF%"

    call :info "..."
    timeout /t 8 /nobreak >nul

    call :info "..."
    supervisorctl -c "%SUPERVISOR_CONF%" status

    call :info ""
    call :info "Supervisor : http://localhost:9001"
    call :info ": http://localhost:3000"
    goto :eof

:stop_services
    call :info "..."

    if exist "%PID_FILE%" (
        set /p pid=<"%PID_FILE%"
        tasklist /FI "PID eq !pid!" | findstr /i supervisord >nul 2>&1
        if !errorlevel! equ 0 (
            supervisorctl -c "%SUPERVISOR_CONF%" stop all
            supervisorctl -c "%SUPERVISOR_CONF%" shutdown
            timeout /t 2 /nobreak >nul
        ) else (
            call :warn "supervisord "
        )
    ) else (
        call :warn "PID "
    )

    call :release_ports

    if exist "%PID_FILE%" del "%PID_FILE%"

    call :info ""
    goto :eof

:restart_services
    call :info "..."
    call :stop_services
    timeout /t 2 /nobreak >nul
    call :start_services
    goto :eof

:status_services
    call :info "..."
    supervisorctl -c "%SUPERVISOR_CONF%" status
    goto :eof

:view_logs
    if "%1"=="" (
        call :info "..."
        tail -f "%LOG_DIR%\*.log"
    ) else (
        call :info " %1 ..."
        if exist "%LOG_DIR%\%1.log" (
            tail -f "%LOG_DIR%\%1.log"
        ) else if exist "%LOG_DIR%\%1.err.log" (
            tail -f "%LOG_DIR%\%1.err.log"
        ) else (
            call :error ": %1.log"
        )
    )
    goto :eof

:show_help
    echo : %~nx0 ^<command^>
    echo.
    echo :
    echo   start     - 
    echo   stop      - 
    echo   restart   - 
    echo   status    - 
    echo   logs [] - 
    echo   help      - 
    echo.
    echo :
    echo   %~nx0 start
    echo   %~nx0 status
    goto :eof

if "%1"=="" (
    call :show_help
    exit /b 0
)

if /i "%1"=="start" call :start_services
if /i "%1"=="stop" call :stop_services
if /i "%1"=="restart" call :restart_services
if /i "%1"=="status" call :status_services
if /i "%1"=="logs" call :view_logs %2
if /i "%1"=="help" call :show_help
if /i "%1"=="-h" call :show_help
if /i "%1"=="--help" call :show_help

pause