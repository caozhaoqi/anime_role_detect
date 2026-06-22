@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..
cd /d "%PROJECT_ROOT%"

set OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
set PYTORCH_ENABLE_MPS_FALLBACK=1
set MPS_HIGH_WATERMARK_RATIO=0.0
set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set KMP_DUPLICATE_LIB_OK=TRUE
set ACCELERATE_DISABLED=1
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

echo ...
echo : %PROJECT_ROOT%

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe src\services\search_service\search_service_app.py
) else (
    python src\services\search_service\search_service_app.py
)

pause