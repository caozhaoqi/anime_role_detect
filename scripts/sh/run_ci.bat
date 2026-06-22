@echo off
chcp 65001 >nul

echo ======================================
echo    CI/CD 
echo ======================================
echo.

echo [1/4] 
echo : %cd%
echo Git :
git status --porcelain | head -5
echo.

echo [2/4]  Python 
python --version
echo.

echo [3/4] 
echo ...

set packages=torch fastapi onnxruntime celery redis prometheus_client
for %%p in (%packages%) do (
    python -c "import %%p; print('OK %%p')" 2>nul || echo ERROR %%p: 
)
echo.

echo [4/4] API 
echo ...
curl -s http://localhost:8000/api/v1/onnx/models >nul 2>&1
if !errorlevel! equ 0 (
    echo OK API 
    
    echo.
    echo ONNX :
    curl -s http://localhost:8000/api/v1/onnx/models | python -m json.tool
    
    echo.
    echo :
    curl -s http://localhost:8000/api/health | python -m json.tool 2>nul || echo 
) else (
    echo ERROR API 
    echo    : python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
)

echo.
echo ======================================
echo   CI/CD 
echo ======================================

pause