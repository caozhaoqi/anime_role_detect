@echo off
chcp 65001 >nul

echo Docker  (Windows)
echo =====================================

echo : 
echo.

set /p "confirm= Docker ? (y/N): "
if /i not "%confirm%"=="y" (
    echo 
    pause
    exit /b 0
)

echo.
echo ...
docker stop $(docker ps -aq)

echo.
echo ...
docker rm $(docker ps -aq)

echo.
echo ...
docker rmi $(docker images -q)

echo.
echo ...
docker system prune -af

echo.
echo Docker 

pause