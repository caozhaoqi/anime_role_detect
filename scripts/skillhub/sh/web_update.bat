@echo off
chcp 65001 >nul

echo 
echo ==============================

echo.
echo ...
npm run build

echo.
echo ...
echo : Windows 
echo 

xcopy /E /H /Y web\dist\* "C:\inetpub\wwwroot\ardc-web\"

echo.
echo 

pause