@echo off
chcp 65001 >nul

echo 
echo =====================================

echo : Windows 
echo : 
echo.

echo CPU :
wmic cpu get LoadPercentage
echo.
echo :
wmic process where "WorkingSetSize > 1073741824" get Name,ProcessId,WorkingSetSize /format:table
echo.
echo :
echo taskkill /F /PID [ID]

pause