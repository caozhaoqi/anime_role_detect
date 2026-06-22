@echo off
chcp 65001 >nul

echo venvrequirements.txt...

if exist ".venv\Scripts\pip.exe" (
    .venv\Scripts\pip.exe freeze > requirements.txt
) else if exist ".venv\bin\pip.exe" (
    .venv\bin\pip.exe freeze > requirements.txt
) else (
    pip freeze > requirements.txt
)

echo requirements.txt 

pause