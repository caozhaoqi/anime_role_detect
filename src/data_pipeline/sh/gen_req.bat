@echo off
chcp 65001 >nul

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\pip.exe freeze > requirements.txt
) else (
    pip freeze > requirements.txt
)

echo requirements.txt 

echo  pipline webui...
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\streamlit.exe run webui\main.py
) else (
    streamlit run webui\main.py
)