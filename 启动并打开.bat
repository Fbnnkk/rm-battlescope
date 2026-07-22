@echo off
start http://127.0.0.1:8765/
call conda activate .\.conda\envs\rmuc2026
python .\scripts\rmuc_web.py
pause
