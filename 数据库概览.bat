@echo off
call conda activate .\.conda\envs\rmuc2026
python .\scripts\rmuc_sqlite.py summary
pause
