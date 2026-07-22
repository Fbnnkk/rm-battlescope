@echo off
call conda activate .\.conda\envs\rmuc2026
python -m unittest discover -s tests -v
pause
