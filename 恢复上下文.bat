@echo off
echo ============================================
echo   RM BattleScope - 会话上下文
echo ============================================
echo.
echo 告诉 Claude Code 以下信息可快速恢复今天的进度：
echo.
echo 1. 项目在 E:\rm-battlescope
echo 2. Conda 环境: .conda\envs\rmuc2026
echo 3. 数据库: rmuc_2026_region_dataset\rmuc_2026_region_dataset.sqlite
echo 4. 启动服务器: python .\scripts\rmuc_web.py
echo 5. 网址: http://127.0.0.1:8765/
echo.
echo Claude Code 的记忆系统已自动保存了以上信息，
echo 下次打开终端时会自动加载。
echo ============================================
pause
