@echo off
cd /d "%~dp0"
echo Stopping containers. Persistent data will be preserved.
docker compose down
pause
