@echo off
cd /d "%~dp0"
echo === CONTAINER STATUS ===
docker compose ps -a
echo.
echo === RECENT SERVICE LOGS ===
docker compose logs --tail=80 backend frontend ollama-model-init postgres
pause
