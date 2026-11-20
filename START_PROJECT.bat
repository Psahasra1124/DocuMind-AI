@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title DocuMind AI - Startup

echo =============================================================
echo                 DOCUMIND AI - STARTUP
echo =============================================================
where docker >nul 2>&1
if errorlevel 1 goto docker_missing

docker info >nul 2>&1
if errorlevel 1 (
  if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
    echo Starting Docker Desktop...
    start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
  ) else (
    echo Docker Desktop is not running. Please open Docker Desktop.
    goto failed
  )
  call :wait_docker
  if errorlevel 1 goto docker_timeout
)

echo.
echo Building and starting PostgreSQL, Ollama, the API, and the web app.
echo First startup downloads images and the Gemma model and may take several minutes.
echo.
docker compose up --build -d
if errorlevel 1 goto compose_failed

echo.
echo Waiting for the web application and AI model initialization...
call :wait_app
if errorlevel 1 goto app_timeout

echo.
echo DocuMind AI is ready.
echo App: http://localhost:8080
echo API docs: http://localhost:8000/docs
start "" http://localhost:8080
goto finish

:wait_docker
for /L %%i in (1,1,36) do (
  docker info >nul 2>&1
  if not errorlevel 1 exit /b 0
  timeout /t 5 /nobreak >nul
)
exit /b 1

:wait_app
for /L %%i in (1,1,90) do (
  curl.exe -fsS -o nul http://localhost:8080 >nul 2>&1
  if not errorlevel 1 exit /b 0
  timeout /t 10 /nobreak >nul
)
exit /b 1

:docker_missing
echo [ERROR] Docker is not installed or not on PATH.
echo Install Docker Desktop for Windows, then restart this script.
goto failed
:docker_timeout
echo [ERROR] Docker Engine did not become ready in 3 minutes.
goto failed
:compose_failed
echo [ERROR] Startup failed. Run CHECK_STATUS.bat and review the logs.
goto failed
:app_timeout
echo [ERROR] The app did not become ready within 15 minutes.
echo The first model download may still be running or a service may have failed.
echo Run CHECK_STATUS.bat and review the logs.
goto failed
:failed
pause
exit /b 1
:finish
pause
exit /b 0
