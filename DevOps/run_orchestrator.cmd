@echo off
setlocal enabledelayedexpansion

REM Change working directory to project root (parent of this script directory)
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.." >NUL 2>&1

REM Prefer the project's virtualenv Python if available
set "VENV_PY=RAG Files\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  "%VENV_PY%" "DevOps\orchestrate_build_and_smoke.py" %*
) else (
  echo [WARN] Virtual env Python not found at "%CD%\%VENV_PY%". Falling back to system Python on PATH...
  python "DevOps\orchestrate_build_and_smoke.py" %*
)

set "EXITCODE=%ERRORLEVEL%"
popd >NUL 2>&1
exit /b %EXITCODE%
